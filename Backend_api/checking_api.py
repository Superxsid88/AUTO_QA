import os
# ---- Force torchaudio to use libsndfile BEFORE importing torchaudio ----
os.environ["TORCHAUDIO_USE_SOUNDFILE"] = "1"

from dotenv import load_dotenv
load_dotenv()  # make OPENAI_API_KEY available before any module uses it

import io
import re
import json
import uuid
import asyncio
import logging
import traceback
from datetime import timedelta
from typing import Dict, List, Union, Optional

import numpy as np
import soundfile as sf

import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")  # ensure soundfile backend
import torchaudio.transforms as T

from transformers import (
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
    StoppingCriteria,
    LogitsProcessor,
)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from load_model import load_whisper_model  # OpenAI shim or local HF whisper
from sentiment_summarization import unified_agent_evaluation, detect_sentiment
from scoring import tataScoring

# ================== CONFIGURATION ==================
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock for thread-safe transcription
transcription_lock = asyncio.Lock()

# ================== FASTAPI APP ==================
app = FastAPI(title="Audio Transcription API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== AUDIO LOADING (soundfile) ==================
def load_audio_with_soundfile(file_like: io.BytesIO):
    """
    Reads audio with soundfile and returns a torch.Tensor shaped [channels, samples], sr.
    """
    file_like.seek(0)
    data, sr = sf.read(file_like, always_2d=True, dtype="float32")  # [samples, channels]
    data = np.transpose(data, (1, 0))  # -> [channels, samples]
    audio = torch.from_numpy(data)     # torch.float32, cpu
    return audio, sr

# ================== MODEL LOADING ==================
# Will be either:
#   OpenAI: (WhisperShim, None)
#   Local : (WhisperForConditionalGeneration, WhisperProcessor)
whisper_model = None
whisper_processor = None

# ================== SILERO VAD ==================
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True,
    force_reload=False
)
vad_model.to('cpu')
(get_speech_timestamps, _, _, _, _) = vad_utils

# ================== CUSTOM CLASSES ==================
class MinLenSafe(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = int(min_length)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores

class StopOnSegmentEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        # if tokenizer does not have this special token (local), we still keep logic safe
        try:
            self.segment_end_id = tokenizer.convert_tokens_to_ids("<|segment_end|>")
        except Exception:
            self.segment_end_id = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.segment_end_id is None:
            return False
        return self.segment_end_id in input_ids[0].tolist()

# ================== UTILITY FUNCTIONS ==================
def extract_segment(text: str) -> str:
    """Extract clean text from segment markers"""
    special_tokens = [
        "<|segment_start|>", "<|segment_end|>", "<|silence|>", "<|eos_speech|>",
        "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>",
        "<|speaker_start|>", "<|speaker_end|>"
    ]
    special_tokens_pattern = "|".join(map(re.escape, special_tokens))
    segment_pattern = r"<\|segment_start\|>(.*?)<\|segment_end\|>"
    segments = re.findall(segment_pattern, text, re.DOTALL)

    cleaned_segments = [re.sub(special_tokens_pattern, "", seg).strip() for seg in segments if seg.strip()]
    return " ".join(cleaned_segments) if cleaned_segments else re.sub(special_tokens_pattern, "", text).strip()

def is_valid_audio(chunk, sample_rate, min_rms=1e-4):
    """Returns False if audio is mostly silent, empty, or invalid"""
    if chunk is None or chunk.numel() == 0:
        return False

    chunk = chunk.mean(dim=0) if chunk.ndim > 1 else chunk
    chunk = chunk - chunk.mean()
    rms = torch.sqrt(torch.mean(chunk ** 2))

    if rms < min_rms:
        return False

    return True

def _to_wav_bytes(mono_float_tensor: torch.Tensor, sr: int = SAMPLE_RATE) -> bytes:
    """
    Convert a mono float tensor [-1,1] to WAV bytes in-memory.
    """
    buf = io.BytesIO()
    arr = mono_float_tensor.detach().cpu().float().clamp(-1.0, 1.0).numpy()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

async def transcribe_chunk(chunk, sample_rate, segment_index, model, processor) -> str:
    """
    Transcribe a single audio chunk.

    Supports two backends:
      - OpenAI Whisper shim: model has `.transcribe(audio_path)` and processor is None
      - Local HF Whisper: model is WhisperForConditionalGeneration; processor is WhisperProcessor
    """
    async with transcription_lock:
        try:
            # ---------- Common pre-checks ----------
            chunk = chunk.to(dtype=torch.float32, device="cpu")
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-6)
            chunk = chunk.mean(dim=0) if chunk.ndim > 1 else chunk.squeeze(0)
            chunk = chunk.clamp(-1.0, 1.0)

            if not is_valid_audio(chunk, sample_rate):
                logger.info(f"[Segment {segment_index+1}] Audio invalid or silent. Skipping transcription.")
                return ""
            if chunk.abs().max() < 1e-4:
                logger.info(f"[Segment {segment_index+1}] Audio too silent, skipping transcription.")
                return ""
            if torch.isnan(chunk).any() or torch.isinf(chunk).any():
                logger.warning(f"[Segment {segment_index+1}] Audio contains NaN/Inf values, skipping.")
                return ""

            duration = chunk.shape[-1] / sample_rate
            if duration < 0.1:
                logger.info(f"[Segment {segment_index+1}] Audio too short, skipping transcription.")
                return ""

            # ---------- Backend detection ----------
            is_openai_backend = hasattr(model, "transcribe") and processor is None

            if is_openai_backend:
                # ---------- OpenAI Whisper path ----------
                try:
                    wav_bytes = _to_wav_bytes(chunk, SAMPLE_RATE)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        f.write(wav_bytes)
                        f.flush()
                        text = model.transcribe(f.name, language="en")
                    transcription = (text or "").strip()
                    logger.info(f"[Segment {segment_index+1}] Transcription successful (OpenAI).")
                    return transcription
                except Exception as e_ai:
                    logger.error(f"[Segment {segment_index+1}] OpenAI transcribe failed: {str(e_ai)}")
                    return ""
            else:
                # ---------- Local HF Whisper path ----------
                device = next(model.parameters()).device
                dtype = model.dtype

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                try:
                    inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
                except Exception as e_proc:
                    logger.error(f"[Segment {segment_index+1}] Processor failed: {str(e_proc)}")
                    return ""

                input_features = inputs.input_features.to(dtype=dtype, device=device, non_blocking=True)
                attention_mask = (
                    inputs.attention_mask.to(dtype=torch.int64, device=device, non_blocking=True)
                    if "attention_mask" in inputs
                    else None
                )
                input_features = input_features.clamp(-10.0, 10.0)

                if getattr(model.config, "pad_token_id", None) is None:
                    model.config.pad_token_id = model.config.eos_token_id

                # Dynamic token limits
                if int(duration) < 1.0:
                    max_tokens = 16
                    length_penalty = 1.0
                elif int(duration) < 3.0:
                    max_tokens = 32
                    length_penalty = 0.9
                elif int(duration) < 10.0:
                    max_tokens = 64
                    length_penalty = 0.8
                else:
                    max_tokens = 128
                    length_penalty = 0.7

                logits_processor = LogitsProcessorList([
                    MinLenSafe(min_length=2, eos_token_id=model.config.eos_token_id),
                    RepetitionPenaltyLogitsProcessor(1.4),
                ])
                stopping_criteria = StoppingCriteriaList([
                    StopOnSegmentEnd(processor.tokenizer),
                ])

                with torch.inference_mode():
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    predicted_ids = model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=max_tokens,
                        decoder_start_token_id=model.config.decoder_start_token_id,
                        no_repeat_ngram_size=3,
                        length_penalty=length_penalty,
                        early_stopping=True,
                        use_cache=True,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                if predicted_ids is None or len(predicted_ids) == 0:
                    logger.warning(f"[Segment {segment_index+1}] No predicted IDs, returning empty transcription.")
                    return ""

                predicted_ids = [
                    t for t in predicted_ids[0].tolist()
                    if 0 <= t < processor.tokenizer.vocab_size
                ]
                transcription = processor.tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()

                logger.info(f"[Segment {segment_index+1}] Transcription successful (local).")
                return transcription

        except Exception as e:
            logger.error(f"[Segment {segment_index+1}] Transcription failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

def split_channels(audio: torch.Tensor, sr: int, call_id: str):
    """Split stereo audio into two channels"""
    try:
        if audio.shape[0] != 2:
            logger.warning(f"[{call_id}] Not stereo, fallback to mono pipeline.")
            return None, None
        return audio[0].unsqueeze(0), audio[1].unsqueeze(0)
    except Exception as e:
        logger.error(f"[{call_id}] Channel splitting failed: {e}")
        return None, None

def format_ratio(silence: float, talk: float) -> str:
    """Format silence to talk ratio"""
    if talk == 0 and silence == 0:
        return "0:0 (no speech or silence)"
    elif talk == 0:
        return "∞:1 (no talk)"
    elif silence == 0:
        return "0:1 (no silence)"

    ratio = silence / talk
    return f"{ratio:.2f}:1" if ratio >= 1 else f"1:{(talk / silence):.2f}"

def run_vad_on_tensor(audio_tensor: torch.Tensor, sr: int = SAMPLE_RATE, speaker: str = None):
    """Run Voice Activity Detection on audio tensor"""
    try:
        if audio_tensor is None:
            return []
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        audio_tensor = audio_tensor.cpu()

        if sr != SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        segments = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sr)

        results = []
        for seg in segments:
            start = round(seg["start"] / sr, 2)
            end = round(seg["end"] / sr, 2)
            results.append({"speaker": speaker or "unknown", "start": start, "end": end})

        return merge_segments(results)
    except Exception as e:
        logger.error(f"VAD failed for speaker {speaker}: {e}")
        return []

def merge_segments(segments, gap_threshold: float = 0.3):
    """Merge nearby segments from the same speaker"""
    merged = []
    for seg in sorted(segments, key=lambda x: x["start"]):
        if merged and seg["start"] - merged[-1]["end"] <= gap_threshold and seg["speaker"] == merged[-1]["speaker"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged

async def process_audio_streaming(audio: torch.Tensor, sr: int, request_id: str):
    """Process audio and yield results for each segment as they complete"""
    logger.info(f"[process_audio_streaming] Start – ID={request_id}")
    try:
        # Load model (OpenAI shim or local)
        model, tokenizer = load_whisper_model()
        global whisper_model, whisper_processor
        whisper_model, whisper_processor = model, tokenizer

        # Split channels
        agent_audio, customer_audio = split_channels(audio, sr, request_id)
        agent_segments = run_vad_on_tensor(agent_audio, sr, speaker="agent")
        customer_segments = run_vad_on_tensor(customer_audio, sr, speaker="customer")

        # Merge segments
        merged_segments = sorted(agent_segments + customer_segments, key=lambda x: x["start"])

        agent_talk_time, customer_talk_time = 0.0, 0.0
        seg_index = 0
        full_transcription = ""

        logger.info(f"[{request_id}] Processing {len(merged_segments)} segments")
        diarized_data = []
        for seg in merged_segments:
            seg_index += 1
            start, end = seg["start"], seg["end"]
            speaker = seg.get("speaker", "unknown")
            duration = end - start

            # Update talk times
            if speaker.lower() == "agent":
                agent_talk_time += duration
            else:
                customer_talk_time += duration

            # Extract chunk
            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)
            chunk = audio[:, start_sample:end_sample]

            # Skip too short segments
            if chunk.shape[-1] < int(SAMPLE_RATE * 0.2):
                logger.warning(f"[{request_id}] Segment {seg_index} too short. Skipping.")
                continue

            try:
                # Transcribe
                transcription = await transcribe_chunk(chunk, SAMPLE_RATE, seg_index, whisper_model, whisper_processor)
                # Fix a few consistent ASR artifacts per your earlier logic
                transcription = (
                    transcription
                    .replace('bata', 'tata')
                    .replace("tataega", "batayega")
                    .replace("tataiye", "batayei")
                )
                transcription = extract_segment(transcription)

                logger.info(f"[{request_id}] Segment {seg_index} → '{transcription}'")
                if transcription:
                    full_transcription += transcription.strip() + " "

                    # Yield this segment immediately
                    segment_result = {
                        "type": "segment",
                        "segment_index": seg_index,
                        "speaker": speaker,
                        "start_time": start,
                        "end_time": end,
                        "text": transcription.strip()
                    }
                    diarized_data.append(segment_result)
                    yield f"data: {json.dumps(segment_result)}\n\n"

            except Exception as e:
                logger.error(f"[{request_id}] Segment {seg_index} failed: {e}")

        # Calculate final statistics
        audio_duration = round(audio.shape[-1] / sr, 2)
        total_talk_time = agent_talk_time + customer_talk_time
        silence_time = max(audio_duration - total_talk_time, 0.0)

        # Yield final summary
        final_result = {
            "type": "summary",
            "diarized_data": diarized_data,
            "full_transcription": full_transcription.strip(),
            "agent_talk_time": round(agent_talk_time, 2),
            "customer_talk_time": round(customer_talk_time, 2),
            "silence_time": round(silence_time, 2),
            "silence_to_talk_ratio": format_ratio(silence_time, total_talk_time),
            "audio_duration": audio_duration,
            "overall_talk_time": round(total_talk_time, 2)
        }
        yield f"data: {json.dumps(final_result)}\n\n"

        logger.info(f"[process_audio_streaming] Completed – ID={request_id}")

    except Exception as e:
        logger.error(f"[{request_id}] process_audio_streaming failed: {str(e)}")
        error_result = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_result)}\n\n"

    finally:
        # Cleanup
        try:
            torch.cuda.empty_cache()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass

# ================== API ENDPOINTS ==================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Audio Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Upload audio file for transcription (streaming)",
            "/transcribe-batch": "POST - Upload audio file for batch transcription",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    backend = "openai" if (whisper_processor is None and hasattr(whisper_model, "transcribe")) else "local"
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": whisper_model is not None,
        "asr_backend": backend
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file with streaming response.
    Returns Server-Sent Events (SSE) for each segment as it's processed.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received file: {file.filename}")

    try:
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio
        audio, sr = load_audio_with_soundfile(audio_buffer)
        logger.info(f"[{request_id}] Audio loaded: shape={audio.shape}, sr={sr}")

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
            sr = SAMPLE_RATE

        # Return streaming response
        return StreamingResponse(
            process_audio_streaming(audio, sr, request_id),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-batch")
async def transcribe_audio_batch(file: UploadFile = File(...)):
    """
    Transcribe audio file and return complete result at once.
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received file for batch processing: {file.filename}")

    try:
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio
        audio, sr = load_audio_with_soundfile(audio_buffer)
        logger.info(f"[{request_id}] Audio loaded: shape={audio.shape}, sr={sr}")

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
            sr = SAMPLE_RATE

        # Collect all results
        segments = []
        summary = {}

        async for event in process_audio_streaming(audio, sr, request_id):
            if event.startswith("data: "):
                data = json.loads(event[6:])
                if data["type"] == "segment":
                    segments.append({
                        "speaker": data["speaker"],
                        "start_time": data["start_time"],
                        "end_time": data["end_time"],
                        "text": data["text"]
                    })
                elif data["type"] == "summary":
                    summary = data

        result = {
            "request_id": request_id,
            "diarized_data": segments,
            **summary
        }
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ================== STARTUP ==================
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up... Loading models...")
    global whisper_model, whisper_processor
    try:
        whisper_model, whisper_processor = load_whisper_model()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Model load failed at startup: {e}")
        # Don't crash the server; endpoints will still attempt lazy-load via process path.
        # You can raise here if you prefer hard failure:
        # raise

# ==================== AGENT EVALUATION ENDPOINT ====================
@app.post("/agent-evaluation")
async def evaluate_agent(
    diarized_data: Optional[str] = Form(None),
    shared: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Get diarized data from form or file
        if file:
            content = await file.read()
            data_dict = json.loads(content.decode('utf-8'))
            if isinstance(data_dict, dict) and 'diarized_data' in data_dict:
                diarized_list = data_dict['diarized_data']
                shared_flag = data_dict.get('shared', shared)
            elif isinstance(data_dict, list):
                diarized_list = data_dict
                shared_flag = shared
            else:
                raise HTTPException(status_code=400, detail="Invalid file format")
        elif diarized_data:
            diarized_list = json.loads(diarized_data)
            shared_flag = shared
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'diarized_data' or 'file' must be provided"
            )

        if not isinstance(diarized_list, list):
            raise HTTPException(status_code=400, detail="diarized_data must be a list")

        logger.info(f"[AGENT-EVAL] Processing {len(diarized_list)} segments")

        conversation_parts = [
            f'{entry.get("speaker", "unknown")}: {entry.get("text", "")}'
            for entry in diarized_list
        ]
        for_communication_score = " | ".join(conversation_parts)
        logger.info(f"[AGENT-EVAL] Converted to conversation string (length: {len(for_communication_score)})")

        result = unified_agent_evaluation(for_communication_score, shared_flag)
        logger.info("[AGENT-EVAL] Completed successfully")

        return {"status": "success", "data": result}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception("[AGENT-EVAL] Error during processing")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ==================== SENTIMENT DETECTION ENDPOINT ====================
@app.post("/sentiment")
async def analyze_sentiment(
    diarized_data: Optional[str] = Form(None),
    shared: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None)
):
    try:
        if file:
            content = await file.read()
            data_dict = json.loads(content.decode('utf-8'))
            if isinstance(data_dict, dict) and 'diarized_data' in data_dict:
                diarized_list = data_dict['diarized_data']
                shared_flag = data_dict.get('shared', shared)
            elif isinstance(data_dict, list):
                diarized_list = data_dict
                shared_flag = shared
            else:
                raise HTTPException(status_code=400, detail="Invalid file format")
        elif diarized_data:
            diarized_list = json.loads(diarized_data)
            shared_flag = shared
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'diarized_data' or 'file' must be provided"
            )

        if not isinstance(diarized_list, list):
            raise HTTPException(status_code=400, detail="diarized_data must be a list")

        logger.info(f"[SENTIMENT] Processing {len(diarized_list)} segments")
        result = detect_sentiment(diarized_list, shared_flag)
        logger.info("[SENTIMENT] Completed successfully")

        return {"status": "success", "data": result}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception("[SENTIMENT] Error during processing")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ==================== SCORING ENDPOINT ====================
@app.post("/scoring")
async def score_conversation(
    diarized_data: Optional[str] = Form(None),
    shared: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None)
):
    try:
        if file:
            content = await file.read()
            data_dict = json.loads(content.decode('utf-8'))
            if isinstance(data_dict, dict) and 'diarized_data' in data_dict:
                diarized_list = data_dict['diarized_data']
                shared_flag = data_dict.get('shared', shared)
            elif isinstance(data_dict, list):
                diarized_list = data_dict
                shared_flag = shared
            else:
                raise HTTPException(status_code=400, detail="Invalid file format")
        elif diarized_data:
            diarized_list = json.loads(diarized_data)
            shared_flag = shared
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'diarized_data' or 'file' must be provided"
            )

        if not isinstance(diarized_list, list):
            raise HTTPException(status_code=400, detail="diarized_data must be a list")

        logger.info(f"[SCORING] Processing {len(diarized_list)} segments, shared={shared_flag}")
        result = tataScoring(diarized_list, shared_flag)
        logger.info("[SCORING] Completed successfully")

        return {"status": "success", "data": result}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception("[SCORING] Error during processing")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # NOTE: If you see TF32 warnings from PyTorch on Windows, they're harmless.
    uvicorn.run(app, host="0.0.0.0", port=5004)
