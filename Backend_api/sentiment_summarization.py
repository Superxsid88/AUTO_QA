# === DROP-IN REPLACEMENT (keeps same module name & public API) ===
from transformers import GenerationConfig  # kept to preserve package imports
import torch                               # kept for your cleanup paths
import json
import regex as re
from typing import Optional, List, Dict, Any, Tuple
import sys
import os
import signal
from contextlib import contextmanager
import time

# Keep your package path behavior intact
sys.path.append("../")

# Local imports (left unchanged so package structure stays stable)
# from utils.constants import VALID_EMOTIONS
from load_model import DEVICE, DTYPE, load_mt5_large, load_llama_model_3b  # not used, imported to preserve structure
from logger import logger

# New: OpenAI client (for GPT-5-mini)
from dotenv import load_dotenv
from openai import OpenAI

def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)

# ──────────────────────────────────────────────────────────────────────────────
# Token utils + chunking (tokenizer-free)
# ──────────────────────────────────────────────────────────────────────────────
def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)

def _format_conv_block(lines: List[Dict[str, str]], start_line_no: int = 1) -> str:
    return "\n".join(
        f"Line {i}: [{(ln.get('speaker','').upper())}] {ln.get('text','')}"
        for i, ln in enumerate(lines, start_line_no)
    )

def _chunk_lines_by_budget(
    lines: List[Dict[str, str]],
    tok=None,
    input_budget_tokens: int = 6000,
    output_budget_tokens: int = 2048,
    est_out_per_line_tokens: int = 12,
) -> List[List[Dict[str, str]]]:
    chunks, cur = [], []
    cur_prompt_tokens = 0
    cur_out_est = 0

    per_line_tokens = []
    for i, ln in enumerate(lines, 1):
        s = f"Line {i}: [{(ln.get('speaker','').upper())}] {ln.get('text','')}\n"
        per_line_tokens.append(_approx_token_count(s))

    for i, ln in enumerate(lines):
        line_tokens = per_line_tokens[i]
        if cur and (
            cur_prompt_tokens + line_tokens > input_budget_tokens or
            cur_out_est + est_out_per_line_tokens > output_budget_tokens
        ):
            chunks.append(cur); cur = []; cur_prompt_tokens = 0; cur_out_est = 0
        cur.append(ln)
        cur_prompt_tokens += line_tokens
        cur_out_est += est_out_per_line_tokens

    if cur:
        chunks.append(cur)
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Emotion label sets + normalization + heuristics + smoothing
# ──────────────────────────────────────────────────────────────────────────────
AGENT_EMOTIONS    = ["Neutral", "Thankful", "Pleasant", "Happiness"]
CUSTOMER_EMOTIONS = ["Neutral", "Thankful", "Pleasant", "Happiness", "Disappointment", "Frustration", "Anger"]

SYNONYM_MAP = {
    # positive
    "happy": "Happiness", "satisfied": "Happiness", "relieved": "Happiness", "great": "Happiness", "perfect": "Happiness",
    "gratitude": "Thankful", "grateful": "Thankful", "thanks": "Thankful", "thank you": "Thankful", "dhanyavad": "Thankful", "shukriya": "Thankful",
    "polite": "Pleasant", "cordial": "Pleasant", "cooperative": "Pleasant", "friendly": "Pleasant", "warm": "Pleasant", "helpful": "Pleasant",
    "neutral": "Neutral", "matter-of-fact": "Neutral", "ok": "Neutral", "okay": "Neutral", "haan ji": "Neutral", "ji": "Neutral",
    # negative (customer-only)
    "disappointed": "Disappointment", "let down": "Disappointment", "dismay": "Disappointment",
    "frustrated": "Frustration", "annoyed": "Frustration", "irritated": "Frustration", "nahi ho rahi": "Frustration",
    "band ho gayi": "Frustration", "stuck": "Frustration", "delay": "Frustration", "kab tak": "Frustration", "ghante": "Frustration",
    "immediate": "Frustration", "jaldi": "Frustration",
    "angry": "Anger", "upset": "Anger", "hostile": "Anger", "furious": "Anger", "unacceptable": "Anger", "naraz": "Anger", "gussa": "Anger",
    # agent cues
    "apologize": "Pleasant", "sorry": "Pleasant", "assure": "Pleasant", "reassure": "Pleasant",
}

def normalize_label(raw: str, speaker: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Neutral"
    x = raw.strip().lower()
    candidates = AGENT_EMOTIONS if speaker == "agent" else CUSTOMER_EMOTIONS
    for c in candidates:
        if x == c.lower():
            return c
    for k, v in SYNONYM_MAP.items():
        if k in x and v in candidates:
            return v
    if "thank" in x and "Thankful" in candidates: return "Thankful"
    if "sorry" in x and "Pleasant" in candidates: return "Pleasant"
    if "ang" in x and "Anger" in candidates: return "Anger"
    if "frustr" in x and "Frustration" in candidates: return "Frustration"
    if "disappoint" in x and "Disappointment" in candidates: return "Disappointment"
    return "Neutral"

HINDI_NEG    = ["nahi", "nahi ho", "nahi aaya", "nahi aa", "band ho", "nahi ho rahi", "nahi mila", "kab tak",
                "der", "ghante", "ruk", "stuck", "sadak", "road", "raaste", "ruk gaya", "intazaar", "late",
                "delay", "deri", "kitna time", "wait", "exam", "government exam"]
HINDI_POLITE = ["sir", "ji", "kripya", "please", "haan ji", "theek hai"]
HINDI_THANKS = ["shukriya", "dhanyavad", "thank you", "thanks"]

def heuristic_emotion(speaker: str, text: str) -> Optional[str]:
    if not text: return None
    t = text.strip().lower()
    if speaker == "agent":
        if "thank" in t or any(w in t for w in HINDI_THANKS): return "Thankful"
        if "sorry" in t or "apolog" in t or "samajh" in t or "understand" in t: return "Pleasant"
        if "happy to" in t or "glad" in t: return "Happiness"
        if any(w in t for w in HINDI_POLITE): return "Neutral"
        return None
    else:
        if any(w in t for w in HINDI_THANKS): return "Thankful"
        if "perfect" in t or "great" in t or "awesome" in t: return "Happiness"
        if "unacceptable" in t or "gussa" in t or "naraz" in t: return "Anger"
        if any(w in t for w in HINDI_NEG) or "frustrat" in t or "irritat" in t or "annoy" in t:
            if "!" in t or "bahut" in t or "kaise" in t: return "Anger"
            return "Frustration"
        if "let down" in t or "disappoint" in t: return "Disappointment"
        if any(w in t for w in HINDI_POLITE): return "Pleasant" if "please" in t else "Neutral"
        return None

# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            json.loads(match)
            return match
        except Exception:
            continue
    start_idx = text.find('{')
    if start_idx != -1:
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
    potential_json = re.search(r'\{.*\}', text, re.DOTALL)
    if potential_json:
        candidate = potential_json.group(0)
        candidate = re.sub(r',\s*}', '}', candidate)
        candidate = re.sub(r',\s*]', ']', candidate)
        candidate = re.sub(r'(\w+):', r'"\1":', candidate)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    return None

def extract_json_safely(text: str) -> Optional[Dict]:
    js = extract_json_from_text(text or "")
    if not js: 
        return None
    try:
        return json.loads(js)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-backed batch classification (replaces local mT5/LLM calls)
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_LABELS = ["Neutral","Thankful","Pleasant","Happiness","Disappointment","Frustration","Anger"]
_label_re = re.compile(r'"(Neutral|Thankful|Pleasant|Happiness|Disappointment|Frustration|Anger)"')

def _to_string_labels(line_emotions, speakers):
    out = []
    if isinstance(line_emotions, list) and line_emotions and isinstance(line_emotions[0], str):
        out = line_emotions
    elif isinstance(line_emotions, list):
        for i, item in enumerate(line_emotions):
            lab = (item or {}).get("emotion", "Neutral")
            out.append(lab)
    norm = []
    for i, lab in enumerate(out):
        spk = speakers[i]
        norm.append(normalize_label(str(lab), spk))
    return norm

def _salvage_labels_from_text(text: str, speakers: List[str]) -> Optional[List[str]]:
    found = _label_re.findall(text)
    if not found:
        return None
    n = len(speakers)
    found = list(found)[:n]
    while len(found) < n:
        found.append("Neutral")
    return [normalize_label(l, speakers[i]) for i, l in enumerate(found)]

def _extract_parsed_or_salvaged(decoded: str, speakers: List[str]) -> Optional[Dict]:
    parsed = extract_json_safely(decoded)
    if isinstance(parsed, dict) and "line_emotions" in parsed:
        labs = _to_string_labels(parsed["line_emotions"], speakers)
        if len(labs) == len(speakers):
            parsed["line_emotions"] = labs
            return parsed
    try:
        arr = json.loads(decoded)
        if isinstance(arr, list):
            labs = _to_string_labels(arr, speakers) if arr and not isinstance(arr[0], str) else [
                normalize_label(x, speakers[i]) for i, x in enumerate(arr[:len(speakers)])
            ]
            while len(labs) < len(speakers):
                labs.append("Neutral")
            return {
                "line_emotions": labs,
                "overall_customer_emotion": "Neutral",
                "overall_agent_emotion": "Neutral",
                "emotional_trajectory": "Stable throughout conversation",
            }
    except Exception:
        pass
    salvaged = _salvage_labels_from_text(decoded, speakers)
    if salvaged:
        return {
            "line_emotions": salvaged,
            "overall_customer_emotion": "Neutral",
            "overall_agent_emotion": "Neutral",
            "emotional_trajectory": "Stable throughout conversation",
        }
    return None

def _openai_classify_chunk(lines: List[Dict[str, str]]) -> Optional[Dict]:
    client = _get_client()
    numbered = "\n".join(
        f"{i+1}. [{(ln.get('speaker') or 'customer').upper()}] {ln.get('text') or ''}"
        for i, ln in enumerate(lines)
    )
    sys = (
        "You are a strict labeler for call-center emotion analysis. "
        "Allowed labels ONLY: Neutral, Thankful, Pleasant, Happiness, Disappointment, Frustration, Anger. "
        "Return strictly valid JSON like: "
        "{\"line_emotions\":[\"Neutral\",\"Thankful\",...], "
        "\"overall_customer_emotion\":\"Neutral\","
        "\"overall_agent_emotion\":\"Neutral\","
        "\"emotional_trajectory\":\"...\"}"
    )
    user = (
        "Label each line using the allowed labels only. Ensure the array length equals the number of lines. "
        "Keep JSON compact with no extra keys.\n\n" + numbered
    )
    out = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.0,
    ).choices[0].message.content
    speakers = [ (ln.get("speaker") or "customer").lower() for ln in lines ]
    return _extract_parsed_or_salvaged(out, speakers)

def _simplified_batch_classify(lines: List[Dict[str, str]]) -> List[Dict]:
    client = _get_client()
    prompt = "For each line, output one label from: " + ", ".join(ALLOWED_LABELS) + \
             ". Reply with a JSON array of strings only."
    numbered = "\n".join(f"{i+1}. {ln.get('text','')}" for i, ln in enumerate(lines))
    out = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role":"system","content":"Return JSON only."},
            {"role":"user","content": prompt + "\n\n" + numbered}
        ],
        temperature=0.0
    ).choices[0].message.content
    speakers = [ (ln.get("speaker") or "customer").lower() for ln in lines ]
    parsed = _extract_parsed_or_salvaged(out, speakers)
    if parsed:
        return [{"emotion": e} for e in parsed["line_emotions"]]
    return [{"emotion": "Neutral"} for _ in lines]

# ──────────────────────────────────────────────────────────────────────────────
# Chunked batch classifier (signature kept)
# ──────────────────────────────────────────────────────────────────────────────
def _batch_classify_with_retry(model, tokenizer, norm_lines: List[Dict[str, str]], max_retries: int = 2) -> Optional[Dict]:
    for attempt in range(max_retries):
        try:
            return _openai_classify_chunk(norm_lines)
        except Exception as e:
            logger.error(f"Batch classify attempt {attempt + 1} failed: {e}")
    return None

def _batch_classify_chunked(model, tokenizer, norm_lines: List[Dict[str, str]]) -> Optional[Dict]:
    if len(norm_lines) > 30:
        logger.info(f"Large conversation ({len(norm_lines)} lines), using chunking strategy")
        chunks = _chunk_lines_by_budget(
            norm_lines,
            input_budget_tokens=2000,
            est_out_per_line_tokens=10,
            output_budget_tokens=800
        )
    else:
        parsed = _batch_classify_with_retry(model, tokenizer, norm_lines, max_retries=1)
        if parsed:
            return parsed
        chunks = _chunk_lines_by_budget(
            norm_lines,
            input_budget_tokens=3000,
            est_out_per_line_tokens=10,
            output_budget_tokens=1000
        )

    merged_line_emotions = []
    any_failure = False
    for i, ch in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(ch)} lines")
        parsed = _batch_classify_with_retry(model, tokenizer, ch, max_retries=1)
        if parsed and isinstance(parsed.get("line_emotions"), list):
            merged_line_emotions.extend(parsed["line_emotions"])
        else:
            logger.warning(f"Chunk {i+1} failed, using simplified approach")
            simplified = _simplified_batch_classify(ch)
            if simplified:
                merged_line_emotions.extend([it.get("emotion","Neutral") for it in simplified])
            else:
                any_failure = True
                # Per-line fallback using OpenAI
                pl = _per_line_hybrid(ch, model=None, tokenizer=None)
                merged_line_emotions.extend([it.get("detected_emotion","Neutral") for it in pl])

    if not merged_line_emotions:
        return None

    return {
        "line_emotions": merged_line_emotions,
        "overall_customer_emotion": "Neutral",
        "overall_agent_emotion": "Neutral",
        "emotional_trajectory": "Stable throughout conversation" if not any_failure else "Computed post-merge",
    }

# ──────────────────────────────────────────────────────────────────────────────
# Per-line hybrid (signature kept)
# ──────────────────────────────────────────────────────────────────────────────
def _per_line_hybrid(conversation_lines: List[Dict[str, str]], model=None, tokenizer=None) -> List[Dict[str, str]]:
    client = _get_client()
    results = []
    for ln in conversation_lines:
        spk = (ln.get("speaker","customer") or "customer").lower()
        txt = (ln.get("text") or "").strip()
        lab = heuristic_emotion(spk, txt)

        if not lab:
            allowed = AGENT_EMOTIONS if spk == "agent" else CUSTOMER_EMOTIONS
            prompt = (
                "Classify the emotion of this utterance. Allowed labels ONLY: "
                + ", ".join(allowed)
                + ". Return just one word from the allowed list.\n"
                "Utterance: " + txt
            )
            out = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role":"system","content":"Return one allowed label only."},
                          {"role":"user","content":prompt}],
                temperature=0.0
            ).choices[0].message.content.strip()
            lab = out

        lab = normalize_label(lab, spk)
        results.append({
            "speaker": spk,
            "start_time": ln.get("start_time"),
            "end_time": ln.get("end_time"),
            "text": txt,
            "detected_emotion": lab,
        })
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Calibration + smoothing + stats (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def _calibrate_emotions(line_results: List[Dict[str, str]]) -> None:
    for result in line_results:
        spk = result["speaker"]
        txt = (result.get("text","") or "").lower()
        cur = result.get("detected_emotion", "Neutral")

        if spk == "agent":
            if "sorry" in txt or "apolog" in txt:
                if cur == "Neutral":
                    result["detected_emotion"] = "Pleasant"
            elif "thank" in txt and cur != "Thankful":
                result["detected_emotion"] = "Thankful"
        else:
            if any(neg in txt for neg in ["nahi ho", "band ho", "stuck", "delay", "not working"]):
                if cur in ["Happiness", "Pleasant", "Thankful"]:
                    result["detected_emotion"] = "Frustration"
            elif "thank" in txt and cur == "Neutral":
                result["detected_emotion"] = "Thankful"

def _smooth_emotions(lines: List[Dict[str, str]], window: int = 3) -> None:
    strong = {"Happiness","Thankful","Pleasant","Disappointment","Frustration","Anger"}
    for i, line in enumerate(lines):
        txt = (line.get("text","") or "").strip().lower()
        if line.get("detected_emotion","Neutral") == "Neutral" and len(txt) <= 10:
            same = []
            for j in range(max(0, i - window), i):
                if lines[j]["speaker"] == line["speaker"] and lines[j].get("detected_emotion") in strong:
                    same.append(lines[j]["detected_emotion"])
            if same:
                line["detected_emotion"] = same[-1]

def calculate_emotion_statistics(line_sentiments: list) -> dict:
    stats = {
        "customer": {"total_lines": 0, "emotion_counts": {}, "dominant_emotion": None},
        "agent": {"total_lines": 0, "emotion_counts": {}, "dominant_emotion": None},
    }
    for line in line_sentiments:
        spk = line.get("speaker", "customer")
        emo = line.get("detected_emotion", "Neutral")
        if spk in stats:
            stats[spk]["total_lines"] += 1
            stats[spk]["emotion_counts"][emo] = stats[spk]["emotion_counts"].get(emo, 0) + 1

    for spk in ["customer", "agent"]:
        if stats[spk]["emotion_counts"]:
            stats[spk]["dominant_emotion"] = max(stats[spk]["emotion_counts"], key=stats[spk]["emotion_counts"].get)
    return stats

# ──────────────────────────────────────────────────────────────────────────────
# Timeout helper (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ──────────────────────────────────────────────────────────────────────────────
# Main sentiment detection (PUBLIC NAME PRESERVED)
# ──────────────────────────────────────────────────────────────────────────────
def detect_sentiment(
    conversation_lines: List[Dict[str, str]],
    shared_models=None
) -> dict:
    logger.info(f"Detecting sentiment for conversation with {len(conversation_lines) if conversation_lines else 0} lines")

    if not conversation_lines or not isinstance(conversation_lines, list):
        return {"error": "No conversation data provided"}

    norm_lines: List[Dict[str, str]] = []
    for i, ln in enumerate(conversation_lines, 1):
        spk = str(ln.get("speaker", "")).strip().lower()
        txt = str(ln.get("text", "")).strip()
        if spk not in ("agent", "customer"):
            logger.warning(f"Line {i}: unknown speaker '{spk}', defaulting to 'customer'")
            spk = "customer"
        norm_lines.append({
            "speaker": spk,
            "text": txt,
            "start_time": ln.get("start_time"),
            "end_time": ln.get("end_time"),
        })

    parsed = _batch_classify_chunked(model=None, tokenizer=None, norm_lines=norm_lines)

    line_results: List[Dict[str, str]] = []
    overall_cust = "Neutral"
    overall_agent = "Neutral"
    trajectory = "Stable throughout conversation"

    if parsed:
        le = parsed.get("line_emotions", [])
        raw_list = [x.get("emotion","Neutral") for x in le] if (le and isinstance(le[0], dict)) else le

        for idx, src in enumerate(conversation_lines):
            raw_emotion = raw_list[idx] if idx < len(raw_list) else "Neutral"
            emotion = normalize_label(raw_emotion, src.get("speaker", "customer"))
            heur = heuristic_emotion(src.get("speaker", "customer"), src.get("text",""))
            if emotion == "Neutral" and heur:
                emotion = heur

            line_results.append({
                "speaker": src.get("speaker", "customer"),
                "start_time": src.get("start_time"),
                "end_time": src.get("end_time"),
                "text": src.get("text", ""),
                "detected_emotion": emotion,
            })
    else:
        logger.error("All batch attempts failed, using per-line classification")
        line_results = _per_line_hybrid(norm_lines, model=None, tokenizer=None)

    _calibrate_emotions(line_results)
    _smooth_emotions(line_results, window=3)
    stats = calculate_emotion_statistics(line_results)

    if overall_cust == "Neutral" and stats["customer"]["dominant_emotion"]:
        overall_cust = stats["customer"]["dominant_emotion"]
    if overall_agent == "Neutral" and stats["agent"]["dominant_emotion"]:
        overall_agent = stats["agent"]["dominant_emotion"]

    if trajectory == "Stable throughout conversation":
        cust = [r["detected_emotion"] for r in line_results if r["speaker"] == "customer" and r["detected_emotion"] != "Neutral"]
        if len(cust) >= 2 and cust[0] != cust[-1]:
            trajectory = f"Customer shifts from {cust[0].lower()} to {cust[-1].lower()} during the conversation."

    try:
        return {
            "line_sentiments": line_results,
            "overall_customer_sentiment": overall_cust,
            "overall_agent_sentiment": overall_agent,
            "emotional_trajectory": trajectory,
            "summary_stats": stats,
        }
    except Exception as e:
        return {"issue":str(e)}
    finally:
        # keep your original cleanup semantics
        try:
            # Force garbage collection
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Unified Agent Evaluation (PUBLIC NAME PRESERVED)
# ──────────────────────────────────────────────────────────────────────────────
COMM_QUESTIONS = [
    "Did the agent clearly explain the issue?",
    "Did the agent avoid vague or confusing language?",
    "Did the agent repeat important points to ensure clarity?",
    "Was the agent able to maintain professionalism throughout?",
    "Did the agent provide well-structured and complete responses?",
]
ENG_QUESTIONS = [
    "Did the agent acknowledge the customer's frustration or inconvenience?",
    "Did the agent show empathy in their responses?",
    "Did the agent offer reassurance to reduce customer anxiety?",
    "Did the agent provide alternative solutions or suggestions?",
    "Did the agent remain polite and patient even when the customer was upset?",
]
CEXP_QUESTIONS = [
    "Did the agent greet the customer politely and maintain a respectful, empathetic tone?",
    "Did the agent actively listen and understand the customer's issue without interrupting or rushing?",
    "Did the agent offer a clear and helpful resolution or next steps to solve the issue?",
    "Did the agent show ownership of the problem?",
    "Did the agent leave the customer with a positive impression?",
]

def extract_agent_quotes(conversation: str, max_quotes: int = 3) -> List[str]:
    quotes = []
    lines = conversation.split('\n')
    for line in lines:
        if 'agent' in line.lower() or '[agent]' in line.lower():
            m = re.search(r'[:\]]\s*(.+)', line)
            if m:
                agent_text = m.group(1).strip()
                if 10 < len(agent_text) < 200:
                    quotes.append(agent_text[:100])
                    if len(quotes) >= max_quotes:
                        break
    return quotes

def get_specific_improvement(key: str) -> str:
    improvements = {
        "COMM1": "Provide detailed explanation with specific causes and solutions",
        "COMM2": "Use precise technical terms and avoid phrases like 'maybe' or 'probably'",
        "COMM3": "Summarize key points at the end: 'To confirm, your issue is X and solution is Y'",
        "COMM4": "Maintain formal language, avoid slang, use 'Sir/Ma'am' consistently",
        "COMM5": "Provide complete, well-organized responses with clear beginning, middle, and end",
        "ENG1": "Say explicitly: 'I understand this must be frustrating for you'",
        "ENG2": "Express specific empathy: 'I can imagine how inconvenient this delay must be'",
        "ENG3": "Provide timeline: 'This will be resolved within 24 hours'",
        "ENG4": "Offer multiple options: 'You can either do X, Y, or Z'",
        "ENG5": "Stay calm and say 'I appreciate your patience' when customer is upset",
        "CEXP1": "Start with 'Good morning/afternoon, thank you for contacting us'",
        "CEXP2": "Let customer finish, then say 'I've understood your issue completely'",
        "CEXP3": "Provide step-by-step solution: 'First do X, then Y will happen'",
        "CEXP4": "Say 'I take full ownership and will personally ensure this is resolved'",
        "CEXP5": "End with 'Is there anything else I can help you with today?'"
    }
    return improvements.get(key, "Enhance performance with specific actions")

def format_reason_fast(score: int, rationale: str, agent_quotes: List[str], key: str) -> str:
    if score == 2:
        return f"Excellent performance. {rationale}"
    elif score == 1:
        improvement = get_specific_improvement(key)
        return f"Partial performance. {rationale}. To improve: {improvement}"
    else:
        improvement = get_specific_improvement(key)
        return f"Needs improvement. {rationale}. Suggestion: {improvement}"

def unified_agent_evaluation(conversation: str, shared_models=None) -> dict:
    start_time = time.time()
    logger.info("Starting optimized unified evaluation (OpenAI gpt-5-mini)")

    if not conversation or not conversation.strip():
        return {
            "error": "Missing 'conversation' parameter",
            "communication": {"questions": [], "communication_score": 0, "summary": "Error in evaluation"},
            "engagement": {"questions": [], "engagement_score": 0, "summary": "Error in evaluation"},
            "customer_experience": {"questions": [], "customer_experience_score": 0, "summary": "Error in evaluation"}
        }

    client = _get_client()
    agent_quotes = extract_agent_quotes(conversation, max_quotes=5)

    system_prompt = (
        "You are evaluating a customer service conversation. For each criterion, provide:\n"
        "1. Score (0=poor/absent, 1=partial, 2=excellent)\n"
        "2. Specific reason with evidence from the conversation\n\n"
        "IMPORTANT for COMM5 (spelling/formatting): \n"
        "- Score 2 if NO errors found (text is clear and well-formatted)\n"
        "- Score 1 if minor errors that don't affect understanding\n"
        "- Score 0 if major errors that affect clarity\n\n"
        "Output JSON with this structure:\n"
        "{\n"
        '  "summary": {\n'
        '    "full": "detailed summary of what happened",\n'
        '    "issue": "specific customer problem",\n'
        '    "resolution": "what agent did to help"\n'
        "  },\n"
        '  "scores": {\n'
        '    "COMM1": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "COMM2": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "COMM3": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "COMM4": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "COMM5": {"s": <0-2>, "r": "specific reason about response structure"},\n'
        '    "ENG1": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "ENG2": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "ENG3": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "ENG4": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "ENG5": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "CEXP1": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "CEXP2": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "CEXP3": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "CEXP4": {"s": <0-2>, "r": "specific reason with quote"},\n'
        '    "CEXP5": {"s": <0-2>, "r": "specific reason with quote"}\n'
        "  }\n"
        "}"
    )

    user_prompt = (
        "Evaluate this conversation using the schema. Quote exact phrases where possible.\n\n"
        f"{conversation[:15000]}\n\n"
        "Return complete JSON only."
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.0
    ).choices[0].message.content

    parsed = extract_json_safely(response) or {
        "summary": {
            "full": "Customer service interaction analyzed.",
            "issue": "Service issue identified.",
            "resolution": "Agent provided assistance."
        },
        "scores": {}
    }

    # Build questions
    def _score_of(k: str) -> int:
        v = parsed.get("scores", {}).get(k, {"s": 1})
        return v.get("s", 1) if isinstance(v, dict) else int(v)

    def _reason_of(k: str) -> str:
        v = parsed.get("scores", {}).get(k, {"r": "Based on analysis"})
        return v.get("r", "Based on analysis") if isinstance(v, dict) else "Based on analysis"

    comm_questions, eng_questions, cexp_questions = [], [], []
    for i in range(5):
        ck, ek, xk = f"COMM{i+1}", f"ENG{i+1}", f"CEXP{i+1}"
        cscore, crec = _score_of(ck), _reason_of(ck)
        escore, erec = _score_of(ek), _reason_of(ek)
        xscore, xrec = _score_of(xk), _reason_of(xk)
        comm_questions.append({"question": COMM_QUESTIONS[i], "score": cscore, "reason": format_reason_fast(cscore, crec, agent_quotes, ck)})
        eng_questions.append({"question": ENG_QUESTIONS[i], "score": escore, "reason": format_reason_fast(escore, erec, agent_quotes, ek)})
        cexp_questions.append({"question": CEXP_QUESTIONS[i], "score": xscore, "reason": format_reason_fast(xscore, xrec, agent_quotes, xk)})

    comm_total = sum(q["score"] for q in comm_questions)
    eng_total  = sum(q["score"] for q in eng_questions)
    cexp_total = sum(q["score"] for q in cexp_questions)

    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f}s (OpenAI)")

    return {
        "conversation_summary": {
            "full_summary": parsed.get("summary", {}).get("full", "Customer service interaction processed."),
            "customer_issue": parsed.get("summary", {}).get("issue", "Service issue addressed."),
            "agent_resolution": parsed.get("summary", {}).get("resolution", "Agent provided support.")
        },
        "communication": {
            "questions": comm_questions,
            "communication_score": comm_total,
            "summary": f"Communication: {comm_total}/10"
        },
        "engagement": {
            "questions": eng_questions,
            "engagement_score": eng_total,
            "summary": f"Engagement: {eng_total}/10"
        },
        "customer_experience": {
            "questions": cexp_questions,
            "customer_experience_score": cexp_total,
            "summary": f"Customer experience: {cexp_total}/10"
        }
    }
