from typing import List, Dict, Any, Optional, Tuple
import logging, json, re, difflib, os
import torch
import sys
sys.path.append("../")
from logger import logger
# from load_model import load_mt5_large  # ⟵ no longer needed

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _RF = True
except Exception:
    _RF = False

# ──────────────────────────────────────────────────────────────
# NEW: OpenAI client (uses OPENAI_API_KEY from environment)
# ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _openai_client = OpenAI()  # requires env var OPENAI_API_KEY
except Exception as _e:
    _openai_client = None
    logging.getLogger(__name__).warning(f"OpenAI client init failed: {_e}")

# ──────────────────────────────────────────────────────────────
# Rubric (8 params; ASR-tolerant, Hinglish-aware)
# ──────────────────────────────────────────────────────────────
RUBRIC = {
  "Call Opening and Purpose Of Call": {
    "weight": 8,
    "options": ["Yes","No","NA"],
    "conditions": [
      "Opening MUST start before 5s from call start (first agent speak ≤ 5.0s).",
      "Opening MUST complete within 15s (ASR 60% tolerant): ANY of {proper greeting/welcome, brand mention, assistance invite} by ≤15.0s. ('hello/hi/hey' alone is not enough.)",
      "Customer name is confirmed if a name appears; if no name in transcript, do not penalize.",
      "Opening language aligns with customer (English/Hinglish/Hindi in Latin).",
      "If customer monologue blocks opening during first 5s, mark NA."
    ]
  },
  "Acknowledgement and Paraphrasing": {
    "weight": 6,
    "options": ["Yes","No"],
    "conditions": [
      "Explicit acknowledgement (I understand/I see/Okay/Alright/Sure/Noted OR Hinglish: samajh gaya/gayi, theek/thik hai, bilkul, ji/haan ji, achha).",
      "Paraphrases/clarifies concern at least once (to confirm/as I understand/so your concern is OR Hinglish: toh aap keh rahe hain, aapka kehna hai, main confirm kar du/kar doon, main dobara bata du/dun).",
      "Thanks after gathering details (thanks/thank you OR Hinglish: dhanyavaad/shukriya/thanks ji).",
      "Do not count generic nods alone (hmm/umm).",
      "If expected but absent, mark No (text-only)."
    ]
  },
  "Active Listening": {
    "weight": 6,
    "options": ["Yes","No"],
    "conditions": [
      "No interruption/overlap inferred from text flow.",
      "Avoids unnecessary repetition; if repeats due to clarity/ASR, apologizes or signals reason (e.g., sorry/ek baar dobara).",
      "Credit: understood/let me check/give me a moment → relevant next step.",
      "Mark No for repeated identical questions with no new context/apology."
    ]
  },
  "Empathy and Apology along with Assurance and Willingness to help": {
    "weight": 8,
    "options": ["Yes","No","NA"],
    "conditions": [
      "Empathy/apology when inconvenience present (sorry/apologies/we regret/inconvenience OR Hinglish: takleef ke liye maaf, asuvidha, khed hai), typos allowed.",
      "Assurance/willingness (I will help/let me assist/we will resolve/callback scheduled OR Hinglish: main madad karunga/karungi, hum resolve karenge, nishchint rahiye, call back karenge).",
      "Polite phrases at least once (please/may I/thank you OR Hinglish: kripya/please ji/shukriya/dhanyavaad).",
      "Personalizes by name after opening if a name is present; else no penalty.",
      "Two-way communication evident.",
      "If no negative experience described anywhere, NA is acceptable."
    ]
  },
  "Language Proficiency": {
    "weight": 8,
    "options": ["Yes","No"],
    "conditions": [
      "Professional, understandable language with ≤2 clear grammar/word-choice issues per short exchange (ignore obvious ASR artifacts).",
      "Reasonable word choice; pronunciation not judged in text.",
      "Code-switch aligns with customer (English/Hinglish/Hindi in Latin).",
      "Minimal fillers (um/uh/hmm/you know/haan/na); tolerate occasional ASR-inserted fillers.",
      "Do not penalize ASR artifacts unless meaning is unclear."
    ]
  },
  "Hold Protocol": {
    "weight": 6,
    "options": ["Yes","No","NA","CE"],
    "conditions": [
      "Seeks permission before hold (may I/can I/shall I put you on hold/hold for a moment OR Hinglish: zara rukiye/ek minute rukiye/hold par rakhun/kripya hold par), typos allowed.",
      "If hold occurs, refresh within ~60s (thanks for holding/sorry to keep you waiting/I’m back OR Hinglish: intezar ke liye dhanyavaad, rukne ke liye shukriya, main wapas aa gaya/gayi).",
      "Dead air not directly assessable; infer only from explicit long hold with no refresh.",
      "NA if no hold used.",
      "CE only if transcript explicitly shows refresh after >90s or disconnect due to hold."
    ]
  },
  "Call Closing": {
    "weight": 5,
    "options": ["Yes","No","NA"],
    "conditions": [
      "Near end, agent delivers closing. For this project: if any of the last 3 agent turns contains (even with typos) one of: “kimti/keemti samay”, “aapka din shubh rahe/subh rhe/shubh rhe/rahe”, OR English equivalents like “thank you for your valuable/precious time” or “have a nice/great/good day”, mark Yes.",
      "If last speaker is customer and agent does not rejoin, NA.",
      "Allow Hinglish/typos (thank u/than you/hv a nice dey/shubh din)."
    ]
  },
  "Effective Probing": {
    "weight": 6,
    "options": ["Yes","No","NA"],
    "conditions": [
      "Asks needed questions per issue (when/kab/kab se, vehicle/gaadi, dealer/dealer ka naam/vikreta, location/shehar, contact/phone/mobile number, email/mail id, VIN/chassis, registration, callback time/kis time available).",
      "Requests conference/escalation if context requires.",
      "Confirms dealer details and location when relevant.",
      "NA if purely informational/trivial (no probing needed).",
      "Credit paraphrases and partially garbled Hinglish if intent is clear."
    ]
  }
}

ALLOWED = {k: set(v["options"]) for k, v in RUBRIC.items()}
PARAM_LIST = list(RUBRIC.keys())

GENERIC_YES_DESC = {
    "Language Proficiency": "Clear, professional Hinglish; minimal fillers; aligned with customer’s language."
}

# ──────────────────────────────────────────────────────────────
# Normalization & Fuzzy helpers (ASR-tolerant)
# ──────────────────────────────────────────────────────────────
_VARIANTS = [
    (r"dhanya?va+?d|dhanyawa+d|dhanyabad", "dhanyavad"),
    (r"shukri+y?a+", "shukriya"),
    (r"\b(subh|shubh|shub|sub)\b", "shubh"),
    (r"\b(rahe|rehe|reh|rhe|rahey|raheh)\b", "rahe"),
    (r"\b(kimti|ke?e?mti|kemti|keemati|kimty|kimati)\b", "kimti"),
    (r"\b(aapka|apka|aap ki|ap ki)\b", "aapka"),
    (r"\b(madad|madadh|mdd)\b", "madad"),
    (r"\b(kripya|kripiya|krupya|kirpya)\b", "kripya"),
    (r"\b(shahar|shehar|sheharh)\b", "shehar"),
    (r"\b(sorry|sory|soorry)\b", "sorry"),
    (r"\b(thank u|thanx|thx)\b", "thank you"),
    (r"\b(good day|great day|nice day|have a nice dey)\b", "have a nice day"),
    (r"\b(pricious|preciu?s|precius)\s+time\b", "precious time"),
    (r"\b(valuble|valua?ble|valuble)\s+time\b", "valuable time"),
    (r"\b(hold on|hol d|hodl)\b", "hold"),
    (r"\b(rukiye|rukiyega|rukiega|rukiyge|rukiyeh)\b", "rukiye"),
    # apology/inconvenience spellings
    (r"\b(sorry|sory|soorry|srry|srrry|sorri|sori|soori)\b", "sorry"),
    (r"\b(maaf+|maafi+|mafi+)\b", "mafi"),
    (r"\b(kshama|kshema|kshmaa)\b", "kshama"),
    (r"\b(afsos|afsoos)\b", "afsos"),
    (r"asuv?idh?a|asu?vdha|asuvidhaa?", "asuvidha"),
    (r"takl(i|ee|e)f", "takleef"),
    (r"inconveni[ea]nce|inconvini?ence", "inconvenience"),
    ]

def _squash_repeats(s: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def _norm(txt: str) -> str:
    t = (txt or "").lower()
    t = re.sub(r"[^\w\s@]", " ", t)
    t = _squash_repeats(t)
    t = re.sub(r"\s+", " ", t).strip()
    for pat, rep in _VARIANTS:
        t = re.sub(pat, rep, t)
    return t

def _sim(a: str, b: str) -> float:
    if _RF:
        try:
            return _rf_fuzz.ratio(a, b) / 100.0
        except Exception:
            pass
    return difflib.SequenceMatcher(None, a, b).ratio()

def _contains(text: str, patterns) -> bool:
    t = _norm(text)
    return any(p in t for p in patterns)

def _contains_fuzzy(text: str, patterns: List[str], thresh: float = 0.78) -> bool:
    if not text or not patterns:
        return False
    t = _norm(text)
    if not t:
        return False
    for p in patterns:
        pn = _norm(p)
        if pn and pn in t:
            return True
    t_tokens = t.split()
    for raw_p in patterns:
        p = _norm(raw_p)
        if not p:
            continue
        p_tokens = p.split()
        k = max(1, len(p_tokens))
        for i in range(0, len(t_tokens) - k + 1):
            window = " ".join(t_tokens[i:i+k])
            if _sim(window, p) >= thresh:
                return True
    for raw_p in patterns:
        if _sim(_norm(raw_p), t) >= 0.90:
            return True
    return False

def _has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", _norm(text)))

def _has_word_fuzzy(text: str, word: str, thresh: float = 0.80) -> bool:
    t = _norm(text)
    w = _norm(word)
    if w in t:
        return True
    for tok in t.split():
        if _sim(tok, w) >= thresh:
            return True
    return False

def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def fuzzy_contains(text: str, phrases, ratio: float = 0.78) -> bool:
    t = _norm(text)
    for p in phrases:
        if p in t:
            return True
        if difflib.SequenceMatcher(None, p, t).ratio() >= ratio:
            return True
        ptoks = set([w for w in re.findall(r"\w+", p) if len(w) > 2])
        ttoks = set([w for w in re.findall(r"\w+", t) if len(w) > 2])
        if ptoks and (len(ptoks & ttoks) / max(1, len(ptoks)) >= 0.6):
            return True
    return False

def _agent_segments(diarized):
    return [s for s in diarized if s.get("speaker") == "agent"]

# ──────────────────────────────────────────────────────────────
# Phrase banks
# ──────────────────────────────────────────────────────────────
HOLD_PERMISSION_PHRASES = [
    "put you on hold","place you on hold","keep you on hold","hold for a moment","hold for a minute",
    "may i put you on hold","can i put you on hold","shall i put you on hold",
    "zara rukiye","kripya hold par","ek minute rukiye","please be on hold","hold"
]
REFRESH_PHRASES = [
    "thank you for holding","thanks for holding","thank you for waiting","thanks for waiting",
    "sorry for the wait","sorry to keep you waiting","i am back","i'm back","back on the line",
    "thanks for your patience","appreciate your patience"
]
SOFT_HOLD_CUES = [
    "give me a moment","one moment please","just a moment","ek moment","ek minute","zara rukiye"
]

OPEN_START_SEC = 5.0
OPEN_END_SEC   = 15.0
ASR_TOL        = 1.25

# Opening policy: strict requires assist + (proper greeting OR brand).
# Set to False to allow ANY of {proper greeting/welcome, brand, assist}.
OPENING_STRICT = False  # lenient mode for 60% ASR

# Expanded greeting variants for weak ASR
PROPER_GREETINGS = [
    "good morning","good afternoon","good evening",
    "good morn","good aftrn","good evng","good eve","good noon",
    "gud mrng","gud aftrn","gud evng","gd mrng","gd aftrn","gd evng","gd eve",
    "namaste","namaste ji","namaskar","pranam","salaam","aadab","welcome"
]
WELCOME_PHRASES = [
    "thank you for calling","thanks for calling","welcome to",
    "aapka swagat hai","call karne ke liye dhanyavaad","swagat hai"
]
CASUAL_GREETINGS = ["hello","hi","hey"]  # don't complete opening

BRAND_TERMS = [
    "tata", "tata motor", "tata motors", "tata customer care",
    "tata helpline", "tata dealership", "tata service"
]
ASSIST_INVITES = [
    "how may i help", "how can i help", "how may i assist", "how can i assist",
    "kya sahayata", "kya madad", "main madad kar", "mai madad kar",
    "help kar sakta", "help kar sakti", "assist kar sakta", "assist kar sakti",
    "aapki kya sahayata", "kaise madad kar"
]
# Strictness toggle for this parameter
EMPATHY_STRICT = True  # require explicit apology + assurance for "Yes"

# Core apology tokens (explicit apologies, not just "I understand")
APOLOGY_CORE = [
    "sorry","apologies","apologise","apologize","we regret","regret",
    "mafi","maaf","kshama","afsos"
]
APOLOGY_REGEX = re.compile(r"\b(sorry|apolog(?:y|ies|ise|ize|ising|izing)|regret|maaf|mafi|kshama|afsos)\b", re.I)

# Apology / empathy (Hindi/Hinglish + English)
APOLOGY_PHRASES = [
    "sorry", "we are sorry", "apologies", "we regret", "regret",
    "inconvenience", "for the inconvenience", "for any inconvenience",
    "mafi", "maaf kijiye", "maaf kariye", "mafi chahta hoon", "mafi chahti hoon", "mafi chahte hain",
    "kshama", "kshama kijiye", "afsos hai", "mujhe afsos hai", "humein afsos hai",
    "takleef ke liye", "pareshani ke liye", "asuvidha ke liye"
]
EMPATHY_ONLY = [
    "i understand your concern", "i understand your issue", "i understand",
    "i completely understand", "i totally understand", "i truly understand",
    "i can understand", "i do understand", "i hear you",
    "samajh raha hoon", "samajh rahi hoon", "main samajh raha hoon", "main samajh rahi hoon",
    "bilkul samajh", "aapki baat samajh", "aapki takleef samajh",
    "sorry to hear", "sorry to hear that"
]

# Helpful for rubric’s “polite phrase once” check (not strictly required to pass)
POLITENESS_BANK = ["please", "kripya", "thank you", "shukriya", "dhanyavad"]

# Expanded willingness/assurance (commitments)
ASSURANCE_BANK = [
    "i will help","i'll help","let me assist","we will resolve","we'll resolve",
    "i will resolve","i'll resolve","main madad","hum resolve",
    "raising the complaint","raise the complaint","i am raising the complaint",
    "team will connect you","our concern team will connect you","we will arrange a callback","callback",
    "let me check","i will check","we will check","main check karta hoon","main check karungi",
    "check karke batata hoon","check karke batati hoon","verify kar leta hoon","dekh leta hoon","find out karta hoon",
    "complaint raise kar raha hoon","complaint submit kar raha hoon","nayi complaint submit",
    "ticket raise kar raha hoon","ticket create kar raha hoon","i have raised the ticket","i am raising a ticket",
    "concern team connect karegi","team connect karegi","aapko update karenge","will update you",
    "arrange callback","we will arrange a call","aapko call back karenge","call back karenge","call back",
    "nishchint rahiye","be assured","rest assured",
    "within 24 hours","within 48 hours","24 hours ke andar","48 hours ke andar",
    "i will escalate","escalate kar raha hoon","escalate karungi","priority par dekhte hain",
    "call connect karta hoon","connect kar deta hoon","transfer kar raha hoon","transfer ki ja rahi call","your complaint number is", "please note the complaint number", "note down the complaint number",
    "reference number", "ticket number", "case id", "sr number", "service request number",
    "we have raised", "i have raised", "i've raised", "we have submitted", "i have submitted",
    "i'm submitting", "we're submitting", "complaint has been raised", "ticket has been raised",
    "we will arrange a call back", "we will arrange a callback", "we will arrange a call back",
    "we will get back to you", "we will call you back",
    "don't worry", "do not worry", "chinta mat kariye", "tension mat lijiye",
    "within 12 hours", "within 24 hours", "within 48 hours", "within 72 hours", "within 3 days",
    "within seventy two hours", "within forty eight hours"
]

THANKS_WORDS = ["thank you", "thanks", "dhanyavad", "dhanyavaad", "shukriya", "thanks ji"]

ACK_WORDS = [
    "i understand","i see","understand","got it","noted","okay","ok","alright","sure","certainly",
    "samajh gaya","samajh gayi","theek hai","thik hai","bilkul","ji","haan ji","achha","accha","sahi hai"
]
PARA_WORDS = [
    "let me confirm","to confirm","as i understand","what i understand",
    "so you're saying","so your concern is","if i understood","to summarize","i will summarize",
    "toh aap keh rahe","aapka kehna hai","main confirm kar du","main confirm kar doon","main dobara bata du","main dobara bata doon"
]
PARA_LITE_HINTS = [
    "as per your concern","you booked","they promised","old april manufactured",
    "not in proper condition","you cancelled the booking","you want your refund",
    "you want to raise a complaint","raise a complaint against","as per your concern like"
]
PARA_LITE_RIGHT_RE = re.compile(r"\byou\b.*\bright\??\b", re.IGNORECASE)

FILLERS = ["um","umm","uh","hmm","haan","na","you know","like","matlab","toh","achha"]

# Closing (last-3 rule)
CLOSING_LAST3_KEYWORDS = [
    "kimti samay","keemti samay","kimti time","valuable time","precious time",
    "aapka din shubh rahe","aapka din shubh rhe","aapka din subh rhe","shubh din",
    "have a great day","have a nice day","have a good day","good day,Call karne ke liye dhanyavad"
]

# ──────────────────────────────────────────────────────────────
# Evidence helpers
# ──────────────────────────────────────────────────────────────
def build_transcript(d: List[Dict[str,Any]]) -> str:
    lines = []
    for seg in d:
        sp = seg.get("speaker","").capitalize()
        st = seg.get("start_time",0.0); et = seg.get("end_time",0.0)
        txt = (seg.get("text") or "").strip()
        lines.append(f"{sp} ({st:.1f}-{et:.1f}s): {txt}")
    return "\n".join(lines)

@torch.no_grad()
def call_llm(prompt: str, model=None, tokenizer=None) -> str:
    """
    UPDATED: Use OpenAI GPT-5 Mini via environment OPENAI_API_KEY.
    The signature remains the same; 'model' and 'tokenizer' are ignored.
    """
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized. Ensure 'openai' package is installed and OPENAI_API_KEY is set.")
    resp = _openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=900
    )
    return (resp.choices[0].message.content or "").strip()

def extract_json(s: str):
    s = (s or "").strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    first = s.find("["); last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        chunk = s[first:last+1]
        try:
            return json.loads(chunk)
        except Exception:
            try:
                import json5  # type: ignore
                return json5.loads(chunk)
            except Exception:
                pass
    return None

def evidence_ok(quote: str, transcript: str) -> bool:
    if not quote or len(quote.split()) < 2:
        return False
    tq = set([w for w in re.findall(r"\w+", quote.lower()) if len(w) > 2])
    tt = set([w for w in re.findall(r"\w+", transcript.lower()) if len(w) > 2])
    if tq and (len(tq & tt) / max(1, len(tq)) >= 0.35):
        return True
    r = difflib.SequenceMatcher(None, quote.lower(), transcript.lower()).ratio()
    return r >= 0.18

def quote_to_timestamps(quote: str, diarized: List[Dict[str,Any]]) -> Tuple[Optional[float], Optional[float]]:
    best = None; best_r = 0.0
    q = (quote or "").lower().strip()
    if not q:
        return (None, None)
    for seg in diarized:
        stxt = (seg.get("text") or "").lower()
        if q and q in stxt:
            return (seg["start_time"], seg["end_time"])
        r = difflib.SequenceMatcher(None, q, stxt).ratio()
        if r > best_r:
            best_r = r; best = seg
    if best and best_r >= 0.28:
        return (best["start_time"], best["end_time"])
    return (None, None)

def _timestamps_from_evidence(ev: str, diarized: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not ev:
        return (None, None)
    parts = [p.strip() for p in ev.split("...") if p.strip()]
    for p in parts:
        if re.search(r"(sorry|mafi|maaf|apolog|inconvenience|takleef|asuvidha|afsos|kshama)", p, re.I):
            st, et = quote_to_timestamps(p, diarized)
            if st is not None:
                return st, et
    parts.sort(key=len, reverse=True)
    for p in parts:
        st, et = quote_to_timestamps(p, diarized)
        if st is not None:
            return st, et
    return (None, None)

# ──────────────────────────────────────────────────────────────
# Enhanced detectors
# ──────────────────────────────────────────────────────────────
def detect_call_opening_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Opening (ASR tolerant):
      • Start: first agent speech must begin ≤ 5s (+ ASR_TOL). If customer monologue blocks this window → NA.
      • Completion by 15s (+ ASR_TOL):
          - STRICT mode: require Assist + (Proper Greeting OR Brand)
          - LENIENT mode: ANY of {Proper Greeting/Welcome, Brand, Assist} is enough
        Notes:
          - 'hello/hi/hey' alone never completes the opening.
          - 'Good evening. Thank you for calling.' should pass in LENIENT mode.
    """
    if not diarized:
        return ("NA", "")

    first_agent = next((s for s in diarized if s.get("speaker") == "agent"), None)
    if not first_agent:
        return ("NA", "")

    if first_agent["start_time"] > OPEN_START_SEC + ASR_TOL:
        cust0 = [s for s in diarized if s.get("speaker") == "customer" and s["start_time"] <= 0.2]
        if cust0 and any(s["end_time"] >= OPEN_START_SEC for s in cust0):
            return ("NA", "Customer speaking continuously during first 5 seconds")
        return ("No", "Opening did not start within 5 seconds (ASR-tolerant)")

    cutoff = OPEN_END_SEC + ASR_TOL
    opening_window = [s for s in diarized if s.get("speaker") == "agent" and s["end_time"] <= cutoff]
    if not opening_window:
        return ("No", "No agent opening content completed within first 15 seconds")

    def _has_proper_greet_or_welcome(txt: str) -> bool:
        txtn = _norm(txt)
        if _contains_fuzzy(txtn, PROPER_GREETINGS, 0.68) or _contains_fuzzy(txtn, WELCOME_PHRASES, 0.68):
            if _contains_fuzzy(txtn, CASUAL_GREETINGS, 0.95) and not _contains_fuzzy(txtn, ["good","namaste","welcome","aadab","salaam","pranam"], 0.60):
                return False
            return True
        return False

    assist_hit = None
    brand_hit = None
    greet_hit = None

    for s in opening_window:
        txt = s.get("text", "") or ""
        if assist_hit is None and (_contains_fuzzy(txt, ASSIST_INVITES, 0.70)):
            assist_hit = s
        if brand_hit is None and (_contains_fuzzy(txt, BRAND_TERMS, 0.68)):
            brand_hit = s
        if greet_hit is None and _has_proper_greet_or_welcome(txt):
            greet_hit = s

    if OPENING_STRICT:
        if assist_hit and (greet_hit or brand_hit):
            ev = (assist_hit.get("text","") or (greet_hit or brand_hit).get("text",""))[:120]
            return ("Yes", ev or "Opening completed within 15s")
        return ("No", "No assist + (proper greeting or brand) by 15 seconds (ASR-tolerant)")
    else:
        hit = greet_hit or brand_hit or assist_hit
        if hit:
            return ("Yes", (hit.get("text","")[:120] or "Opening completed within 15s"))
        return ("No", "No greeting/welcome/brand/assist by 15 seconds (ASR-tolerant)")

def detect_ack_paraphrase_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    agent = _agent_segments(diarized)
    if not agent:
        return ("No","")

    ack_found = any(_contains_fuzzy(s.get("text",""), ACK_WORDS, 0.78) for s in agent)

    para_idxs = []
    for i, s in enumerate(agent):
        txt = s.get("text","") or ""
        if (_contains_fuzzy(txt, PARA_WORDS, 0.78) or
            _contains_fuzzy(txt, PARA_LITE_HINTS, 0.78) or
            PARA_LITE_RIGHT_RE.search(txt)):
            para_idxs.append(i)

    if not ack_found or not para_idxs:
        return ("No", "Missing acknowledgement" if not ack_found else "Missing paraphrasing")

    para_idx = para_idxs[-1]
    para_time = agent[para_idx]["start_time"]

    for s in agent[para_idx: para_idx+4]:
        if s["start_time"] - para_time <= 60 and _contains_fuzzy(s.get("text",""), THANKS_WORDS, 0.76):
            return ("Yes", agent[para_idx]["text"][:120])

    return ("No", "Paraphrasing present but no thanks after details")

def detect_active_listening_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not diarized:
        return ("No","")

    def _is_micro(seg):
        dur = seg["end_time"] - seg["start_time"]
        wc  = _word_count(seg.get("text",""))
        tx  = _norm(seg.get("text",""))
        return (dur <= 1.2 or wc <= 2) and tx in {"ok","okay","ji","haan ji","haan","hmm","yeah","yes","okey","okayy"}

    interruptions = 0
    repeat_hits   = 0

    for i in range(len(diarized)-1):
        cur, nxt = diarized[i], diarized[i+1]
        if cur.get("speaker")=="customer" and nxt.get("speaker")=="agent":
            if _is_micro(nxt):
                continue
            if nxt["start_time"] < cur["end_time"] - 0.05:
                interruptions += 1
            elif (nxt["start_time"] - cur["end_time"] < 0.15) and _word_count(cur.get("text","")) >= 10:
                interruptions += 1

        if cur.get("speaker")=="agent" and nxt.get("speaker")=="agent":
            a = _norm(cur.get("text","")); b = _norm(nxt.get("text",""))
            if a and b and _sim(a, b) >= 0.92:
                window = diarized[i+1: i+4]
                if not any(_contains_fuzzy(x.get("text",""), ["sorry","maaf","mafi"], 0.78) for x in window):
                    repeat_hits += 1

    if interruptions >= 2 or repeat_hits >= 1:
        return ("No", "Interruptions/repeats detected without apology")
    return ("Yes", "No interruptions; appropriate response timing")

def detect_empathy_enhanced(diarized: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """
    STRICT apology-first rule:
      • If any agent turn contains an explicit apology token (sorry/maaf/afsos/asuvidha/takleef/inconvenience),
        return Yes immediately (evidence = that apology line).
      • If customer describes a negative experience but no explicit apology appears → No.
      • If no negative experience and no apology → NA (return None to let rubric treat as NA).
      • 'I understand' style empathy WITHOUT apology does not count.
    """
    if not diarized:
        return None

    def _hit(txt: str, bank, thr: float = 0.70) -> bool:
        return _contains_fuzzy(txt, bank, thr) or fuzzy_contains(txt, bank, ratio=thr)

    NEGATIVE_BANK = [
        "problem","issue","complaint","angry","frustrat","delay","bad experience",
        "not working","stuck","broken","wrong","kharab","nahi chal","kaam nahi","takleef",
        "pareshani","asuvidha","nahi ho raha","nhi ho raha","start nahi ho raha",
        "open nahi ho raha","chal nahi raha","error","failed","not able","unable",
    ]
    has_negative = any(
        s.get("speaker") == "customer" and _hit(s.get("text",""), NEGATIVE_BANK, 0.70)
        for s in diarized
    )

    # Explicit apology tokens only (strict)
    def is_apology(txt: str) -> bool:
        if _hit(txt, APOLOGY_PHRASES, 0.68):
            return True
        t = _norm(txt)
        return bool(re.search(r"\b(sorry|maaf|mafi|afsos|asuvidha|takleef|inconvenience)\b", t))

    # If any agent apologizes → YES (evidence = that line)
    for seg in diarized:
        if seg.get("speaker") != "agent":
            continue
        tx = seg.get("text", "") or ""
        if is_apology(tx):
            return ("Yes", tx[:120])

    # No apology found
    if has_negative:
        return ("No", "No explicit apology despite negative experience")

    return None  # NA when no negative + no apology


def detect_language_proficiency_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    agent = _agent_segments(diarized)
    if not agent:
        return ("No","No agent speech")
    total_words = sum(_word_count(s.get("text","")) for s in agent) or 1
    filler_hits = sum(_norm(s.get("text","")).count(f) for s in agent for f in FILLERS)
    slang_hits  = sum(1 for s in agent for k in [" u ", " plz ", " thx ", " btw "] if k in f" {_norm(s.get('text',''))} ")
    if filler_hits/total_words > 0.08 or slang_hits > 4:
        return ("No", f"Excessive fillers/slang (fillers={filler_hits}, slang={slang_hits})")
    return ("Yes", "Clear, professional language; minimal fillers")

def detect_hold_protocol_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not diarized:
        return ("NA", "No content")

    agent_idx = [i for i, s in enumerate(diarized) if s.get("speaker")=="agent"]

    def has_hold_intent(txt: str) -> bool:
        return (_contains_fuzzy(txt, HOLD_PERMISSION_PHRASES, 0.76) or
                _contains_fuzzy(txt, SOFT_HOLD_CUES, 0.76))

    hold_i = None
    for i in agent_idx:
        if has_hold_intent(diarized[i].get("text","")):
            hold_i = i
            break
    if hold_i is None:
        if any(_contains_fuzzy(diarized[i].get("text",""), REFRESH_PHRASES, 0.76) for i in agent_idx):
            return ("No", "Refresh phrase but no clear hold permission")
        return ("NA", "No hold used by agent")

    hold_seg = diarized[hold_i]
    hold_start = hold_seg["end_time"]

    # Customer ack within 2 turns?
    cust_ack = False
    for j in range(hold_i+1, min(len(diarized), hold_i+4)):
        s = diarized[j]
        if s.get("speaker")=="customer" and _contains_fuzzy(s.get("text",""), ["ok","okay","haan","yes","theek hai","ji"], 0.76):
            cust_ack = True; break

    # Agent return
    next_agent_after_hold = None
    for j in agent_idx:
        if j > hold_i and diarized[j]["start_time"] >= hold_start:
            next_agent_after_hold = diarized[j]; break
    if next_agent_after_hold is None:
        dur = diarized[-1]["end_time"] - hold_start
        if dur > 90: return ("CE","Hold exceeded 90 seconds and agent did not return")
        return ("No","Hold requested but agent did not return with refresh")

    dur = max(0.0, next_agent_after_hold["start_time"] - hold_start)

    refresh_hit = None
    cand = [next_agent_after_hold]
    # add immediate next agent turn
    for j in agent_idx:
        if diarized[j] is next_agent_after_hold:
            kpos = agent_idx.index(j)
            if kpos+1 < len(agent_idx):
                cand.append(diarized[agent_idx[kpos+1]])
            break
    for s in cand:
        if _contains_fuzzy(s.get("text",""), REFRESH_PHRASES, 0.76):
            refresh_hit = s; break

    if refresh_hit:
        if dur <= 60.0:
            return ("Yes", hold_seg.get("text","")[:120])
        if dur <= 90.0:
            return ("No", "Hold refresh delayed (60–90s)")
        return ("CE", "Hold refresh after >90s")

    if cust_ack or dur >= 4.0:
        return ("Yes", hold_seg.get("text","")[:120])

    return ("NA", "Hold intent unclear due to ASR / immediate return")

def detect_probing_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not diarized:
        return ("NA", "No content")
    agent = _agent_segments(diarized)
    if not agent:
        return ("NA", "No agent speech")

    cats = {
        "contact": ["email","mail id","alternate number","alt number","phone number","mobile number","contact number","callback","call back","preferred time","time available","pin code","pincode"],
        "vehicle": ["vin","chassis","registration","reg no","reg number","model","variant","kilometer","kilometre","km","odo","odometer","manufacturing date"],
        "dealer_loc": ["dealer","workshop","service center","service centre","location","city","branch","shehar","shahar","area","pin code"],
        "issue_timing": ["when did","since when","kab se","kab","how long","since how long","what exactly","describe the issue","problem details","symptom"]
    }
    hits = {k: [] for k in cats}
    for s in agent:
        txt = s.get("text","") or ""
        for k, bank in cats.items():
            if _contains_fuzzy(txt, bank, 0.76):
                hits[k].append(s)

    num_cat = sum(1 for v in hits.values() if v)
    if num_cat >= 2:
        for k in ["contact","vehicle","dealer_loc","issue_timing"]:
            if hits[k]:
                return ("Yes", hits[k][0].get("text","")[:120])

    if num_cat == 1:
        key = next(k for k,v in hits.items() if v)
        return ("No", f"Limited probing—only asked about {key.replace('_',' ')}")

    duration = diarized[-1]["end_time"] - diarized[0]["start_time"]
    if duration < 30.0 or len(agent) < 3:
        return ("NA", "Short informational call—no probing required")
    return ("No", "No probing questions detected")

def detect_closing_enhanced(diarized: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not diarized:
        return ("NA", "No content")

    agent = _agent_segments(diarized)
    if not agent:
        return ("NA", "No agent speech")

    if diarized[-1].get("speaker") == "customer" and all(s["start_time"] < diarized[-1]["start_time"] for s in agent):
        return ("NA", "Customer disconnected before closing")

    last3 = agent[-3:] if len(agent) >= 3 else agent
    for s in reversed(last3):
        if _contains_fuzzy(s.get("text",""), CLOSING_LAST3_KEYWORDS, 0.76):
            return ("Yes", s.get("text","")[:160])

    return ("No", "No proper closing keywords in last 3 agent turns")

# ──────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────
def make_prompt_all(transcript: str) -> str:
    param_blocks = []
    for p in PARAM_LIST:
        conds = "\n".join(f"- {c}" for c in RUBRIC[p]["conditions"])
        opts = "/".join(ALLOWED[p])
        param_blocks.append(f"{p} (options: {opts})\n{conds}")
    block = "\n\n".join(param_blocks)

    strict_note = (
        "- Opening must START ≤5s and FINISH ≤15s (assist + (proper greeting OR brand))."
        if OPENING_STRICT else
        "- Opening must START ≤5s and FINISH ≤15s (ANY of: proper greeting/welcome, brand mention, or assistance invite)."
    )

    return f"""
You are a contact-center QA evaluator. The transcript is Hinglish (English + Hindi in Latin script) with ~60–85% ASR accuracy. Be tolerant to typos, spelling variants, and broken phrases. Do not hallucinate content.

Evaluate ONLY these 8 parameters and choose ONE allowed option for each using the listed conditions:
{block}

Additional strict rules:
{strict_note}
- Acknowledgement+Paraphrasing requires BOTH; natural paraphrases allowed.
- Active Listening is "No" if agent interrupts customer or repeats without apology.
- Hold: CE if refresh after >90s (or no return >90s).

Output JSON only: array of exactly 8 items with keys parameter, option, evidence.
Transcript:
{transcript}
""".strip()

def normalize_option(opt: str) -> str:
    o = (opt or "").strip().upper()
    if o in {"YES","Y"}: return "Yes"
    if o in {"NO","N"}: return "No"
    if o == "NA": return "NA"
    if o == "CE": return "CE"
    return "NA"

# ──────────────────────────────────────────────────────────────
# Build items & reasons
# ──────────────────────────────────────────────────────────────
def reasons_for_no(param: str, diarized: List[Dict[str,Any]]) -> str:
    if param == "Call Opening and Purpose Of Call":
        return "Opening non-compliant"
    if param == "Acknowledgement and Paraphrasing":
        return "Missing paraphrasing"
    if param == "Active Listening":
        return "Interruptions or repeated prompts without apology"
    if param == "Language Proficiency":
        return "Excessive fillers/slang or unclear phrasing"
    if param == "Call Closing":
        return "No proper closing keywords in last 3 agent turns"
    if param == "Effective Probing":
        return "Insufficient probing across required details"
    if param == "Hold Protocol":
        return "Hold without timely refresh"
    return "Does not meet rubric requirements"

def build_item(param: str, option_name: str, evidence: str, diarized: List[Dict[str,Any]]):
    weight = RUBRIC[param]["weight"]
    if option_name == "Yes":
        awarded, score_val = weight, weight
    elif option_name in ("No","CE"):
        awarded, score_val = 0, weight
    else:  # NA
        awarded, score_val = 0, 0

    if option_name == "Yes" and param in GENERIC_YES_DESC:
        description = GENERIC_YES_DESC[param]; st = et = None
    elif option_name == "No":
        description = reasons_for_no(param, diarized); st = et = None
    elif option_name in {"NA","CE"}:
        description = evidence or ""; st = et = None
    else:
        description = evidence or ""
        if evidence:
            if "..." in evidence:
                st, et = _timestamps_from_evidence(evidence, diarized)
            elif evidence_ok(evidence, build_transcript(diarized)):
                st, et = quote_to_timestamps(evidence, diarized)
            else:
                st = et = None
        else:
            st = et = None

    return {
        "parameter_name": param,
        "option_name": option_name,
        "overall_score": awarded,
        "score": score_val,
        "description": description,
        "start_time": st,
        "end_time": et,
        "Conditions": RUBRIC[param]["conditions"]
    }

# ──────────────────────────────────────────────────────────────
# Rule-based overrides (deterministic, ASR-robust)
# ──────────────────────────────────────────────────────────────
def rule_based_overrides(diarized: List[Dict[str,Any]]):
    overrides: Dict[str, Tuple[str,str]] = {}
    overrides["Call Opening and Purpose Of Call"] = detect_call_opening_enhanced(diarized)
    overrides["Acknowledgement and Paraphrasing"] = detect_ack_paraphrase_enhanced(diarized)
    overrides["Active Listening"] = detect_active_listening_enhanced(diarized)

    emo = detect_empathy_enhanced(diarized)
    if emo is not None:
        overrides["Empathy and Apology along with Assurance and Willingness to help"] = emo
    else:
        # Fallback: decide NA vs No deterministically
        has_negative = any(
            s.get("speaker")=="customer" and _contains_fuzzy(s.get("text",""), [
                "problem","issue","complaint","angry","frustrat","delay","bad experience",
                "not working","stuck","broken","wrong","kharab","nahi","takleef","pareshani","asuvidha",
            ], 0.70)
            for s in diarized
        )
        if has_negative:
            overrides["Empathy and Apology along with Assurance and Willingness to help"] = ("No","No explicit apology + assurance found")
        else:
            overrides["Empathy and Apology along with Assurance and Willingness to help"] = ("NA","No clear negative described")

    overrides["Language Proficiency"] = detect_language_proficiency_enhanced(diarized)
    overrides["Hold Protocol"] = detect_hold_protocol_enhanced(diarized)
    overrides["Call Closing"] = detect_closing_enhanced(diarized)
    overrides["Effective Probing"] = detect_probing_enhanced(diarized)
    return overrides


def _clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────
# Main scoring
# ──────────────────────────────────────────────────────────────
def tataScoring(diarized_data: List[Dict[str,Any]], shared = None):
    # NOTE: 'model' and 'tokenizer' are no longer used; kept for compatibility.
    global model, tokenizer
    try:
        diarized = diarized_data
        transcript = build_transcript(diarized)

        overrides = rule_based_overrides(diarized)
        # model, tokenizer = load_mt5_large()  # ⟵ removed
        logger.info("Using OpenAI GPT-5 Mini for QA scoring (Hinglish-aware)")

        prompt = make_prompt_all(transcript)
        try:
            raw = call_llm(prompt, None, None)  # signature unchanged
        except Exception:
            logger.exception("LLM generation failed")
            raw = "[]"

        arr = extract_json(raw)
        if not isinstance(arr, list):
            arr = []

        # Prefer deterministic overrides; only use LLM if it returns valid evidence
        llm_map: Dict[str, Dict[str,str]] = {}
        for item in arr:
            p = str(item.get("parameter","")).strip()
            o = normalize_option(str(item.get("option","")))
            ev = str(item.get("evidence","")).strip()
            if p in RUBRIC and o in ALLOWED[p] and ev and evidence_ok(ev, transcript):
                llm_map[p] = {"option": o, "evidence": ev}

        final = []
        for param in PARAM_LIST:
            allowed = ALLOWED[param]
            if param in overrides and overrides[param][0] in allowed:
                opt, ev = overrides[param]
                final.append(build_item(param, opt, ev, diarized))
            elif param in llm_map:
                final.append(build_item(param, llm_map[param]["option"], llm_map[param]["evidence"], diarized))
            else:
                fallback = "NA" if "NA" in allowed else "No"
                final.append(build_item(param, fallback, "", diarized))

        return {"results": final}

    except Exception:
        logger.exception("Unexpected error in tataScoring")
        return {"results": []}
    finally:
        # _clear_gpu()
        logger.info("<< GPU memory cleared after scoring >>")
        try:
            # Delete model, tokenizer, and tensors (kept for API compatibility; may be None)
            for var_name in ['model', 'tokenizer']:
                try:
                    if var_name in locals():
                        del locals()[var_name]
                except Exception as e_inner:
                    logger.warning(f"Failed to delete {var_name}: {e_inner}")

            # Force garbage collection
            try:
                import gc
                gc.collect()
            except Exception as e_gc:
                logger.warning(f"Garbage collection failed: {e_gc}")

            # Clear PyTorch cache
            try:
                torch.cuda.empty_cache()
            except Exception as e_cache:
                logger.warning(f"torch.cuda.empty_cache() failed: {e_cache}")

            # Synchronize CUDA
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e_sync:
                logger.warning(f"torch.cuda.synchronize() failed: {e_sync}")

            logger.info("GPU memory cleared and temporary variables deleted.")

        except Exception as e_final:
            logger.error(f"Unexpected error in final cleanup: {e_final}")
