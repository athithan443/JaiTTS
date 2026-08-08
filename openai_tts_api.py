import io
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict


logger = logging.getLogger("jaitts.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    latency_preset: Optional[str] = None


class SaveWavRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    voice: str = "alloy"
    speed: float = 1.0
    filename: Optional[str] = None
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    latency_preset: Optional[str] = None


class JaiTTSBackend:
    def __init__(self, model_path: Path, vocab_path: Path, device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device
        self._engine = None
        self._engine_lock = threading.Lock()

    def _lazy_load(self) -> None:
        if self._engine is not None:
            return
        # F5TTS initialization is expensive and should happen only once.
        # This lock also prevents a race when multiple requests hit first-time load.
        with self._engine_lock:
            if self._engine is not None:
                return
            try:
                from f5_tts.api import F5TTS
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: f5-tts. Install it with `pip install f5-tts`."
                ) from exc

            kwargs: Dict[str, Any] = {
                "ckpt_file": str(self.model_path),
                "vocab_file": str(self.vocab_path),
            }
            if self.device:
                kwargs["device"] = self.device

            self._engine = F5TTS(**kwargs)

    def synthesize(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str,
        speed: float,
        nfe_step: int,
        remove_silence: bool,
    ) -> Tuple[np.ndarray, int]:
        self._lazy_load()

        # F5-TTS infer path is not safe for concurrent access in this API setup.
        # Serialize per-chunk infer calls to avoid internal cache/tensor mismatches.
        with self._engine_lock:
            result = self._engine.infer(
                ref_file=reference_audio_path,
                ref_text=reference_text,
                gen_text=text,
                nfe_step=nfe_step,
                speed=speed,
                remove_silence=remove_silence,
            )

        audio, sample_rate = _coerce_audio_result(result)
        return audio, sample_rate


def _coerce_audio_result(result: Any) -> Tuple[np.ndarray, int]:
    sample_rate = 24000
    audio = None

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, (np.ndarray, list, tuple)) and audio is None:
                audio = np.asarray(item)
            elif isinstance(item, (int, np.integer)):
                sample_rate = int(item)
            elif isinstance(item, str) and Path(item).exists():
                audio, sample_rate = _read_wave_file(Path(item))

    elif isinstance(result, dict):
        maybe_audio = result.get("audio") or result.get("wav")
        if maybe_audio is not None:
            audio = np.asarray(maybe_audio)
        sample_rate = int(result.get("sample_rate", result.get("sr", sample_rate)))

        maybe_path = result.get("audio_path") or result.get("wav_path")
        if audio is None and maybe_path and Path(maybe_path).exists():
            audio, sample_rate = _read_wave_file(Path(maybe_path))

    elif isinstance(result, str) and Path(result).exists():
        audio, sample_rate = _read_wave_file(Path(result))

    if audio is None:
        raise RuntimeError("Could not parse audio output from F5-TTS inference result")

    if audio.ndim > 1:
        audio = np.squeeze(audio)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Model generated empty audio")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio, sample_rate


def _read_wave_file(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(n_frames)

    if sample_width != 2:
        raise RuntimeError(f"Unsupported sample width in wav file: {sample_width}")

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    return pcm, sample_rate


def _float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = _float32_to_pcm16(audio)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    return _float32_to_pcm16(audio).tobytes()


def _encode_compressed(audio: np.ndarray, sample_rate: int, format_name: str) -> bytes:
    tensor = torch.from_numpy(audio).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, tensor, sample_rate, format=format_name)
    return buffer.getvalue()


def _resolve_reference_audio_path(raw_path: str, catalog_path: Path) -> str:
    path_obj = Path(raw_path).expanduser()
    candidates = []

    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.extend(
            [
                Path.cwd() / path_obj,
                catalog_path.parent / path_obj,
                BASE_DIR / path_obj,
                BASE_DIR / "ex_sound" / path_obj.name,
            ]
        )

    resolved_candidates = [candidate.resolve(strict=False) for candidate in candidates]
    for candidate in resolved_candidates:
        if candidate.exists():
            return str(candidate)

    # Keep deterministic output for diagnostics if nothing exists.
    if resolved_candidates:
        return str(resolved_candidates[0])
    return raw_path


def _split_text_for_inference(text: str, max_chars: int) -> list[str]:
    def _is_thai_char(ch: str) -> bool:
        """Check if character is Thai."""
        return "\u0E00" <= ch <= "\u0E7F"
    
    def _is_thai_combining_mark(ch: str) -> bool:
        """Check if character is a Thai combining mark (vowel, tone, etc)."""
        # Thai vowels above, below, and combining marks
        thai_combining = "\u0E31\u0E34\u0E35\u0E36\u0E37\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D"
        return ch in thai_combining or unicodedata.combining(ch) != 0
    
    def _find_safe_thai_break(s: str, end: int) -> int:
        """Find a safe break point for Thai text by backtracking if needed."""
        if end <= 0 or end > len(s):
            return end
        
        # If we're about to cut right before a Thai combining mark, move forward
        while end < len(s) and _is_thai_combining_mark(s[end]):
            end += 1
        
        # If we're cutting in the middle of Thai characters, try to backtrack
        # to find a safe boundary (space, punctuation, or complete syllable)
        if end > 0 and _is_thai_char(s[end - 1]):
            # Look back from the potential cut point to find a safe break
            lookback_pos = end - 1
            
            # Skip back over Thai combining marks to find the consonant
            while lookback_pos > 0 and _is_thai_combining_mark(s[lookback_pos]):
                lookback_pos -= 1
            
            # Now lookback_pos is at a Thai consonant or the character before Thai text
            # Check if there's a space or punctuation that would be a natural break
            search_pos = lookback_pos
            while search_pos > 0:
                if s[search_pos] == ' ' or s[search_pos] in ".,!?;:\n":
                    # Found a natural boundary (punctuation or space)
                    return search_pos + 1
                elif not _is_thai_char(s[search_pos]):
                    # Found a non-Thai character (but not space/punct), safe to break after previous Thai
                    return search_pos + 1
                else:
                    # It's Thai, keep looking back
                    if search_pos > 0 and _is_thai_combining_mark(s[search_pos - 1]):
                        search_pos -= 1
                    else:
                        search_pos -= 1
                        if search_pos == 0:
                            break
                        if not _is_thai_char(s[search_pos]):
                            return search_pos + 1
        
        return end
    
    def _take_safe_segment(s: str, start: int, max_len: int) -> tuple[str, int]:
        end = min(len(s), start + max_len)
        
        # Smart Thai-aware break point detection
        end = _find_safe_thai_break(s, end)
        
        # Also respect combining marks that come after (Unicode normalization safety)
        while end < len(s) and unicodedata.combining(s[end]) != 0:
            end += 1
        
        segment = s[start:end]
        return segment, end

    # Normalize into canonical composed form to keep Thai marks stable.
    normalized = unicodedata.normalize("NFC", text)
    clean_text = re.sub(r"\s+", " ", normalized.strip())
    if not clean_text:
        return []

    if len(clean_text) <= max_chars:
        return [clean_text]

    sentence_like_parts = re.split(r"(?<=[\.!?\n。！？])\s+", clean_text)
    chunks: list[str] = []
    current = ""

    for part in sentence_like_parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            words = part.split(" ")
            if len(words) == 1:
                i = 0
                while i < len(part):
                    segment, next_i = _take_safe_segment(part, i, max_chars)
                    if not segment:
                        break
                    chunks.append(segment)
                    i = next_i
                continue

            for word in words:
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(word) > max_chars:
                        i = 0
                        while i < len(word):
                            segment, next_i = _take_safe_segment(word, i, max_chars)
                            if not segment:
                                break
                            chunks.append(segment)
                            i = next_i
                        current = ""
                    else:
                        current = word
            continue

        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def _is_tensor_mismatch_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Sizes of tensors must match" in msg and "Expected size" in msg


def _sanitize_output_filename(filename: str) -> str:
    stem = Path(filename).stem.strip()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    if not safe_stem:
        safe_stem = "speech"
    return f"{safe_stem}.wav"


THAI_VOWEL_CHARS = set("ะาำิีึืุูเแโใไๅ")
THAI_TONE_MARKS = set("่้๊๋")


def _auto_tune_realtime_preset() -> Dict[str, float]:
    has_cuda = bool(torch.cuda.is_available())
    cpu_cores = max(1, os.cpu_count() or 1)

    # Prioritize latency for conversational, real-time interactions.
    if has_cuda:
        preset: Dict[str, float] = {
            "nfe_step": 4,
            "max_chars": 120,
            "chunk_silence": 0.0,
            "speed_mult": 1.22,
            "min_chars": 24,
        }
    else:
        preset = {
            "nfe_step": 5,
            "max_chars": 90,
            "chunk_silence": 0.0,
            "speed_mult": 1.25,
            "min_chars": 22,
        }

    # Use available CPU capacity for preprocessing and encoding.
    try:
        torch.set_num_threads(cpu_cores)
    except Exception:
        pass

    if has_cuda:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    logger.info(
        "Realtime preset auto-tuned | has_cuda=%s | cpu_cores=%d | nfe_step=%d | max_chars=%d | speed_mult=%.2f",
        has_cuda,
        cpu_cores,
        int(preset["nfe_step"]),
        int(preset["max_chars"]),
        float(preset["speed_mult"]),
    )
    return preset


def _get_latency_preset_defaults(preset: str) -> Dict[str, float]:
    presets: Dict[str, Dict[str, float]] = {
        "quality": {
            "nfe_step": 24,
            "max_chars": 180,
            "chunk_silence": 0.06,
            "speed_mult": 1.0,
            "min_chars": 45,
        },
        "balanced": {
            "nfe_step": 16,
            "max_chars": 170,
            "chunk_silence": 0.04,
            "speed_mult": 1.0,
            "min_chars": 40,
        },
        "fast": {
            "nfe_step": 10,
            "max_chars": 150,
            "chunk_silence": 0.03,
            "speed_mult": 1.08,
            "min_chars": 35,
        },
        "ultra": {
            "nfe_step": 6,
            "max_chars": 130,
            "chunk_silence": 0.02,
            "speed_mult": 1.15,
            "min_chars": 30,
        },
        "realtime_max": _auto_tune_realtime_preset(),
    }
    return presets.get(preset, presets["ultra"])


def _resolve_latency_runtime_values(
    latency_preset_override: Optional[str],
) -> Tuple[str, int, int, int, float, float]:
    if latency_preset_override is None:
        return (
            LATENCY_PRESET,
            NFE_STEP,
            MAX_CHARS_PER_CHUNK,
            MIN_CHARS_PER_CHUNK,
            CHUNK_SILENCE_SECONDS,
            SPEED_MULTIPLIER,
        )

    preset = latency_preset_override.strip().lower()
    if preset not in {"quality", "balanced", "fast", "ultra", "realtime_max"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "latency_preset must be one of: quality, balanced, fast, ultra, realtime_max",
                    "type": "invalid_request_error",
                    "param": "latency_preset",
                    "code": "invalid_latency_preset",
                }
            },
        )

    defaults = _get_latency_preset_defaults(preset)
    return (
        preset,
        int(defaults["nfe_step"]),
        int(defaults["max_chars"]),
        int(defaults["min_chars"]),
        float(defaults["chunk_silence"]),
        float(defaults["speed_mult"]),
    )


def _analyze_input_text(text: str) -> Dict[str, Any]:
    normalized = unicodedata.normalize("NFC", text)
    text_len = len(normalized)

    thai_chars = [ch for ch in normalized if "\u0E00" <= ch <= "\u0E7F"]
    thai_len = len(thai_chars)
    thai_ratio = (thai_len / text_len) if text_len > 0 else 0.0

    vowel_count = sum(1 for ch in normalized if ch in THAI_VOWEL_CHARS)
    tone_count = sum(1 for ch in normalized if ch in THAI_TONE_MARKS)
    combining_count = sum(1 for ch in normalized if unicodedata.combining(ch) != 0)

    has_thai = thai_len > 0
    has_vowel = vowel_count > 0
    suspicious_missing_marks = has_thai and vowel_count == 0 and tone_count == 0 and combining_count == 0

    preview = normalized[:120].replace("\n", " ")
    codepoint_preview = " ".join(f"U+{ord(ch):04X}" for ch in normalized[:40])

    return {
        "text_len": text_len,
        "thai_len": thai_len,
        "thai_ratio": thai_ratio,
        "vowel_count": vowel_count,
        "tone_count": tone_count,
        "combining_count": combining_count,
        "has_thai": has_thai,
        "has_vowel": has_vowel,
        "suspicious_missing_marks": suspicious_missing_marks,
        "preview": preview,
        "codepoint_preview": codepoint_preview,
    }


def _load_voice_catalog(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("voices.json must contain a JSON object")

    catalog: Dict[str, Dict[str, str]] = {}
    for voice_name, value in data.items():
        if not isinstance(value, dict):
            continue
        ref_wav = value.get("reference_audio_path")
        ref_text = value.get("reference_text")
        if isinstance(ref_wav, str) and isinstance(ref_text, str):
            resolved_ref_wav = _resolve_reference_audio_path(ref_wav, path)
            if resolved_ref_wav != ref_wav:
                logger.info(
                    "Resolved reference path from catalog | voice=%s | raw=%s | resolved=%s",
                    voice_name,
                    ref_wav,
                    resolved_ref_wav,
                )
            catalog[voice_name] = {
                "reference_audio_path": resolved_ref_wav,
                "reference_text": ref_text,
            }
    return catalog


def _resolve_reference(
    request: SpeechRequest,
    catalog: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    if request.reference_audio_path and request.reference_text:
        logger.info(
            "Using inline reference data | voice=%s | ref_audio_path=%s",
            request.voice,
            request.reference_audio_path,
        )
        return request.reference_audio_path, request.reference_text

    profile = catalog.get(request.voice)
    if profile:
        logger.info(
            "Using voice profile from catalog | voice=%s | ref_audio_path=%s",
            request.voice,
            profile["reference_audio_path"],
        )
        return profile["reference_audio_path"], profile["reference_text"]

    # Compatibility mode for clients that send OpenAI-style synthetic IDs
    # (for example "voice_id_0") instead of catalog keys.
    voice_names = sorted(catalog.keys())
    alias_match = re.fullmatch(r"voice_id_(\d+)", request.voice or "")
    if alias_match and voice_names:
        idx = int(alias_match.group(1))
        if 0 <= idx < len(voice_names):
            mapped_voice = voice_names[idx]
            mapped_profile = catalog[mapped_voice]
            logger.warning(
                "Mapped synthetic voice id to catalog voice | requested_voice=%s | mapped_voice=%s",
                request.voice,
                mapped_voice,
            )
            return mapped_profile["reference_audio_path"], mapped_profile["reference_text"]

    default_voice = os.getenv("JAITTS_DEFAULT_VOICE", "alloy")
    if voice_names:
        fallback_voice = default_voice if default_voice in catalog else voice_names[0]
        fallback_profile = catalog[fallback_voice]
        logger.warning(
            "Unknown voice, falling back to configured default | requested_voice=%s | fallback_voice=%s",
            request.voice,
            fallback_voice,
        )
        return fallback_profile["reference_audio_path"], fallback_profile["reference_text"]

    available_voices = sorted(catalog.keys())
    logger.warning(
        "Voice not configured | requested_voice=%s | catalog_path=%s | available_voices=%s",
        request.voice,
        VOICE_CATALOG_PATH,
        available_voices,
    )

    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    "No reference voice data found. Provide both reference_audio_path and "
                    "reference_text, or configure this voice in voices.json. "
                    f"Available voices: {available_voices}"
                ),
                "type": "invalid_request_error",
                "param": "voice",
                "code": "voice_not_configured",
            }
        },
    )


def _require_api_key(authorization: Optional[str], expected_key: Optional[str]) -> None:
    if not expected_key:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Bearer token",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    provided_key = authorization.split(" ", 1)[1].strip()
    if provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )


app = FastAPI(title="JaiTTS OpenAI-Compatible TTS API", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path(os.getenv("JAITTS_MODEL_PATH", BASE_DIR / "model" / "JaiTTS-F5TTS" / "model.pt"))
DEFAULT_VOCAB_PATH = Path(os.getenv("JAITTS_VOCAB_PATH", BASE_DIR / "model" / "JaiTTS-F5TTS" / "vocab.txt"))
VOICE_CATALOG_PATH = Path(os.getenv("JAITTS_VOICES_JSON", BASE_DIR / "voices.json"))
EXPECTED_API_KEY = os.getenv("OPENAI_API_KEY")
DEVICE = os.getenv("JAITTS_DEVICE")
# ปรับคุณภาพของเสียงตาม preset ที่กำหนด (quality, balanced, fast, ultra, realtime_max)
LATENCY_PRESET = os.getenv("JAITTS_LATENCY_PRESET", "realtime_max").lower()
_PRESET_DEFAULTS = _get_latency_preset_defaults(LATENCY_PRESET)

NFE_STEP = int(os.getenv("JAITTS_NFE_STEP", str(int(_PRESET_DEFAULTS["nfe_step"]))))
MAX_CHARS_PER_CHUNK = int(os.getenv("JAITTS_MAX_CHARS_PER_CHUNK", str(int(_PRESET_DEFAULTS["max_chars"]))))
REMOVE_SILENCE = os.getenv("JAITTS_REMOVE_SILENCE", "0").lower() in {"1", "true", "yes"}
CHUNK_SILENCE_SECONDS = float(
    os.getenv("JAITTS_CHUNK_SILENCE_SECONDS", str(float(_PRESET_DEFAULTS["chunk_silence"])))
)
MIN_CHARS_PER_CHUNK = int(os.getenv("JAITTS_MIN_CHARS_PER_CHUNK", str(int(_PRESET_DEFAULTS["min_chars"]))))
SPEED_MULTIPLIER = float(os.getenv("JAITTS_SPEED_MULTIPLIER", str(float(_PRESET_DEFAULTS["speed_mult"]))))
SAVE_WAV_DIR = Path(os.getenv("JAITTS_SAVE_WAV_DIR", BASE_DIR / "save_wav"))

backend = JaiTTSBackend(
    model_path=DEFAULT_MODEL_PATH,
    vocab_path=DEFAULT_VOCAB_PATH,
    device=DEVICE,
)
voice_catalog = _load_voice_catalog(VOICE_CATALOG_PATH)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _synthesize_request_audio(
    request: SpeechRequest,
    latency_preset_override: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    (
        runtime_preset,
        runtime_nfe_step,
        runtime_max_chars,
        runtime_min_chars,
        runtime_chunk_silence,
        runtime_speed_multiplier,
    ) = _resolve_latency_runtime_values(latency_preset_override)

    effective_speed = request.speed * runtime_speed_multiplier

    if not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "input must not be empty",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "invalid_input",
                }
            },
        )

    input_stats = _analyze_input_text(request.input)
    logger.info(
        "Input text analysis | len=%d | thai_chars=%d | thai_ratio=%.3f | vowels=%d | tones=%d | combining=%d | has_vowel=%s",
        input_stats["text_len"],
        input_stats["thai_len"],
        input_stats["thai_ratio"],
        input_stats["vowel_count"],
        input_stats["tone_count"],
        input_stats["combining_count"],
        input_stats["has_vowel"],
    )
    logger.info("Input preview | text=%s", input_stats["preview"])

    if input_stats["suspicious_missing_marks"]:
        logger.warning(
            "Input may be missing Thai marks from upstream | codepoints=%s",
            input_stats["codepoint_preview"],
        )

    if request.speed <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "speed must be > 0",
                    "type": "invalid_request_error",
                    "param": "speed",
                    "code": "invalid_speed",
                }
            },
        )

    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_VOCAB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Model files not found. Check JAITTS_MODEL_PATH and JAITTS_VOCAB_PATH.",
                    "type": "server_error",
                    "param": None,
                    "code": "model_not_found",
                }
            },
        )

    ref_wav, ref_text = _resolve_reference(request, voice_catalog)
    if not Path(ref_wav).exists():
        logger.warning(
            "Reference audio not found | voice=%s | ref_audio_path=%s",
            request.voice,
            ref_wav,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"reference_audio_path does not exist: {ref_wav}",
                    "type": "invalid_request_error",
                    "param": "reference_audio_path",
                    "code": "missing_reference_audio",
                }
            },
        )

    text_chunks = _split_text_for_inference(request.input, runtime_max_chars)
    if not text_chunks:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "input must not be empty",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "invalid_input",
                }
            },
        )

    logger.info(
        "Prepared chunks | count=%d | max_chars_per_chunk=%d | nfe_step=%d | remove_silence=%s | preset=%s",
        len(text_chunks),
        runtime_max_chars,
        runtime_nfe_step,
        REMOVE_SILENCE,
        runtime_preset,
    )

    try:
        parts: list[np.ndarray] = []
        sample_rate: Optional[int] = None
        t0 = time.perf_counter()
        for idx, chunk in enumerate(text_chunks, start=1):
            pending_subchunks = [chunk]
            while pending_subchunks:
                current_chunk = pending_subchunks.pop(0)
                chunk_t0 = time.perf_counter()
                try:
                    chunk_audio, chunk_sample_rate = backend.synthesize(
                        text=current_chunk,
                        reference_audio_path=ref_wav,
                        reference_text=ref_text,
                        speed=effective_speed,
                        nfe_step=runtime_nfe_step,
                        remove_silence=REMOVE_SILENCE,
                    )
                except Exception as exc:
                    if _is_tensor_mismatch_error(exc) and len(current_chunk) > runtime_min_chars:
                        smaller_limit = max(runtime_min_chars, len(current_chunk) // 2)
                        smaller_chunks = _split_text_for_inference(current_chunk, smaller_limit)
                        if len(smaller_chunks) > 1:
                            logger.warning(
                                "Chunk retry with smaller splits | parent_chars=%d | new_count=%d | new_max_chars=%d",
                                len(current_chunk),
                                len(smaller_chunks),
                                smaller_limit,
                            )
                            pending_subchunks = smaller_chunks + pending_subchunks
                            continue
                    raise

                if sample_rate is None:
                    sample_rate = chunk_sample_rate
                elif sample_rate != chunk_sample_rate:
                    raise RuntimeError(
                        f"Inconsistent sample rates across chunks: {sample_rate} vs {chunk_sample_rate}"
                    )

                parts.append(chunk_audio.astype(np.float32))

                chunk_duration = float(chunk_audio.shape[0]) / float(chunk_sample_rate)
                elapsed = time.perf_counter() - chunk_t0
                logger.info(
                    "Chunk synthesized | index=%d/%d | chars=%d | duration_sec=%.2f | elapsed_sec=%.2f",
                    idx,
                    len(text_chunks),
                    len(current_chunk),
                    chunk_duration,
                    elapsed,
                )

        if sample_rate is None:
            raise RuntimeError("No audio generated from chunks")

        if len(parts) == 1:
            audio = parts[0]
        else:
            silence_len = max(0, int(sample_rate * runtime_chunk_silence))
            silence = np.zeros(silence_len, dtype=np.float32)
            interleaved: list[np.ndarray] = []
            for i, part in enumerate(parts):
                interleaved.append(part)
                if i < len(parts) - 1 and silence_len > 0:
                    interleaved.append(silence)
            audio = np.concatenate(interleaved)

        total_elapsed = time.perf_counter() - t0
        logger.info(
            "Synthesis success | voice=%s | sample_rate=%d | samples=%d | chunks=%d | elapsed_sec=%.2f",
            request.voice,
            sample_rate,
            int(audio.shape[0]),
            len(text_chunks),
            total_elapsed,
        )
        return audio, sample_rate
    except Exception as exc:
        logger.exception("Synthesis failed | voice=%s", request.voice)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Synthesis failed: {exc}",
                    "type": "server_error",
                    "param": None,
                    "code": "synthesis_failed",
                }
            },
        ) from exc


@app.post("/v1/audio/speech")
def create_speech(
    request: SpeechRequest,
    authorization: Optional[str] = Header(default=None),
):
    effective_speed = request.speed * SPEED_MULTIPLIER

    logger.info(
        "Incoming speech request | model=%s | voice=%s | format=%s | speed=%.3f | effective_speed=%.3f | input_len=%d",
        request.model,
        request.voice,
        request.response_format,
        request.speed,
        effective_speed,
        len(request.input or ""),
    )

    _require_api_key(authorization, EXPECTED_API_KEY)

    format_name = request.response_format.lower()
    if format_name not in {"wav", "pcm", "mp3", "opus", "aac", "flac"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Supported response_format values are wav, pcm, mp3, opus, aac, flac",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "unsupported_audio_format",
                }
            },
        )

    audio, sample_rate = _synthesize_request_audio(
        request,
        latency_preset_override=request.latency_preset,
    )

    if format_name == "wav":
        payload = _encode_wav(audio, sample_rate)
        media_type = "audio/wav"
    elif format_name == "pcm":
        payload = _encode_pcm(audio)
        media_type = "audio/pcm"
    else:
        try:
            payload = _encode_compressed(audio, sample_rate, format_name)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Failed to encode {format_name}: {exc}",
                        "type": "server_error",
                        "param": "response_format",
                        "code": "audio_encode_failed",
                    }
                },
            ) from exc
        media_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
        }[format_name]

    return StreamingResponse(io.BytesIO(payload), media_type=media_type)


@app.post("/v1/audio/speech/save")
def create_speech_and_save(
    request: SaveWavRequest,
    authorization: Optional[str] = Header(default=None),
):
    _require_api_key(authorization, EXPECTED_API_KEY)

    speech_request = SpeechRequest(
        model="jaitts-v1",
        input=request.text,
        voice=request.voice,
        response_format="wav",
        speed=request.speed,
        reference_audio_path=request.reference_audio_path,
        reference_text=request.reference_text,
    )

    logger.info(
        "Incoming save request | voice=%s | speed=%.3f | input_len=%d | latency_preset=%s",
        speech_request.voice,
        speech_request.speed,
        len(speech_request.input or ""),
        request.latency_preset,
    )

    audio, sample_rate = _synthesize_request_audio(
        speech_request,
        latency_preset_override=request.latency_preset,
    )
    wav_payload = _encode_wav(audio, sample_rate)

    SAVE_WAV_DIR.mkdir(parents=True, exist_ok=True)
    if request.filename:
        out_name = _sanitize_output_filename(request.filename)
    else:
        out_name = f"speech_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"

    out_path = SAVE_WAV_DIR / out_name
    out_path.write_bytes(wav_payload)

    duration_sec = float(audio.shape[0]) / float(sample_rate)
    logger.info(
        "Saved wav file | path=%s | sample_rate=%d | duration_sec=%.2f",
        out_path,
        sample_rate,
        duration_sec,
    )

    return {
        "status": "ok",
        "file_path": str(out_path),
        "filename": out_name,
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 3),
        "samples": int(audio.shape[0]),
    }


@app.exception_handler(HTTPException)
def http_exception_handler(_, exc: HTTPException):
    logger.warning(
        "HTTPException returned | status=%s | detail=%s",
        exc.status_code,
        exc.detail,
    )
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )
