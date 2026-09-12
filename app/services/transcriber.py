from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import logging
import subprocess
import sys
from typing import Callable, Optional

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single timed segment from Whisper."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Structured result returned by transcribe_file()."""
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None


def _check_ffmpeg_available() -> bool:
    """Check whether ffmpeg is available on this system.

    Strategy:
    1. Look for a local .bin directory (created by setup_ffmpeg.py) and add it to PATH.
    2. Fall back to the global PATH.
    """
    project_root = Path(__file__).resolve().parents[2]
    local_bin = project_root / ".bin"

    is_windows = sys.platform.startswith("win")
    exe_name = "ffmpeg.exe" if is_windows else "ffmpeg"
    local_ffmpeg = local_bin / exe_name

    if local_ffmpeg.exists():
        logger.info(f"Local ffmpeg found at: {local_ffmpeg}")
        current_path = os.environ.get("PATH", "")
        if str(local_bin) not in current_path:
            os.environ["PATH"] = str(local_bin) + os.pathsep + current_path
            logger.info("Added .bin directory to PATH")
        return True

    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            logger.info("ffmpeg found in global PATH")
            return True
        except Exception:
            pass

    logger.warning("ffmpeg not found. Run the setup script or install it manually.")
    return False


def _probe_audio_stream(media_path: Path) -> bool:
    """Check if a media file contains at least one audio stream.

    Uses ffprobe if available (checking the local .bin dir first, then PATH),
    otherwise falls back to parsing ``ffmpeg -i`` stderr output.

    Args:
        media_path: Path to the media file to probe.

    Returns:
        True if the file has an audio stream, False otherwise.
    """
    # Prefer the local .bin ffprobe so probe and extraction use the same binary.
    project_root = Path(__file__).resolve().parents[2]
    probe_exe = "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"

    ffprobe_bin: Optional[str] = None
    local_probe = project_root / ".bin" / probe_exe
    if local_probe.exists():
        ffprobe_bin = str(local_probe)
    else:
        ffprobe_bin = shutil.which("ffprobe")

    if ffprobe_bin:
        try:
            result = subprocess.run(
                [
                    ffprobe_bin,
                    "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(media_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            has_audio = bool(result.stdout.strip())
            if not has_audio:
                logger.warning(f"No audio stream found (ffprobe): {media_path}")
                logger.debug(f"ffprobe stdout: {result.stdout}")
                logger.debug(f"ffprobe stderr: {result.stderr}")
            return has_audio
        except Exception as probe_err:
            logger.debug(f"ffprobe failed, trying fallback: {probe_err}")

    # Fallback: parse ffmpeg -i stderr for "Audio:" stream lines.
    # ffmpeg is guaranteed to be in PATH (called after _check_ffmpeg_available).
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(media_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        has_audio = "Audio:" in combined
        if not has_audio:
            logger.warning(f"No audio stream found (ffmpeg -i): {media_path}")
            logger.debug(f"ffmpeg combined output: {combined}")
        return has_audio
    except Exception as e:
        logger.debug(f"Audio probe failed: {e}")
        # Cannot determine — assume audio present; let extraction fail naturally.
        return True


def _extract_audio_to_wav(media_path: Path) -> Path:
    """Extract audio from a media file (video or audio) to 16 kHz mono WAV.

    Tries three progressively more permissive ffmpeg strategies so that
    unusual containers (fragmented MP4, DASH with muxed audio, MKV with
    non-default stream ordering, etc.) are handled correctly.

    Strategy 1 – explicit audio map:
        ``ffmpeg -i <input> -map 0:a:0 -acodec pcm_s16le -ar 16000 -ac 1``
        Selects the first audio stream by index; most reliable for files that
        have audio embedded but not as the default stream.

    Strategy 2 – strip video, let ffmpeg pick audio:
        ``ffmpeg -i <input> -vn -acodec pcm_s16le -ar 16000 -ac 1``
        Classic approach; fails on video-only DASH files.

    Strategy 3 – copy all streams, no filter:
        ``ffmpeg -i <input> -map 0 -acodec pcm_s16le -ar 16000 -ac 1``
        Last resort; picks up any stream that can be decoded as audio.

    Args:
        media_path: Path to the input media file.

    Returns:
        Path to the extracted WAV file.

    Raises:
        RuntimeError: If all extraction strategies fail.
    """
    wav_path = media_path.with_suffix(".wav")

    if wav_path.exists():
        wav_path.unlink()

    logger.info(f"Extracting audio from {media_path} → {wav_path}")

    common_tail = [
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(wav_path),
    ]

    # Collect all ffmpeg binaries to try: local .bin first, then system PATH
    ffmpeg_binaries: list[str] = []
    project_root = Path(__file__).resolve().parents[2]
    ffmpeg_exe = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    local_ffmpeg = project_root / ".bin" / ffmpeg_exe
    if local_ffmpeg.exists():
        ffmpeg_binaries.append(str(local_ffmpeg))
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg and sys_ffmpeg not in ffmpeg_binaries:
        ffmpeg_binaries.append(sys_ffmpeg)
    if not ffmpeg_binaries:
        ffmpeg_binaries = ["ffmpeg"]  # last resort: rely on PATH

    strategies: list[tuple[str, list[str]]] = []
    for binary in ffmpeg_binaries:
        binary_label = Path(binary).parent.name  # ".bin" or "bin"
        base = [binary, "-i", str(media_path)]
        strategies += [
            (f"[{binary_label}] map 0:a:0",  base + ["-map", "0:a:0"] + common_tail),
            (f"[{binary_label}] -vn",         base + ["-vn"]           + common_tail),
            (f"[{binary_label}] map 0",       base + ["-map", "0"]     + common_tail),
        ]

    last_error: Optional[str] = None

    for strategy_name, cmd in strategies:
        try:
            logger.info(f"Audio extraction strategy: {strategy_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                logger.info(
                    f"Audio extracted successfully with strategy '{strategy_name}': {wav_path}"
                )
                return wav_path

            # Non-zero return or empty WAV — log and try next strategy
            err_msg = result.stderr[-400:] if result.stderr else "(no stderr)"
            logger.warning(
                f"Strategy '{strategy_name}' failed (rc={result.returncode}): ...{err_msg}"
            )
            last_error = err_msg

            # Clean up an empty/corrupt WAV before next attempt
            if wav_path.exists():
                wav_path.unlink()

        except subprocess.TimeoutExpired:
            logger.warning(f"Strategy '{strategy_name}' timed out")
            last_error = "timeout"
            if wav_path.exists():
                wav_path.unlink()
        except Exception as exc:
            logger.warning(f"Strategy '{strategy_name}' raised exception: {exc}")
            last_error = str(exc)
            if wav_path.exists():
                wav_path.unlink()

    raise RuntimeError(
        f"Audio extraction failed after all strategies. Last error: {last_error}"
    )



def _resolve_faster_whisper_model_path(model_name: str) -> str:
    """Resolve a faster-whisper model name to its local HuggingFace hub snapshot path.

    Enables fully offline operation by bypassing HuggingFace Hub network calls.
    If model_name is already a valid local path it is returned as-is.

    Args:
        model_name: Whisper model size (e.g. 'base', 'small') or a direct path.

    Returns:
        Local path string to pass to WhisperModel(), or the original model_name
        if no cached snapshot is found (may fail offline).
    """
    if Path(model_name).exists():
        return model_name

    hf_cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    repo_dir_name = f"models--Systran--faster-whisper-{model_name}"
    repo_path = hf_cache_dir / repo_dir_name

    if repo_path.is_dir():
        snapshots_dir = repo_path / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(snapshots_dir.iterdir())
            if snapshots:
                snapshot_path = snapshots[0]
                if (snapshot_path / "model.bin").exists():
                    logger.info(
                        f"faster-whisper: using local snapshot at {snapshot_path}"
                    )
                    return str(snapshot_path)

    logger.warning(
        f"faster-whisper: no local snapshot found for '{model_name}'. "
        "Will attempt network resolution (may fail in offline mode)."
    )
    return model_name


def transcribe_file(
    media_path: Path,
    model_name: str | None = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> TranscriptionResult:
    """Transcribe a media file to text using a Whisper backend.

    Priority:
    1) openai-whisper  (when ffmpeg is available)
    2) faster-whisper  (always available as fallback)

    Returns a TranscriptionResult with full text, per-segment timestamps, and
    detected language.

    Args:
        media_path: Path to the media file.
        model_name: Whisper model size (base, small, medium, etc.). Defaults to config.
        progress_callback: Optional function receiving (percent: int, message: str).
    """
    if model_name is None:
        model_name = settings.whisper_model_default

    if progress_callback:
        progress_callback(0, "Starting transcription...")

    # Check if we're in a Vercel/serverless environment
    is_serverless = os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")

    if is_serverless:
        # For Vercel/serverless, use a simplified approach or API
        logger.info("Running in serverless environment - using lightweight transcription")
        return _transcribe_serverless(media_path, progress_callback)

    ffmpeg_available = _check_ffmpeg_available()

    if not ffmpeg_available:
        logger.info("ffmpeg unavailable; openai-whisper disabled, using faster-whisper only")
    else:
        logger.info("ffmpeg available; openai-whisper enabled")

    force_device = settings.transcription_device

    if force_device == "cuda":
        gpu_available = bool(torch is not None and getattr(torch.cuda, "is_available", lambda: False)())
        if not gpu_available:
            logger.warning("TRANSCRIPTION_DEVICE=cuda configured but CUDA unavailable. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda"
    elif force_device == "cpu":
        device = "cpu"
        gpu_available = False
    else:  # auto
        gpu_available = bool(torch is not None and getattr(torch.cuda, "is_available", lambda: False)())
        device = "cuda" if gpu_available else "cpu"

    try:
        if torch is None:
            logger.info("torch unavailable; proceeding with CPU/compat mode")
        else:
            cuda_is_avail = getattr(torch.cuda, "is_available", lambda: False)()
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
            if gpu_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "(unknown)"
                logger.info(f"Using CUDA (GPU='{gpu_name}', torch.version.cuda={cuda_ver})")
            else:
                logger.info(f"Using CPU (torch.cuda.is_available={cuda_is_avail}, torch.version.cuda={cuda_ver})")
    except Exception:
        pass

    whisper_err = None
    wav_path = None

    # -----------------------------------------------------------------------
    # Audio extraction
    # -----------------------------------------------------------------------
    # Strategy: attempt extraction first (trying multiple ffmpeg approaches);
    # only if all strategies fail AND ffprobe confirms no audio stream do we
    # surface a clear user-facing error. This handles unusual containers such
    # as fragmented DASH, Matroska with non-standard codec tags, etc.
    if ffmpeg_available:
        try:
            if progress_callback:
                progress_callback(5, "Extracting audio from file...")
            wav_path = _extract_audio_to_wav(media_path)

            # Extra sanity check: WAV must have more than just a header (44 bytes)
            if wav_path and wav_path.exists() and wav_path.stat().st_size <= 44:
                logger.warning(f"Extracted WAV is empty: {wav_path}")
                wav_path.unlink()
                wav_path = None

        except Exception as extract_err:
            logger.error(f"Audio extraction failed: {extract_err}")
            wav_path = None

        if wav_path is None:
            # Confirm with probe before raising the user-facing error
            if not _probe_audio_stream(media_path):
                raise RuntimeError(
                    f"The file '{media_path.name}' does not contain an audio track. "
                    "Please make sure your video has audio before uploading.\n"
                    "Tip: Videos downloaded from YouTube in DASH format often have "
                    "no embedded audio. Re-download the video using a tool such as "
                    "yt-dlp, or use the YouTube URL feature in this app instead."
                )
            # Probe says audio exists but extraction still failed — continue;
            # faster-whisper will also fail and surface a cleaner error below.

    # -----------------------------------------------------------------------
    # Backend 1: openai-whisper
    # -----------------------------------------------------------------------
    if ffmpeg_available and wav_path and wav_path.exists():
        try:
            import whisper  # type: ignore
            logger.info("Trying openai-whisper backend...")
            if progress_callback:
                progress_callback(10, "Loading openai-whisper model...")

            model = whisper.load_model(
                model_name,
                device=device,
                download_root=os.path.expanduser("~/.cache/whisper"),
            )

            if progress_callback:
                progress_callback(20, "Processing audio (this may take a while)...")

            raw = model.transcribe(str(wav_path), fp16=(device == "cuda"))

            if progress_callback:
                progress_callback(100, "Transcription complete!")

            logger.info(f"backend=openai-whisper device={device} fp16={device == 'cuda'} model={model_name}")

            segments = [
                TranscriptionSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                )
                for seg in raw.get("segments", [])
            ]

            if wav_path.exists():
                wav_path.unlink()

            return TranscriptionResult(
                text=raw.get("text", "").strip(),
                segments=segments,
                language=raw.get("language"),
            )
        except Exception as e1:
            whisper_err = e1
            logger.exception("openai-whisper backend failed", exc_info=e1)
    else:
        if not ffmpeg_available:
            whisper_err = RuntimeError("ffmpeg unavailable; openai-whisper skipped")
        elif not wav_path or not wav_path.exists():
            whisper_err = RuntimeError("audio extraction unsuccessful")

    # -----------------------------------------------------------------------
    # Backend 2: faster-whisper (fallback)
    # -----------------------------------------------------------------------
    try:
        from faster_whisper import WhisperModel  # type: ignore
        logger.info("Trying faster-whisper backend...")

        if progress_callback:
            progress_callback(10, "Loading faster-whisper model...")

        model = None
        fw_model_path = _resolve_faster_whisper_model_path(model_name)

        if gpu_available:
            try:
                model = WhisperModel(fw_model_path, device="cuda", compute_type="float16")
                logger.info("faster-whisper initialized with CUDA (compute_type=float16)")
            except Exception as cuda_init_err:
                logger.warning(
                    "faster-whisper CUDA unavailable/unsupported; falling back to CPU",
                    exc_info=cuda_init_err,
                )

        if model is None:
            model = WhisperModel(fw_model_path, device="cpu", compute_type="int8")
            logger.info("faster-whisper initialized with CPU (compute_type=int8)")

        if progress_callback:
            progress_callback(20, "Starting segmentation...")

        # Do NOT fall back to the raw media_path when wav_path is None —
        # a video-only file will cause PyAV's container.decode(audio=0) to
        # raise IndexError: tuple index out of range.
        if not wav_path or not wav_path.exists():
            raise RuntimeError(
                "Could not extract audio from the file. "
                "Please verify the file contains an audio track."
            )

        transcribe_input = wav_path
        logger.info(f"Transcribing from: {transcribe_input}")

        raw_segments, info = model.transcribe(str(transcribe_input))

        total_duration = info.duration
        text_parts: list[str] = []
        captured_segments: list[TranscriptionSegment] = []

        for seg in raw_segments:
            seg_text = seg.text.strip()
            text_parts.append(seg_text)
            captured_segments.append(
                TranscriptionSegment(start=float(seg.start), end=float(seg.end), text=seg_text)
            )
            if progress_callback and total_duration > 0:
                current_percent = 20 + int((seg.end / total_duration) * 75)
                current_percent = min(95, current_percent)
                progress_callback(
                    current_percent,
                    f"Transcribing: {int(seg.end)}s / {int(total_duration)}s",
                )

        text = " ".join(t for t in text_parts if t).strip()

        if progress_callback:
            progress_callback(100, "Finalizing...")

        if wav_path and wav_path.exists():
            wav_path.unlink()

        logger.info(
            f"backend=faster-whisper "
            f"device={'cuda' if (gpu_available and getattr(model, 'device', 'cpu') == 'cuda') else 'cpu'} "
            f"compute_type={'float16' if gpu_available else 'int8'} "
            f"model={model_name}"
        )
        return TranscriptionResult(
            text=text,
            segments=captured_segments,
            language=getattr(info, "language", None),
        )

    except Exception as e2:
        if wav_path and wav_path.exists():
            wav_path.unlink()

        logger.exception("faster-whisper backend failed", exc_info=e2)
        details = []
        if whisper_err is not None:
            details.append(f"openai-whisper: {type(whisper_err).__name__}: {whisper_err}")
        details.append(f"faster-whisper: {type(e2).__name__}: {e2}")
        raise RuntimeError(
            "No transcription backend available. "
            "Install 'openai-whisper' or 'faster-whisper'. "
            "| Details: " + " | ".join(details)
        ) from e2


def _transcribe_serverless(media_path: Path, progress_callback: Optional[Callable[[int, str], None]] = None) -> TranscriptionResult:
    """Lightweight transcription for serverless environments like Vercel.

    This function uses a minimal approach to work within size constraints.
    """
    if progress_callback:
        progress_callback(10, "Initializing lightweight transcription...")

    # For serverless, we'll use a basic approach or return a placeholder
    # In production, you might want to use an external API like:
    # - OpenAI Whisper API
    # - Google Cloud Speech-to-Text
    # - AssemblyAI
    # - Rev.ai

    # For now, return a demo result to show the UI works
    if progress_callback:
        progress_callback(50, "Processing audio...")

    if progress_callback:
        progress_callback(90, "Finalizing transcription...")

    # Return a demo transcription
    demo_text = "This is a demo transcription. In a production serverless environment, you would integrate with a cloud-based transcription API like OpenAI Whisper API, Google Cloud Speech-to-Text, or AssemblyAI to handle the actual transcription processing."

    segments = [
        TranscriptionSegment(start=0.0, end=2.0, text="This is a demo transcription."),
        TranscriptionSegment(start=2.0, end=5.0, text="In a production serverless environment,"),
        TranscriptionSegment(start=5.0, end=8.0, text="you would integrate with a cloud-based transcription API."),
        TranscriptionSegment(start=8.0, end=12.0, text="like OpenAI Whisper API, Google Cloud Speech-to-Text, or AssemblyAI."),
        TranscriptionSegment(start=12.0, end=15.0, text="to handle the actual transcription processing.")
    ]

    if progress_callback:
        progress_callback(100, "Transcription complete!")

    return TranscriptionResult(
        text=demo_text,
        segments=segments,
        language="en"
    )