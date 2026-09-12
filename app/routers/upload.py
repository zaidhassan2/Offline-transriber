from __future__ import annotations
from pathlib import Path
import logging
import uuid
import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from ..core.config import settings
from ..services.youtube import download_from_youtube
from ..services.transcriber import transcribe_file, TranscriptionResult
from ..services.file_manager import save_upload, save_transcription, get_unique_stem, sanitize_filename
from ..services.progress import progress_manager
from ..core.templates import templates

router = APIRouter(prefix="/transcribe")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task D — Allowed media extensions (server-side validation)
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    # Audio
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac",
    ".wma", ".aiff", ".au", ".mp2", ".amr", ".webm",
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".ts",
    ".m2ts", ".mts", ".3gp", ".ogv",
})


def _validate_extension(filename: str) -> None:
    """Raise HTTPException if the file extension is not in the allowed set."""
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        readable_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix or '(none)'}'. "
                f"Supported formats: {readable_list}"
            ),
        )


# ---------------------------------------------------------------------------
# Helper: format seconds → HH:MM:SS string for template rendering
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


async def process_transcription(task_id: str, media_path: Path, original_filename: str):
    """Função background para processar a transcrição e atualizar o progresso."""
    try:
        # Wrapper para adaptar a assinatura do callback (sync -> async bridge)
        def sync_callback(p, m):
            try:
                # Criar novo loop se necessário, pois estamos numa thread separada
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(progress_manager.update_progress(task_id, p, m))
                loop.close()
            except Exception as e:
                logger.error(f"Erro no callback de progresso: {e}")

        # Executar transcrição — now returns TranscriptionResult (Task C)
        result: TranscriptionResult = await asyncio.to_thread(
            transcribe_file, media_path, None, sync_callback
        )

        stem = get_unique_stem(original_filename)
        out_path = save_transcription(result.text, stem, ".txt")
        
        # Limpeza
        try:
            if media_path and media_path.exists():
                media_path.unlink()
        except Exception as e:
            logger.warning(f"Erro na limpeza: {e}")

        # Build timed segments list for the template
        timed_segments = [
            {
                "time": _fmt_time(seg.start),
                "text": seg.text,
            }
            for seg in result.segments
        ]

        # Atualizar status final — include timed segments + language in result payload
        await progress_manager.complete_task(task_id, {
            "filename": out_path.name,
            # Full plain text (for download / backwards compat)
            "text": result.text,
            # Timed segments for the progress view
            "segments": timed_segments,
            # Language detected by Whisper (may be None)
            "language": result.language,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Erro na tarefa {task_id}")
        await progress_manager.fail_task(task_id, str(e))
        # Tentar limpar
        if media_path and media_path.exists():
            try:
                media_path.unlink()
            except Exception:
                pass


@router.post("/youtube", response_class=HTMLResponse)
async def transcribe_youtube(request: Request, background_tasks: BackgroundTasks, url: str = Form(...)):
    try:
        task_id = str(uuid.uuid4())
        await progress_manager.create_task(task_id)

        # Update progress to show download started
        await progress_manager.update_progress(task_id, 5, "Downloading from YouTube...")

        # Download síncrono, mas rápido o suficiente para esperar antes de mostrar progresso
        media_path = await asyncio.to_thread(download_from_youtube, url)

        # Update progress to show download completed
        await progress_manager.update_progress(task_id, 10, f"Downloaded: {media_path.name}")

        background_tasks.add_task(process_transcription, task_id, media_path, media_path.name)

        return templates.TemplateResponse(request=request, name="progress.html", context={"task_id": task_id})

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        await progress_manager.fail_task(task_id, f"YouTube download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_class=HTMLResponse)
async def transcribe_upload(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Task D — Validate extension before doing any work
        _validate_extension(file.filename or "")

        task_id = str(uuid.uuid4())
        await progress_manager.create_task(task_id)

        # Update progress to show upload started
        await progress_manager.update_progress(task_id, 5, f"Uploading {file.filename}...")

        # Salvar arquivo temporário (sanitizando o nome para evitar problemas com ffmpeg)
        safe_filename = sanitize_filename(file.filename)
        temp_path = Path(settings.storage_uploads) / f"{task_id}_{safe_filename}"

        with temp_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        logger.info(f"File uploaded successfully: {file.filename} -> {temp_path}")

        # Update progress to show upload completed
        await progress_manager.update_progress(task_id, 10, f"File uploaded: {file.filename}")

        # Iniciar background task
        background_tasks.add_task(process_transcription, task_id, temp_path, file.filename)

        return templates.TemplateResponse(request=request, name="progress.html", context={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        await progress_manager.fail_task(task_id, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
