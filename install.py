import subprocess
import sys
import os
import shutil
from pathlib import Path

def setup_ffmpeg():
    print("\n[3/4] Setting up FFmpeg locally...")
    try:
        import imageio_ffmpeg
    except ImportError:
        print("      ⚠️ imageio-ffmpeg is not installed yet. Skipping FFmpeg setup.")
        return

    project_root = Path(__file__).resolve().parent
    local_bin = project_root / ".bin"
    local_bin.mkdir(exist_ok=True)

    is_windows = sys.platform.startswith("win")
    exe_name = "ffmpeg.exe" if is_windows else "ffmpeg"
    target_path = local_bin / exe_name

    if target_path.exists():
        print(f"      ✅ FFmpeg is already configured at: {target_path}")
        return

    print("      Detecting FFmpeg via imageio-ffmpeg...")
    source_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    if not source_exe or not Path(source_exe).exists():
        print("      ❌ Failed to obtain FFmpeg executable from imageio-ffmpeg.")
        return

    try:
        shutil.copy(source_exe, target_path)
        if not is_windows:
            target_path.chmod(target_path.stat().st_mode | 0o111)
            
        print(f"      ✅ FFmpeg successfully installed to: {target_path}")
    except Exception as e:
        print(f"      ❌ Error copying FFmpeg: {e}")


def predownload_models():
    """Pre-download Whisper model weights so that the first transcription job
    is fully offline. Both openai-whisper and faster-whisper caches are populated.
    
    Cache locations:
      - openai-whisper : ~/.cache/whisper/<model>.pt
      - faster-whisper : ~/.cache/huggingface/hub/models--Systran--faster-whisper-<model>/
    """
    # Read the default model from .env if present, otherwise use 'base'.
    model_name = os.getenv("WHISPER_MODEL", "base")
    print(f"\n[4/4] Pre-downloading Whisper model weights (model='{model_name}')...")
    print("      This downloads model weights so transcription can run fully offline.")
    print("      Weights are cached and will NOT be re-downloaded on subsequent runs.")

    # --- openai-whisper ---
    try:
        import whisper  # type: ignore
        cache_dir = Path.home() / ".cache" / "whisper"
        cached_file = cache_dir / f"{model_name}.pt"
        if cached_file.exists():
            print(f"      ✅ openai-whisper: '{model_name}' already cached at {cached_file}")
        else:
            print(f"      ⬇  openai-whisper: downloading '{model_name}'…")
            whisper.load_model(model_name)
            print(f"      ✅ openai-whisper: '{model_name}' downloaded successfully.")
    except ImportError:
        print("      ℹ️  openai-whisper not installed — skipping its cache population.")
    except Exception as e:
        print(f"      ⚠️  openai-whisper download failed (non-fatal): {e}")

    # --- faster-whisper ---
    try:
        from faster_whisper import WhisperModel  # type: ignore
        # Check if already cached by looking for the huggingface hub directory
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        # faster-whisper stores as: models--Systran--faster-whisper-<name>
        cached_dir = hf_cache / f"models--Systran--faster-whisper-{model_name}"
        if cached_dir.exists():
            print(f"      ✅ faster-whisper: '{model_name}' already cached at {cached_dir}")
        else:
            print(f"      ⬇  faster-whisper: downloading '{model_name}' (CPU/int8)…")
            # Download using CPU so no GPU is required at install time
            WhisperModel(model_name, device="cpu", compute_type="int8")
            print(f"      ✅ faster-whisper: '{model_name}' downloaded successfully.")
    except ImportError:
        print("      ℹ️  faster-whisper not installed — skipping its cache population.")
    except Exception as e:
        print(f"      ⚠️  faster-whisper download failed (non-fatal): {e}")


def install():
    print("=" * 40)
    print("    Transcriber - Installation Setup")
    print("=" * 40)
    
    # 1. Install PyTorch with CUDA
    print("\n[1/4] Installing PyTorch with GPU (CUDA) support...")
    print("      This might take a few minutes depending on your internet connection.")
    
    urls = [
        "https://download.pytorch.org/whl/cu124",
        "https://download.pytorch.org/whl/cu121",
        "https://download.pytorch.org/whl/cu118"
    ]
    
    torch_installed = False
    for url in urls:
        print(f"      Trying repository: {url}")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", url,
            "--no-cache-dir"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("      ✅ PyTorch (CUDA) successfully installed!")
            torch_installed = True
            break
        else:
            print("      ⚠️ Failed on this version, trying next...")
            
    if not torch_installed:
        print("\n❌ Could not install the CUDA version of PyTorch automatically.")
        print("   This usually happens if you are using a very recent Python version (e.g., 3.13 or 3.14).")
        print("   We strongly recommend using Python 3.10, 3.11, or 3.12.")
        print("   Installing the default (CPU) version as a fallback...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

    # 2. Install project dependencies
    print("\n[2/4] Installing project dependencies...")
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if os.path.exists(req_path):
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path])
        if result.returncode == 0:
            print("      ✅ Dependencies successfully installed!")
        else:
            print("\n❌ An error occurred while installing dependencies from requirements.txt.")
            return
    else:
        print(f"\n❌ File {req_path} not found!")
        return

    # 3. Setup FFmpeg
    setup_ffmpeg()

    # 4. Pre-download model weights (so first transcription is fully offline)
    predownload_models()
    
    print("\n✅ Setup Complete!")
    print("\nTo start the application, run:")
    print("python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    install()