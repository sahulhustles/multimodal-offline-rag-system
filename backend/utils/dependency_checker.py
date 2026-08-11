import os
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any

from backend.config import settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_runtime_environment() -> str:
    """Determine if running inside Docker or local Python."""
    if Path("/.dockerenv").exists():
        return "docker_container"
    return "local_python"

def check_dependency(name: str, configured_path: str | None, common_windows_paths: list[str], version_cmd: list[str] | None = None) -> Dict[str, Any]:
    """Generic checker for an external dependency."""
    result = {
        "available": False,
        "executable_name": name,
        "resolved_path": None,
        "detection_source": "not_found",
        "version": None,
        "error_message": f"{name} is not installed or not found on PATH."
    }

    # 1. Check configured explicit path
    if configured_path:
        cp = Path(configured_path)
        if cp.exists() and cp.is_file():
            result["resolved_path"] = str(cp)
            result["detection_source"] = "configured_path"
            
    # 2. Check PATH
    if not result["resolved_path"]:
        which_path = shutil.which(name)
        if which_path:
            result["resolved_path"] = which_path
            result["detection_source"] = "PATH"
            
    # 3. Check common locations (Windows)
    if not result["resolved_path"] and platform.system() == "Windows":
        import glob
        for pattern in common_windows_paths:
            matches = glob.glob(pattern)
            if matches:
                result["resolved_path"] = matches[0]
                result["detection_source"] = "common_location_scan"
                break

    if result["resolved_path"]:
        result["available"] = True
        result["error_message"] = None
        # Try to get version
        if version_cmd:
            try:
                cmd = [result["resolved_path"]] + version_cmd
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    result["version"] = res.stdout.split('\n')[0].strip()
            except Exception as e:
                logger.warning(f"Found {name} at {result['resolved_path']} but failed to get version: {e}")
        else:
            result["version"] = "Detected (Active)"

    return result

def get_ffmpeg_status() -> Dict[str, Any]:
    return check_dependency(
        name="ffmpeg",
        configured_path=settings.ffmpeg_path,
        common_windows_paths=[
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            r"C:\Users\*\AppData\Local\CapCut\Apps\*\ffmpeg.exe",
        ],
        version_cmd=["-version"]
    )

def get_libreoffice_status() -> Dict[str, Any]:
    # On Windows the executable is usually soffice.exe
    exec_name = "soffice" if platform.system() != "Windows" else "soffice.exe"
    
    return check_dependency(
        name=exec_name,
        configured_path=settings.libreoffice_path,
        common_windows_paths=[
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
        ],
        version_cmd=None  # Skip executing the soffice subprocess to avoid console popups on Windows
    )

def get_ollama_status() -> Dict[str, Any]:
    import httpx
    result = {
        "connected": False,
        "llava_available": False,
        "error_message": None
    }
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                result["connected"] = True
                model_list = resp.json().get("models", [])
                result["llava_available"] = any(
                    settings.ollama_model in m.get("name", "")
                    for m in model_list
                )
            else:
                result["error_message"] = f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        result["error_message"] = f"Failed to connect: {exc}"
    return result

_whisper_cached_status = None

def get_whisper_status(deep_check: bool = False) -> Dict[str, Any]:
    global _whisper_cached_status
    
    # 1. Check in-memory state of active processors
    model_loaded = False
    try:
        from backend.processors import audio_processor
        if audio_processor._model is not None:
            model_loaded = True
    except Exception:
        pass

    if model_loaded:
        _whisper_cached_status = {
            "model_name": settings.whisper_model_size,
            "compute_type": settings.whisper_compute_type,
            "device": settings.whisper_device,
            "dependency_present": True,
            "operational": True,
            "load_test_status": "passed",
            "available": True,
            "error_message": None
        }

    if _whisper_cached_status and not deep_check:
        return _whisper_cached_status

    # 2. Lightweight import check
    dependency_present = False
    error_msg = None
    try:
        import faster_whisper
        dependency_present = True
    except ImportError as e:
        error_msg = f"faster-whisper is not installed: {e}"

    # Check ffmpeg availability to establish operational state
    ffmpeg_ok = False
    try:
        ffmpeg_ok = get_ffmpeg_status()["available"]
    except Exception:
        pass

    status = {
        "model_name": settings.whisper_model_size,
        "compute_type": settings.whisper_compute_type,
        "device": settings.whisper_device,
        "dependency_present": dependency_present,
        "operational": dependency_present and ffmpeg_ok,
        "load_test_status": "passed" if model_loaded else ("not_tested" if dependency_present else "failed"),
        "available": dependency_present,  # Backwards compatibility
        "error_message": error_msg
    }

    if deep_check and dependency_present:
        try:
            from faster_whisper import WhisperModel
            logger.info("Deep check: Initializing WhisperModel...")
            WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            status["load_test_status"] = "passed"
            status["operational"] = True
            status["available"] = True
            status["error_message"] = None
        except Exception as e:
            status["load_test_status"] = "failed"
            status["operational"] = False
            status["available"] = False
            status["error_message"] = f"faster-whisper {settings.whisper_model_size} load failed: {e}"
        
        _whisper_cached_status = status
    elif not dependency_present:
        _whisper_cached_status = status
            
    return status

def get_all_system_dependencies() -> Dict[str, Any]:
    return {
        "runtime_environment": get_runtime_environment(),
        "platform": platform.platform(),
        "ffmpeg": get_ffmpeg_status(),
        "libreoffice": get_libreoffice_status(),
        "whisper": get_whisper_status(deep_check=False)
    }
