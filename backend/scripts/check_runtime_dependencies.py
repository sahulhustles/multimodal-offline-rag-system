import sys
import json
from backend.utils.dependency_checker import get_all_system_dependencies, get_ollama_status

def run():
    print("========================================")
    print(" Multimodal RAG Dependency Checker")
    print("========================================")
    
    deps = get_all_system_dependencies()
    ollama = get_ollama_status()
    
    from backend.utils.dependency_checker import get_whisper_status
    print("\nRunning deep check for faster-whisper...")
    whisper = get_whisper_status(deep_check=True)
    
    print(f"\nRuntime Environment: {deps['runtime_environment']}")
    print(f"Platform: {deps['platform']}")
    
    print("\n[ ffmpeg ] - Required for Audio Processing")
    ff = deps["ffmpeg"]
    print(f"  Available: {ff['available']}")
    if ff["available"]:
        print(f"  Resolved Path: {ff['resolved_path']} (Source: {ff['detection_source']})")
        print(f"  Version: {ff['version']}")
    else:
        print(f"  Error: {ff['error_message']}")

    print("\n[ LibreOffice ] - Required for Legacy .doc Conversion")
    lo = deps["libreoffice"]
    print(f"  Available: {lo['available']}")
    if lo["available"]:
        print(f"  Resolved Path: {lo['resolved_path']} (Source: {lo['detection_source']})")
        print(f"  Version: {lo['version']}")
    else:
        print(f"  Error: {lo['error_message']}")
        
    print("\n[ Ollama ] - Required for Vision Description (LLaVA)")
    print(f"  Connected: {ollama['connected']}")
    if ollama["connected"]:
        print(f"  LLaVA Model Available: {ollama['llava_available']}")
    else:
        print(f"  Error: {ollama['error_message']}")
        
    print("\n[ faster-whisper ] - Required for Audio Transcription")
    print(f"  Model: {whisper['model_name']} (Compute: {whisper['compute_type']}, Device: {whisper['device']})")
    print(f"  Available: {whisper['available']}")
    print(f"  Load Test: {whisper['load_test_status']}")
    if not whisper["available"]:
        print(f"  Error: {whisper['error_message']}")
        
    print("\n========================================")

if __name__ == "__main__":
    run()
