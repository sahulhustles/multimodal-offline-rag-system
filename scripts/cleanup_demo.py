#!/usr/bin/env python3
"""Cleanup script for Phase 1 & 2 UI Demo temporary files.

This script safely removes uploaded files and processed artifacts 
created by the frontend demonstration UI, which are stored in `data/demo`.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_demo_files():
    demo_dir = Path("data/demo")
    
    if not demo_dir.exists():
        print(f"Directory {demo_dir} does not exist. Nothing to clean.")
        return

    print(f"[{datetime.now().isoformat()}] Starting cleanup of {demo_dir}...")
    
    # Track metrics
    bytes_freed = 0
    files_removed = 0
    
    for root, dirs, files in os.walk(demo_dir):
        for f in files:
            # We don't delete .gitkeep if it exists
            if f == ".gitkeep":
                continue
            
            file_path = Path(root) / f
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                bytes_freed += size
                files_removed += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
                
    # Also clean up empty directories
    for root, dirs, files in os.walk(demo_dir, topdown=False):
        for d in dirs:
            dir_path = Path(root) / d
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except Exception as e:
                pass
                
    print(f"Cleanup complete! Removed {files_removed} files, freeing {bytes_freed / (1024*1024):.2f} MB.")

if __name__ == "__main__":
    cleanup_demo_files()
