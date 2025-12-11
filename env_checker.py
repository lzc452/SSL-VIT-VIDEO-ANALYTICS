# src/utils/env_check.py
import os
import sys
import platform
import subprocess
import importlib
from datetime import datetime
from pathlib import Path

REPORT_PATH = Path("logs/env_report.txt")
REQUIREMENT_KEYS = [
    "torch", "torchvision", "timm", "opencv-python", "ffmpeg-python",
    "pandas", "tqdm", "pyyaml", "matplotlib", "seaborn",
    "flwr", "grpcio", "tensorboard", "plotly", "insightface",
    "mediapipe", "scikit-learn", "scikit-image"
]

def check_python():
    return f"{platform.python_version()} ({platform.system()} {platform.release()})"

def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        cudnn = torch.backends.cudnn.version()
        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_name": device_name,
            "cudnn_version": cudnn
        }
    except Exception as e:
        return {"error": str(e)}

def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            return f"Available {first_line}"
        else:
            return f"[Warning] ffmpeg not detected (return code {result.returncode})"
    except FileNotFoundError:
        return "[Error] ffmpeg not found in PATH"

def check_package(pkg):
    try:
        module = importlib.import_module(pkg.split("==")[0])
        version = getattr(module, "__version__", "unknown")
        return f"{pkg}: {version}"
    except Exception:
        return f"{pkg}: not installed"

def main():
    print("\n[Info] Checking environment...\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"Environment Check Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Python & System
    lines.append(f"Python Version : {check_python()}")
    lines.append(f"Executable Path: {sys.executable}")
    lines.append("")

    # CUDA / Torch
    lines.append("CUDA / Torch Information")
    cuda_info = check_cuda()
    for k, v in cuda_info.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # ffmpeg
    lines.append("FFmpeg")
    lines.append(f"  {check_ffmpeg()}")
    lines.append("")

    # Packages
    lines.append("Package Versions")
    for pkg in REQUIREMENT_KEYS:
        lines.append(f"  {check_package(pkg)}")
    lines.append("")

    # GPU memory info (optional)
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            lines.append(f"GPU Memory Total: {total:.2f} GB")
    except Exception:
        lines.append("GPU Memory: not available")

    lines.append("\n Environment check completed successfully.\n")

    # Save report
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Print summary
    print("\n".join(lines[:20]))
    print(f"...\nReport saved to: {REPORT_PATH.resolve()}\n")

if __name__ == "__main__":
    main()
