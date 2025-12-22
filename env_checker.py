import os
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime


def _run_cmd(cmd):
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, shell=True, text=True
        )
        return out.strip()
    except Exception as e:
        return None


def check_python():
    info = {}
    info["python_version"] = sys.version.replace("\n", " ")
    info["executable"] = sys.executable
    return info


def check_os():
    info = {}
    info["platform"] = platform.platform()
    info["system"] = platform.system()
    info["release"] = platform.release()
    info["machine"] = platform.machine()
    return info


def check_cuda_torch():
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            gpus = []
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
            info["gpu_names"] = gpus
        else:
            info["gpu_count"] = 0
            info["gpu_names"] = []
    except Exception as e:
        info["error"] = str(e)
    return info


def check_ffmpeg():
    info = {}
    ver = _run_cmd("ffmpeg -version")
    if ver is None:
        info["available"] = False
    else:
        info["available"] = True
        info["version"] = ver.splitlines()[0]
    return info


def check_packages():
    pkgs = [
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "timm",
        "opencv-python-headless",
        "cv2",
        "yaml",
        "pandas",
        "matplotlib",
        "sklearn",
        "einops",
        "tqdm",
    ]

    results = {}
    for p in pkgs:
        try:
            if p == "cv2":
                import cv2
                results["cv2"] = cv2.__version__
            elif p == "yaml":
                import yaml
                results["pyyaml"] = yaml.__version__
            elif p == "sklearn":
                import sklearn
                results["scikit-learn"] = sklearn.__version__
            else:
                mod = __import__(p.replace("-", "_"))
                results[p] = getattr(mod, "__version__", "unknown")
        except Exception:
            results[p] = None
    return results


def format_section(title, kv):
    lines = []
    lines.append(f"== {title} ==")
    for k, v in kv.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    return "\n".join(lines)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "env_report.txt"

    lines = []
    lines.append("Environment Check Report")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 50)
    lines.append("")

    # Python
    py_info = check_python()
    print("[INFO] Python version:", py_info["python_version"])
    lines.append(format_section("Python", py_info))

    # OS
    os_info = check_os()
    print("[INFO] OS:", os_info["platform"])
    lines.append(format_section("Operating System", os_info))

    # Torch / CUDA
    torch_info = check_cuda_torch()
    if "error" in torch_info:
        print("[ERROR] Torch check failed:", torch_info["error"])
    else:
        print("[INFO] Torch:", torch_info.get("torch_version", "unknown"))
        print("[INFO] CUDA available:", torch_info.get("cuda_available", False))
        if torch_info.get("cuda_available", False):
            for g in torch_info.get("gpu_names", []):
                print("[INFO] GPU:", g)
    lines.append(format_section("PyTorch & CUDA", torch_info))

    # FFmpeg
    ffmpeg_info = check_ffmpeg()
    if ffmpeg_info["available"]:
        print("[INFO] FFmpeg:", ffmpeg_info["version"])
    else:
        print("[WARN] FFmpeg not found")
    lines.append(format_section("FFmpeg", ffmpeg_info))

    # Packages
    pkg_info = check_packages()
    print("[INFO] Checking Python packages...")
    for k, v in pkg_info.items():
        if v is None:
            print(f"[WARN] Package missing: {k}")
        else:
            print(f"[INFO] {k}: {v}")
    lines.append(format_section("Python Packages", pkg_info))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Environment report saved to {report_path}")


if __name__ == "__main__":
    main()
