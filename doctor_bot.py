import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REQUIRED_MODULES = [
    ("discord.py", "discord.py"),
    ("discord-ext-voice-recv", "discord-ext-voice-recv"),
    ("aiohttp", "aiohttp"),
    ("numpy", "numpy"),
    ("faster-whisper", "faster-whisper"),
    ("python-dotenv", "python-dotenv"),
    ("ollama", "ollama"),
]

OPTIONAL_MODULES = [
    ("sentence-transformers", "sentence-transformers"),
    ("faiss-cpu", "faiss-cpu"),
]


def check_package(package_name: str, label: str) -> bool:
    try:
        version = metadata.version(package_name)
        print(f"[ok] {label}: {version}")
        return True
    except Exception as exc:
        print(f"[missing] {label}: {exc}")
        return False


def check_ffmpeg() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[missing] ffmpeg: not found on PATH")
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        print(f"[ok] ffmpeg: {first_line}")
        return True
    except Exception as exc:
        print(f"[missing] ffmpeg: {exc}")
        return False


def check_env_file(repo_root: Path) -> bool:
    env_file = repo_root / ".env"
    if env_file.exists():
        print(f"[ok] .env: {env_file}")
        return True
    print(f"[warn] .env: missing, copy {repo_root / '.env.example'} to .env")
    return False


def check_lm_studio() -> bool:
    try:
        import json
        import urllib.request

        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
            payload = json.load(response)
        count = len(payload.get("data") or [])
        print(f"[ok] LM Studio: reachable, {count} loaded model(s)")
        return True
    except Exception as exc:
        print(f"[warn] LM Studio: not reachable or no models loaded ({exc})")
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    print(f"Python: {sys.executable}")
    print(f"Repo:   {repo_root}")
    print("")

    failures = 0
    for package_name, label in REQUIRED_MODULES:
        if not check_package(package_name, label):
            failures += 1

    print("")
    for package_name, label in OPTIONAL_MODULES:
        check_package(package_name, label)

    print("")
    if not check_ffmpeg():
        failures += 1

    check_env_file(repo_root)
    check_lm_studio()

    print("")
    main_file = repo_root / "Main.py"
    if main_file.exists():
        print(f"[ok] bot entrypoint: {main_file}")
    else:
        print(f"[missing] bot entrypoint: {main_file}")
        failures += 1

    if failures:
        print("")
        print(f"Doctor failed with {failures} required issue(s).")
        return 1

    print("")
    print("Doctor passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
