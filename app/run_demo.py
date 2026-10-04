"""Start the API and the Streamlit demo together; stop both on Ctrl+C.

Plain Python so it also works where `make` is not installed (e.g. Windows). The API is
started first and the demo only once /health answers, so the first click never hits a
model that is still loading. HF_HUB_OFFLINE keeps everything local (no network).

    python app/run_demo.py            # API on :8000, demo on :8501
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API_PORT = os.environ.get("INTENT_API_PORT", "8000")
UI_PORT = os.environ.get("INTENT_UI_PORT", "8501")
API_URL = f"http://127.0.0.1:{API_PORT}"


def wait_for_api(proc: subprocess.Popen, timeout_s: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("the API process exited during startup; see its log above")
        try:
            if httpx.get(f"{API_URL}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"API not ready after {timeout_s:.0f} s")


def main() -> None:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "intent.serve:app", "--host", "127.0.0.1",
                            "--port", API_PORT], cwd=ROOT, env=env)
    ui = None
    try:
        print(f"Loading the model (API on {API_URL}) ...", flush=True)
        wait_for_api(api)
        ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "demo.py"),
                               "--server.port", UI_PORT, "--server.headless", "true",
                               "--server.address", "127.0.0.1"],
                              cwd=ROOT, env={**env, "INTENT_API_URL": API_URL})
        print(f"Demo ready: http://localhost:{UI_PORT}  (Ctrl+C to stop)", flush=True)
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (ui, api):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)


if __name__ == "__main__":
    main()
