"""Start the local Research Studio with: python run.py."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def run_command(command, directory):
    subprocess.run(command, cwd=directory, check=True)


def ensure_python_dependencies():
    modules = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
               "gymnasium": "gymnasium", "torch": "torch", "stable-baselines3": "stable_baselines3",
               "PyYAML": "yaml", "fastapi": "fastapi", "uvicorn": "uvicorn"}
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    missing = [line for line in requirements for name, module in modules.items()
               if line.startswith(name + ">=") and importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing Python dependencies…", flush=True)
        run_command([sys.executable, "-m", "pip", "install", *missing], ROOT)


def frontend_needs_build():
    index = FRONTEND / "dist" / "index.html"
    if not index.is_file():
        return True
    inputs = [*filter(Path.is_file, (FRONTEND / "src").rglob("*"))]
    inputs += [FRONTEND / name for name in ("package.json", "package-lock.json", "index.html",
               "vite.config.ts", "tsconfig.json", "tailwind.config.js", "postcss.config.js")]
    return any(path.stat().st_mtime_ns > index.stat().st_mtime_ns for path in inputs)


def prepare_frontend():
    if not frontend_needs_build():
        return
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("Install Node.js LTS from https://nodejs.org, reopen PowerShell, then run python run.py again.")
    dependency_lock = FRONTEND / "node_modules" / ".package-lock.json"
    if not dependency_lock.exists() or dependency_lock.stat().st_mtime_ns < (FRONTEND / "package-lock.json").stat().st_mtime_ns:
        print("Installing frontend dependencies from the lockfile…", flush=True)
        run_command([npm, "ci"], FRONTEND)
    print("Building Research Studio…", flush=True)
    run_command([npm, "run", "build"], FRONTEND)


def create_app():
    from fastapi.staticfiles import StaticFiles
    from src.api import app
    # API routes retain priority. The built React app uses /api on this same origin.
    if not any(getattr(route, "name", None) == "studio" for route in app.routes):
        app.mount("/", StaticFiles(directory=FRONTEND / "dist", html=True), name="studio")
    return app


def open_when_ready(url):
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open_new_tab(url)
                    return
        except (OSError, TimeoutError):
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Prepare dependencies and UI without starting a server")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    os.chdir(ROOT)
    if not args.check:
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", args.port))
        except OSError as exc:
            raise RuntimeError(f"Port {args.port} is busy. Close the existing server or use python run.py --port {args.port + 1}.") from exc
    ensure_python_dependencies()
    prepare_frontend()
    app = create_app()
    if args.check:
        print("Research Studio is ready. Run: python run.py")
        return
    import uvicorn
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nOncoEvo Research Studio: {url}\nPress Ctrl+C to stop. Training is never started by this launcher.\n", flush=True)
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        from src.api import jobs, pool
        for job in jobs.values():
            job["event"].set()
        pool.shutdown(wait=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nCould not start Research Studio: {exc}", file=sys.stderr)
        sys.exit(1)
