"""Launch every microservice (each as its own uvicorn process) plus the
gateway, so the whole SOA backend comes up with one command:

    python -m services.run_all

Stop with Ctrl-C. For production each service would run in its own container;
this is the local-dev equivalent.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time

SERVICES = [
    ("services.preprocessing_service:app", 8001),
    ("services.indexing_service:app", 8002),
    ("services.retrieval_service:app", 8003),
    ("services.ranking_eval_service:app", 8004),
    ("services.query_refinement_service:app", 8005),
    ("services.gateway:app", 8000),
]


def main():
    procs = []
    print("Starting IR microservices ...")
    for target, port in SERVICES:
        cmd = [sys.executable, "-m", "uvicorn", target,
               "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
        procs.append(subprocess.Popen(cmd))
        print(f"  -> {target} on http://127.0.0.1:{port}")
        time.sleep(1.0)
    print("\nAll services up. Gateway: http://127.0.0.1:8000  (Ctrl-C to stop)")

    def shutdown(*_):
        print("\nShutting down ...")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
