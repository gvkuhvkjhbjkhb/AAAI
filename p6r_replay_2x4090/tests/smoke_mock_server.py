"""
smoke_mock_server.py -- end-to-end check of the replay pipeline with no GPU.

Spins up two tiny OpenAI-compatible mock endpoints on localhost, runs the real
`replay_runner` against a 1-matrix / 2-seed cut of the protocol, then runs the
real `analyze_replay` over the output.  The mock's compliance probability is a
knob, so this also checks that the pre-registered verdict logic fires correctly
in both directions:

    COMPLY=0.99  ->  verdict A_environment_artifact
    COMPLY=0.05  ->  verdict B_reproduces

Run:  python tests/smoke_mock_server.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

COMPLY = float(os.environ.get("COMPLY", "0.99"))


def make_handler(model_name: str, role: int):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # silence
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": model_name}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode())
            user = req["messages"][-1]["content"]
            # the advisory cue names the assigned letter; comply with prob COMPLY
            import random
            seed = req.get("seed", 0)
            rnd = random.Random(seed * 7919 + role)
            letter = "A"
            if "recommends Action " in user:
                letter = user.split("recommends Action ")[1][0]
                if rnd.random() >= COMPLY:
                    letter = "B" if letter == "A" else "A"
            else:
                letter = "A" if rnd.random() < 0.5 else "B"
            text = f"ACTION: {letter}"
            body = json.dumps({"choices": [{"message":
                              {"role": "assistant", "content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def serve(port: int, model: str, role: int):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model, role))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    out = ROOT / "out_smoke"
    if out.exists():
        shutil.rmtree(out)

    proto = json.loads((ROOT / "protocols" /
                        "p6r_replay_frozen.json").read_text(encoding="utf-8"))
    proto["model_pair"]["roles"] = {"0": "mock-a", "1": "mock-b"}
    proto["model_pair"]["endpoints"] = {"0": "http://127.0.0.1:18002/v1",
                                        "1": "http://127.0.0.1:18003/v1"}
    proto["model_pair"]["revisions"] = {"mock-a": "deadbeef", "mock-b": "deadbeef"}
    proto["model_pair"]["revision_lock_required"] = False
    proto["p6r"]["seeds"] = [700, 701]
    # warm-up must clear the (C0) coverage guard (min_total_observations=40),
    # i.e. >= 8 episodes of 5 steps; otherwise ActionSafe abstains and the
    # faithfulness gate correctly rejects the run.
    proto["sampling"]["warmup_episodes"] = 10
    proto["sampling"]["total_episodes"] = 14
    p = out / "protocol_smoke.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(proto), encoding="utf-8")

    serve(18002, "mock-a", 0)
    serve(18003, "mock-b", 1)
    time.sleep(0.4)

    env = dict(os.environ, VLLM_API_KEY="smoke")
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "replay_runner.py"),
         "--protocol", str(p), "--out", str(out / "replay_r0"),
         "--workers", "8", "--repeat", "0",
         "--families", "anti_safe", "anti_tradeoff",
         "--allow-reconstructed-prompt"],
        env=env, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        return 1

    r2 = subprocess.run(
        [sys.executable, str(ROOT / "src" / "analyze_replay.py"),
         "--runs", str(out / "replay_r0"),
         "--original-fidelity", "0.041",
         "--out", str(out / "REPLAY_VERDICT.json")],
        env=env, capture_output=True, text=True)
    print(r2.stdout[-4000:])
    if r2.stderr.strip():
        print(r2.stderr[-2000:], file=sys.stderr)

    v = json.loads((out / "REPLAY_VERDICT.json").read_text(encoding="utf-8"))
    expect = "A_environment_artifact" if COMPLY > 0.5 else "B_reproduces"
    ok = v["verdict"] == expect
    print(f"\nSMOKE: COMPLY={COMPLY} -> verdict={v['verdict']} "
          f"(expected {expect}) : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
