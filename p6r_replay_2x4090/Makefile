PY ?= python3
PROTO ?= protocols/p6r_replay_frozen.json
OUT ?= out
WORKERS ?= 64
P6RCODE ?=

.PHONY: help check phase0 fingerprint replay repeats analyze smoke clean

help:
	@echo "make check        registry invariants + protocol sanity (no GPU, no server)"
	@echo "make smoke        end-to-end run against a local mock server (no GPU)"
	@echo "make phase0       ARCHIVE=<path>  zero-GPU forensics on the original run"
	@echo "make fingerprint  serving fingerprint + greedy canary"
	@echo "make replay       full replay, repeat 0"
	@echo "make repeats      anti-safe repeats 1 and 2"
	@echo "make analyze      faithfulness gates + pre-registered verdict"

check:
	@$(PY) src/registry.py

phase0:
	@test -n "$(ARCHIVE)" || (echo "set ARCHIVE=<path to original P6R results>"; exit 2)
	@$(PY) src/forensics_reparse.py --archive "$(ARCHIVE)" --out $(OUT)/PHASE0_FORENSICS.json

fingerprint:
	@$(PY) src/fingerprint.py --protocol $(PROTO) --out $(OUT)/FINGERPRINT_$$(date +%Y%m%dT%H%M%S).json \
	  $(if $(P6RCODE),--p6r-code-dir $(P6RCODE),--allow-reconstructed-prompt)

replay:
	@$(PY) src/replay_runner.py --protocol $(PROTO) --out $(OUT)/replay_r0 \
	  --workers $(WORKERS) --repeat 0 \
	  $(if $(P6RCODE),--p6r-code-dir $(P6RCODE),--allow-reconstructed-prompt)

repeats:
	@$(PY) src/replay_runner.py --protocol $(PROTO) --out $(OUT)/replay_r1 \
	  --workers $(WORKERS) --repeat 1 --families anti_safe \
	  $(if $(P6RCODE),--p6r-code-dir $(P6RCODE),--allow-reconstructed-prompt)
	@$(PY) src/replay_runner.py --protocol $(PROTO) --out $(OUT)/replay_r2 \
	  --workers $(WORKERS) --repeat 2 --families anti_safe \
	  $(if $(P6RCODE),--p6r-code-dir $(P6RCODE),--allow-reconstructed-prompt)

analyze:
	@$(PY) src/analyze_replay.py --runs $(OUT)/replay_r0 $(OUT)/replay_r1 $(OUT)/replay_r2 \
	  --fingerprint "$(OUT)/FINGERPRINT_*.json" --original-fidelity 0.041 \
	  --out $(OUT)/REPLAY_VERDICT.json

smoke:
	@$(PY) tests/smoke_mock_server.py

clean:
	@rm -rf $(OUT) __pycache__ src/__pycache__
