# All targets run inside the existing `parttwin_sim` Conda env (activate it first).
SHELL := /bin/bash
CONFIG ?= configs/examples/hinged_box.yaml
PORT ?= 8080
ACT := source scripts/activate_sim.sh >/dev/null &&

.PHONY: setup doctor test test-gpu gui render inspect lint evidence clean-outputs

setup:            ## submodules + editable --no-deps installs + gsplat JIT check
	bash scripts/bootstrap_submodules.sh

doctor:           ## provenance / compatibility report
	bash scripts/check_environment.sh

test:             ## unit tests + collection of all tests (GPU/GUI skipped unless opted in)
	bash scripts/test.sh

test-gpu:         ## REAL PhysX + gsplat CUDA + browser GUI tests
	bash scripts/test_gpu.sh

gui:              ## live interactive GUI on http://127.0.0.1:$(PORT)
	bash scripts/run_gui.sh $(CONFIG) --port $(PORT)

render:           ## offline kinematic sweep to outputs/hinged_box
	$(ACT) python -m parttwin1d.cli.render --config $(CONFIG) --joint door_hinge \
	  --q-start 0 --q-end 1.3 --frames 20 --out outputs/hinged_box --segmentation

inspect:          ## field-by-field config diagnostics
	$(ACT) python -m parttwin1d.cli.inspect --config $(CONFIG)

evidence:         ## GUI screenshots in a real browser -> outputs/evidence
	$(ACT) python scripts/capture_gui_evidence.py --config $(CONFIG)

lint:             ## byte-compile + pyflakes if available
	$(ACT) python -m compileall -q src tests examples && (python -m pyflakes src tests 2>/dev/null || true)

clean-outputs:
	rm -rf outputs
