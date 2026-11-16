"""Command-line entry points: ``parttwin {doctor,inspect,render,gui}``."""
from __future__ import annotations

import sys

COMMANDS = {"doctor": "doctor", "inspect": "inspect", "render": "render", "gui": "gui",
            "maniskill-gui": "maniskill_gui"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: parttwin {doctor,inspect,render,gui,maniskill-gui} [options]\n"
              "  doctor   environment + source provenance (--provenance)\n"
              "  inspect  field-by-field config/URDF/PLY diagnostics\n"
              "  render   offline kinematic joint sweep -> RGB/depth/alpha/metadata\n"
              "  gui      live interactive 3D Gaussian GUI in the browser (viser + nerfview)\n"
              "  maniskill-gui  Gaussians composited into ManiSkill's SAPIEN viewer (desktop window)")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    import importlib

    mod = importlib.import_module(f"parttwin1d.cli.{COMMANDS[argv[0]]}")
    return int(mod.main(argv[1:]) or 0)
