"""Verified dependency pins (see requirements/environment.lock.md, docs/third_party.md)."""

# submodule dir -> (python module, distribution version, pinned commit SHA)
SUBMODULES = {
    "ManiSkill": ("mani_skill", "3.0.1", "a4a4f9272ad64b1564035874b605ceb687b63ed8"),
    "gsplat": ("gsplat", "1.5.3", "937e29912570c372bed6747a5c9bf85fed877bae"),
    "nerfview": ("nerfview", "0.1.2", "4538024fe0d15fd1a0e4d760f3695fc44ca72787"),
}

# binary / pip runtime dependencies (not submodules)
RUNTIME = {
    "torch": "2.11.0+cu128",
    "sapien": "3.0.3",
    "viser": "1.1.1",
    "plyfile": "1.1.5",
    "scipy": "1.17.1",
    "numpy": "2.4.6",
}

CUDA_TOOLKIT = "12.8"
