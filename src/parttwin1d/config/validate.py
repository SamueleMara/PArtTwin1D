"""Field-level validation helpers shared by the loader and the inspector."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.cameras import validate_K
from ..geometry.se3 import validate_se3


class ConfigError(ValueError):
    """Invalid configuration; message names the offending field."""


@dataclass
class Diagnostic:
    level: str  # "ok" | "warn" | "error"
    field: str
    message: str


@dataclass
class Report:
    items: list[Diagnostic] = field(default_factory=list)

    def ok(self, f, m):
        self.items.append(Diagnostic("ok", f, m))

    def warn(self, f, m):
        self.items.append(Diagnostic("warn", f, m))

    def error(self, f, m):
        self.items.append(Diagnostic("error", f, m))

    @property
    def errors(self):
        return [d for d in self.items if d.level == "error"]

    def format(self) -> str:
        tag = {"ok": "  OK ", "warn": "WARN ", "error": "ERROR"}
        return "\n".join(f"[{tag[d.level]}] {d.field}: {d.message}" for d in self.items)


def require_keys(d: dict, allowed: set, where: str) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def se3_field(value, where: str) -> np.ndarray:
    if value is None:
        raise ConfigError(f"{where}: required 4x4 transform is missing (identity is never assumed)")
    try:
        return validate_se3(value, where)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def K_field(value, where: str, width: int, height: int) -> np.ndarray:
    try:
        return validate_K(value, where, width, height)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def resolve_path(cfg_dir: Path, value, where: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a non-empty path string")
    p = Path(value).expanduser()
    p = p if p.is_absolute() else (cfg_dir / p)
    p = p.resolve()
    if must_exist and not p.is_file():
        raise ConfigError(f"{where}: file not found: {p}")
    return p


def rgb_field(value, where: str) -> tuple[float, float, float]:
    try:
        bg = tuple(float(v) for v in value)
    except TypeError as exc:
        raise ConfigError(f"{where}: expected three numbers") from exc
    if len(bg) != 3 or any(not 0.0 <= v <= 1.0 for v in bg):
        raise ConfigError(f"{where}: expected three values in [0, 1], got {value}")
    return bg  # type: ignore[return-value]
