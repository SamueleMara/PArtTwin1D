"""Object configuration: schema, loader, validation."""
from .loader import load_config
from .schema import CameraSpec, LinkBinding, ObjectConfig, RenderSettings, SimSettings, ViewerSettings
from .validate import ConfigError

__all__ = [
    "load_config", "ConfigError", "ObjectConfig", "LinkBinding", "CameraSpec",
    "RenderSettings", "ViewerSettings", "SimSettings",
]
