import importlib.metadata
from . import calcs, helpers, io, proc

__all__ = ["io", "proc", "calcs", "helpers"]
# version is defined in pyproject.toml
__version__ = importlib.metadata.version("ctdproc")
