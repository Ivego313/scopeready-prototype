"""ScopeReady core prototype: requirements gap detection over a document corpus."""

from importlib.metadata import metadata

_dist = metadata(__name__)

__title__ = _dist["Name"]
__version__ = _dist["Version"]
__description__ = _dist["Summary"]

__all__ = ["__description__", "__title__", "__version__"]
