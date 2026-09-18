"""Top-level package for Guild."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("guild")
except PackageNotFoundError:
    # Not installed (e.g. guild/ reached via sys.path rather than a real
    # install) -- docs/conf.py needs this attribute to exist either way.
    __version__ = "0.0.0"
