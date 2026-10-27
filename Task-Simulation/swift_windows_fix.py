"""
Runtime fix for the Swift simulator on Windows.

Swift's file server receives mesh requests as /retrieve/<path>. On Windows the
path arrives as "/C:/Users/..." - the leading slash makes it invalid, so every
.stl/.dae returns "404 File not found" and the browser page crashes with
"Application error: a client-side exception has occurred".

Swift opens those files with the built-in open() looked up in the
swift.SwiftRoute module. We give that module its own open() that strips the
stray slash in front of a drive letter. Nothing is installed or edited on disk.

Usage: import this module before calling env.launch():
    import swift_windows_fix  # noqa: F401
"""
import builtins
import os
import re

import swift.SwiftRoute as _swift_route

# "/C:/Users/..." or "//C:\Users\..." -> "C:/Users/..."
_LEADING_SLASH_BEFORE_DRIVE = re.compile(r"^[/\\]+(?=[A-Za-z]:[/\\])")


def _open_with_windows_paths(file, *args, **kwargs):
    if isinstance(file, str):
        file = _LEADING_SLASH_BEFORE_DRIVE.sub("", file)
    return builtins.open(file, *args, **kwargs)


def apply(force=False):
    """Install the fix (only does anything on Windows unless force=True)."""
    if os.name == "nt" or force:
        _swift_route.open = _open_with_windows_paths
        return True
    return False


apply()
