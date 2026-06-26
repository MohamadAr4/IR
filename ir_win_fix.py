"""
Windows fix for ir_datasets download bug.

ir_datasets' Download.path() creates the download target via
tempfile.NamedTemporaryFile(delete=False) but never closes the handle.
Later, util.finialized_file() does os.replace('<tmp>.tmp', '<tmp>').
On POSIX you can replace a file that still has an open handle; on Windows
you cannot -> WinError 5 (Access is denied) / WinError 32 (file in use).

This shim replaces `tempfile` *only* inside ir_datasets.util.download so the
temp file's handle is closed immediately after creation (the file stays on
disk because delete=False). os.replace() can then overwrite it on Windows.

Import this module BEFORE loading any ir_datasets dataset:

    import ir_win_fix  # noqa: F401  (applies the patch on import)
    import ir_datasets
"""
import tempfile as _tempfile
import ir_datasets.util.download as _download


class _TempfileShim:
    """Proxies the real tempfile module but closes NamedTemporaryFile handles."""

    def NamedTemporaryFile(self, *args, **kwargs):
        f = _tempfile.NamedTemporaryFile(*args, **kwargs)
        f.close()  # release the Windows file handle so os.replace() can overwrite it
        return f

    def __getattr__(self, name):
        return getattr(_tempfile, name)


# Patch only the download module's view of `tempfile`; global tempfile untouched.
if not isinstance(getattr(_download, "tempfile", None), _TempfileShim):
    _download.tempfile = _TempfileShim()
