"""
System constants
"""

from pathlib import Path

"""
Working directory path
"""
WORKING_DIR_PATH = Path(__file__).parent.parent.parent.absolute()
PYTHON_EXECUTABLE = f"{WORKING_DIR_PATH}/.venv/bin/python"
PROJECTS_FOLDER = f"{WORKING_DIR_PATH}/data"
SUPPORT_FOLDER = f"{WORKING_DIR_PATH}/guild/support"

"""
Shell silencer

Appended to commands run via ``subprocess.run(..., shell=True)`` to suppress
their stdout/stderr. Must be POSIX-compatible: in dash (Debian's /bin/sh) the
``&>file`` form is parsed as ``&`` (background) followed by ``>file`` (a
no-op redirect), so commands using the bash-only syntax get backgrounded and
the parent returns immediately — silently breaking any code that expects the
subprocess to have finished. ``> /dev/null 2>&1`` is portable across bash
and dash.
"""
SHELL_SILENCER = "> /dev/null 2>&1"
