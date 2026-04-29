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
"""
SHELL_SILENCER = "&>/dev/null"
