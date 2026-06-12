"""
Non specific tools for the package.
"""

import json
import logging
import time
from functools import wraps

import pandas as pd


def read_json_as_dict(input_file):
    """
    Read a json file and return a nested dictionary.
    :param input_file: json file to be read
    """
    with open(input_file, "r") as file:
        nested_dictonary = json.load(file)
    return nested_dictonary


def timeit(log_file: str = None, log_level: int = logging.INFO):
    """
    Decorator to time a function and log execution time to specified log file.

    If log_file is None, the decorator will try to extract it from:
    1. self.output_log_file (for instance methods)
    2. Fall back to "time.log" if not found

    Usage:
        # As a method decorator (automatically uses self.output_log_file)
        @timeit()
        def my_method(self):
            pass

        # With explicit log file
        @timeit(log_file="custom.log")
        def my_function():
            pass

        # With log level
        @timeit(log_file="debug.log", log_level=logging.DEBUG)
        def another_function():
            pass

    :param log_file: Path to log file (default: None, will auto-detect from self.output_log_file)
    :param log_level: Logging level (default: logging.INFO)
    """

    def decorator(method):
        @wraps(method)
        def timed(*args, **kw):
            # Determine the log file to use
            actual_log_file = log_file

            # If log_file is None, try to get it from self.output_log_file
            if actual_log_file is None:
                if args and hasattr(args[0], "output_log_file"):
                    actual_log_file = args[0].output_log_file
                else:
                    actual_log_file = "time.log"

            # Set up logger for this specific log file
            logger = logging.getLogger(f"timeit_{method.__name__}")
            logger.setLevel(log_level)

            # Remove existing handlers to avoid duplicate logs
            if logger.handlers:
                logger.handlers.clear()

            # Create file handler
            file_handler = logging.FileHandler(actual_log_file)
            file_handler.setLevel(log_level)

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(file_handler)

            # Time the function execution
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()

            elapsed_time = te - ts

            # Log with function name and execution time
            logger.log(
                log_level, f"Function '{method.__name__}' executed in {elapsed_time:.4f} seconds"
            )

            # Clean up handler
            file_handler.close()
            logger.removeHandler(file_handler)

            return result

        return timed

    return decorator