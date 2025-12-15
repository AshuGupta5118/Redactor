import os
import shutil
import logging
import json
from pathlib import Path
import uuid  # For unique temp filenames
import sys   # For exit on critical config error

CONFIG_FILE = 'config.json'
DEFAULT_TEMP_DIR = "redactor_temp"
DEFAULT_LOG_LEVEL = "INFO"

# --- Configuration Loading ---
def load_config():
    """
    Loads configuration from config.json found in the script's directory.
    Creates temp directory if specified and doesn't exist.
    Returns the config dictionary or None on critical error.
    """
    # Determine script directory to find config relative to it
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / CONFIG_FILE
    config_data = None

    if not config_path.is_file():
        # Use logging if possible, but it might not be initialized yet
        print(f"CRITICAL ERROR: Configuration file '{CONFIG_FILE}' not found in directory '{script_dir}'.")
        logging.critical(f"Configuration file '{CONFIG_FILE}' not found in directory '{script_dir}'.")
        return None  # Cannot proceed without config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # Don't log here yet, wait for logging setup
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: Failed to parse JSON in '{config_path}': {e}")
        logging.critical(f"Failed to parse JSON in '{config_path}': {e}")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred while loading configuration: {e}")
        logging.critical(f"An unexpected error occurred while loading configuration: {e}", exc_info=True)
        return None

    # Ensure temp directory exists AFTER successful config load
    try:
        # Use loaded config data, fallback to default name if key missing
        temp_dir_name = config_data.get("temp_dir", DEFAULT_TEMP_DIR)
        # Assume temp dir is relative to project root (script dir) unless absolute path given
        temp_dir_path = script_dir / temp_dir_name if not os.path.isabs(temp_dir_name) else Path(temp_dir_name)

        temp_dir_path.mkdir(parents=True, exist_ok=True)
        # Store absolute path in config dict for consistent access later
        config_data['temp_dir_path'] = str(temp_dir_path.resolve())
        # Log temp dir path AFTER logging is set up

    except Exception as e:
        # Log error, but maybe don't make it critical? App might work without temp files initially.
        print(f"ERROR: Error ensuring temporary directory '{temp_dir_name}' exists: {e}")
        logging.error(f"Error ensuring temporary directory '{temp_dir_name}' exists: {e}", exc_info=True)
        # Add placeholder path to config to avoid KeyErrors later?
        config_data['temp_dir_path'] = None # Indicate temp path issue

    return config_data

# --- Initial Config Load ---
# Load config immediately when the module is imported.
CONFIG = load_config()

# --- Logging Setup ---
# Setup logging AFTER trying to load config, as config might define log level/path
log_level_str = DEFAULT_LOG_LEVEL
if CONFIG:
    log_level_str = CONFIG.get("log_level", DEFAULT_LOG_LEVEL).upper()

# Determine logging level object from string
log_level = getattr(logging, log_level_str, logging.INFO)
if not isinstance(log_level, int):
    print(f"Warning: Invalid log level '{log_level_str}' in config. Defaulting to {DEFAULT_LOG_LEVEL}.")
    log_level = logging.INFO

# Define logging format
log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# Configure basic logging (send to console)
# Use force=True to clear any pre-existing handlers (useful for re-runs)
logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, force=True, stream=sys.stdout)

# --- Log initial status ---
if CONFIG:
    logging.info(f"Logging initialized at level {log_level_str}.")
    logging.info(f"Configuration loaded successfully from {CONFIG_FILE}")
    if CONFIG.get('temp_dir_path'):
        logging.info(f"Temporary directory set to: {CONFIG['temp_dir_path']}")
    else:
        logging.warning("Temporary directory path could not be established.")
else:
    # This message might not appear if logging init failed due to config load fail,
    # but critical errors are logged directly in load_config
    logging.error("Logging initialized, but configuration loading failed or returned None.")

# --- Temporary File Management ---
def get_temp_filepath(filename_suffix):
    """
    Generates a unique, absolute path for a temporary file within the configured temp directory.
    Returns the path as a string or None on error.
    """
    if not CONFIG or 'temp_dir_path' not in CONFIG or CONFIG['temp_dir_path'] is None:
        logging.error("Configuration or valid temp directory path not available. Cannot generate temp filepath.")
        return None
    try:
        temp_dir_path = Path(CONFIG['temp_dir_path'])
        # Use UUID for uniqueness to avoid collisions, especially if multiple runs happen quickly
        unique_id = uuid.uuid4()
        # Sanitize suffix: replace spaces, keep alphanumeric, dot, underscore, hyphen
        safe_suffix = "".join(c for c in filename_suffix if c.isalnum() or c in ('.', '_', '-')).strip()
        if not safe_suffix: safe_suffix = "file" # Ensure suffix isn't empty

        temp_filename = f"tmp_{unique_id}_{safe_suffix}"
        temp_path = temp_dir_path / temp_filename
        logging.debug(f"Generated temporary filepath: {temp_path}")
        return str(temp_path) # Return as string for compatibility
    except Exception as e:
        logging.error(f"Error generating temporary filepath: {e}", exc_info=True)
        return None

def cleanup_temp_files(*args):
    """Removes specified temporary files if they exist and logs activity."""
    logging.info("Attempting to clean up specified temporary files...")
    removed_count = 0
    skipped_count = 0
    for file_path in args:
        # Check if path is not None or empty before proceeding
        if file_path and isinstance(file_path, (str, Path)):
            path_obj = Path(file_path)
            if path_obj.is_file(): # Check if it exists and is a file
                try:
                    path_obj.unlink() # Use unlink for files
                    logging.debug(f"Removed temporary file: {path_obj}")
                    removed_count += 1
                except PermissionError:
                    logging.error(f"Permission error removing temp file {path_obj}. It might be in use.")
                    skipped_count += 1
                except OSError as e:
                    logging.error(f"OS error removing temp file {path_obj}: {e}")
                    skipped_count += 1
            # Optionally log if file didn't exist, but can be noisy
            elif not path_obj.exists():
                logging.debug(f"Temporary file specified for cleanup not found: {path_obj}")
                skipped_count += 1
            else: # Exists but is not a file (e.g., a directory)
                logging.warning(f"Path specified for temp file cleanup exists but is not a file: {path_obj}")
                skipped_count += 1
        else:
            # Log if None or invalid type was passed, helps debugging workflow issues
            # logging.debug(f"Invalid or empty file path provided for cleanup: {file_path}")
            skipped_count += 1 # Count Nones/invalid paths as skipped

    logging.info(f"Temporary file cleanup finished. Removed: {removed_count}, Skipped/Not Found/Errors: {skipped_count}.")

def cleanup_temp_directory():
    """Removes the entire temporary directory specified in the config."""
    if not CONFIG or 'temp_dir_path' not in CONFIG or CONFIG['temp_dir_path'] is None:
        logging.error("Configuration or valid temp directory path not available. Cannot clean temp directory.")
        return

    temp_dir_path = Path(CONFIG['temp_dir_path'])
    if temp_dir_path.exists() and temp_dir_path.is_dir():
        logging.info(f"Attempting to remove temporary directory: {temp_dir_path}")
        try:
            shutil.rmtree(temp_dir_path)
            logging.info(f"Successfully removed temporary directory: {temp_dir_path}")
        except PermissionError:
            logging.error(f"Permission error removing temp directory {temp_dir_path}. Check file permissions or running processes.")
        except OSError as e:
            logging.error(f"OS error removing temp directory {temp_dir_path}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error removing temp directory {temp_dir_path}: {e}", exc_info=True)
    else:
        # This might happen if it was already cleaned or never created properly
        logging.warning(f"Temporary directory '{temp_dir_path}' not found or is not a directory. Skipping cleanup.")

# --- Helper Function ---
def seconds_to_milliseconds(seconds):
    """Converts seconds (float or int) to milliseconds (int). Handles None input."""
    if seconds is None:
        logging.warning("seconds_to_milliseconds received None input, returning 0.")
        return 0
    try:
        # Ensure input is treated as float before multiplying
        # Use max(0, ...) to ensure result isn't negative if input was negative
        return max(0, int(float(seconds) * 1000))
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid input for seconds_to_milliseconds: '{seconds}'. Returning 0. Error: {e}")
        return 0
