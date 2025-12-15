from gui import RedactorApp
from utils import cleanup_temp_directory, load_config, CONFIG
import logging
import sys # To potentially show GUI error if config fails
import tkinter as tk
from tkinter import messagebox

# --- Define a simple GUI error display if needed ---
def show_critical_error(message):
    """Displays a simple Tkinter error message box and exits."""
    root = tk.Tk()
    root.withdraw() # Hide the main Tk window
    messagebox.showerror("Critical Error", message)
    root.destroy()
    sys.exit(1)

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure config is loaded at startup
    if not CONFIG:
        logging.critical("Failed to load configuration file 'config.json'. Application cannot start.")
        # Show a user-friendly error before exiting
        show_critical_error("Failed to load configuration file 'config.json'.\nPlease ensure it exists and is correctly formatted.\nApplication will now exit.")
        # The show_critical_error function handles the exit

    # Optional: Clean up temp directory from previous runs at startup
    # Consider if this is desired behaviour - might remove files needed for debugging
    # cleanup_temp_directory()

    try:
        app = RedactorApp()
        # FIXED: "WM_DELETE_WINDOW" (removed spaces)
        app.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle closing gracefully
        app.mainloop()
    except Exception as e:
        logging.critical("An unhandled exception occurred in the main application loop.", exc_info=True)
        show_critical_error(f"An unexpected error occurred:\n{e}\nPlease check the logs.\nApplication will now exit.")

    # Optional: Clean up temp directory on successful exit
    # cleanup_temp_directory()
    logging.info("Redactor application finished.")
    sys.exit(0) # Ensure clean exit code
