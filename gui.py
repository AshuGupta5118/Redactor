from tkinter import filedialog, messagebox
import customtkinter as ctk # ADDED: Missing in original PDF
import os
import subprocess # For opening folder
import platform # To check OS for opening folder
from pathlib import Path
import sys # For sys.exit

from workflow import ProcessingWorkflow
# We don't strictly need CONFIG here, but might be useful for defaults later
# from utils import CONFIG

ctk.set_appearance_mode("System") # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue") # Themes: "blue" (default), "green", "dark-blue"

class RedactorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Redactor - Local Hindi Video Censor")
        self.geometry("600x450")
        self.minsize(500, 400) # Minimum window size

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) # Allow status frame to expand

        # --- Input File ---
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.input_button = ctk.CTkButton(self.input_frame, text="Browse Video...", command=self.browse_input_file)
        self.input_button.grid(row=0, column=0, padx=10, pady=10)
        self.input_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Select input video file")
        self.input_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.input_entry.configure(state="disabled") # Make it read-only initially
        self.selected_input_file = None

        # --- Output Directory ---
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.output_frame.grid_columnconfigure(1, weight=1)

        self.output_button = ctk.CTkButton(self.output_frame, text="Output Folder...", command=self.browse_output_folder)
        self.output_button.grid(row=0, column=0, padx=10, pady=10)
        self.output_entry = ctk.CTkEntry(self.output_frame, placeholder_text="Select output directory")
        self.output_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.output_entry.configure(state="disabled") # Make it read-only initially
        self.selected_output_dir = None

        # --- Options (Placeholder - Add model selection etc. if needed) ---
        # self.options_frame = ctk.CTkFrame(self)
        # self.options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # Add widgets for Whisper model size, confidence threshold, etc. here if desired

        # --- Status & Progress ---
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_frame.grid_rowconfigure(0, weight=1) # Allow textbox to expand vertically

        self.status_textbox = ctk.CTkTextbox(self.status_frame, wrap="word", fg_color="transparent")
        self.status_textbox.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        self.status_textbox.configure(state="disabled") # Start as read-only

        self.progressbar = ctk.CTkProgressBar(self.status_frame, orientation="horizontal")
        self.progressbar.set(0)
        self.progressbar.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        # --- Action Button ---
        self.action_button = ctk.CTkButton(self, text="Start Processing", command=self.start_processing, height=40)
        self.action_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.processing_thread = None

    def browse_input_file(self):
        """Opens file dialog to select input video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*"))
        )
        if file_path:
            self.selected_input_file = file_path
            self.input_entry.configure(state="normal") # Enable to insert text
            self.input_entry.delete(0, ctk.END)
            self.input_entry.insert(0, file_path)
            self.input_entry.configure(state="disabled") # Disable again
            self.log_status(f"Input file selected: {Path(file_path).name}")

    def browse_output_folder(self):
        """Opens directory dialog to select output folder."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.selected_output_dir = dir_path
            self.output_entry.configure(state="normal") # Enable to insert text
            self.output_entry.delete(0, ctk.END)
            self.output_entry.insert(0, dir_path)
            self.output_entry.configure(state="disabled") # Disable again
            self.log_status(f"Output directory selected: {dir_path}")

    def log_status(self, message):
        """Appends a message to the status textbox in a thread-safe way."""
        # Use 'after' to schedule the GUI update in the main thread
        # This prevents potential issues with Tkinter from background threads
        self.after(0, self._update_status_textbox, message)

    def _update_status_textbox(self, message):
        """Internal method to update textbox - *must* run in main thread."""
        current_state = self.status_textbox.cget("state")
        self.status_textbox.configure(state="normal")
        self.status_textbox.insert(ctk.END, message + "\n")
        self.status_textbox.see(ctk.END) # Scroll to the bottom
        self.status_textbox.configure(state=current_state) # Restore original state (usually disabled)
        # No need for update_idletasks() when using 'after'

    def update_progress(self, value):
        """Updates the progress bar value (thread-safe)."""
        # CustomTkinter widgets are generally thread-safe for configure/set methods
        # but using 'after' is the most robust way for GUI updates from threads
        self.after(0, self.progressbar.set, value)

    def processing_complete(self, success, final_path):
        """Callback function when the workflow finishes (runs in main thread)."""
        self.action_button.configure(state="normal", text="Start Processing") # Re-enable button
        if success and final_path:
            self.log_status("Processing finished successfully!")
            self.log_status(f"Output file: {final_path}")
            # Optionally offer to open the output folder
            if messagebox.askyesno("Success", f"Processing complete!\nOutput: {final_path}\n\nOpen output folder?"):
                self._open_folder(self.selected_output_dir)
        elif success and not final_path: # Should not happen if success is True
            self.log_status("Processing reported success but no output path was returned.")
        else:
            self.log_status("Processing failed or was cancelled.")
            # Optionally show an error message box as well
            # messagebox.showerror("Processing Failed", "An error occurred during processing. Please check the status messages and logs.")
        self.processing_thread = None # Clear the thread reference

    def _open_folder(self, folder_path):
        """Opens the specified folder in the system's file explorer."""
        try:
            if platform.system() == "Windows":
                os.startfile(os.path.realpath(folder_path)) # Use realpath for safety
            elif platform.system() == "Darwin": # macOS
                subprocess.Popen(["open", folder_path])
            else: # Linux
                # FIXED: 'xdg - open' typo to 'xdg-open'
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            self.log_status(f"Could not open output folder automatically: {e}")
            messagebox.showwarning("Open Folder", f"Could not automatically open the folder.\nPlease navigate to:\n{folder_path}")

    def start_processing(self):
        """Validates inputs and starts the processing workflow thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # Validate inputs
        if not self.selected_input_file or not os.path.isfile(self.selected_input_file):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return
        if not self.selected_output_dir or not os.path.isdir(self.selected_output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return

        # Clear previous status and reset progress
        self.status_textbox.configure(state="normal")
        self.status_textbox.delete("1.0", ctk.END)
        self.status_textbox.configure(state="disabled")
        self.progressbar.set(0)

        self.log_status(f"Starting processing for: {Path(self.selected_input_file).name}")
        self.action_button.configure(state="disabled", text="Processing...") # Disable button

        # --- Start the workflow in a separate thread ---
        self.processing_thread = ProcessingWorkflow(
            video_path=self.selected_input_file,
            output_dir=self.selected_output_dir,
            # Pass GUI update methods as callbacks
            status_callback=self.log_status,
            progress_callback=self.update_progress,
            # Ensure completion_callback runs in the main GUI thread using 'after'
            completion_callback=lambda s, p: self.after(0, self.processing_complete, s, p)
        )
        self.processing_thread.start()

    def on_closing(self):
        """Handle window closing event, prompting if processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askyesno("Exit Confirmation", "Processing is ongoing.\nExiting now might leave temporary files or result in incomplete output.\nAre you sure you want to exit?"):
                # Attempt to signal the thread to stop gracefully
                if hasattr(self.processing_thread, 'stop'):
                    self.log_status("Attempting to stop processing thread...")
                    self.processing_thread.stop()
                    # Ideally, we might wait briefly for the thread to acknowledge
                    # self.processing_thread.join(timeout=1.0) # Wait max 1 sec
                self.destroy() # Close window regardless if user confirms
            else:
                return # Don't close if user selects 'No'
        else:
            self.destroy() # Destroy window if no thread is running
