import threading
import os
import logging # ADDED: Missing in original PDF
from pathlib import Path
import time # For timing workflow

# Import functions from other modules
from audio_processor import extract_audio, censor_audio
from stt_processor import transcribe_audio
from classification_processor import classify_word
from video_processor import reassemble_video
# Need utils for cleanup and path generation
from utils import cleanup_temp_files, get_temp_filepath, CONFIG, cleanup_temp_directory

class ProcessingWorkflow(threading.Thread):
    """Runs the entire video processing workflow in a separate thread."""

    def __init__(self, video_path, output_dir, status_callback, progress_callback, completion_callback):
        super().__init__()
        # Make thread a daemon so main program can exit even if this thread hangs (though cleanup might be missed)
        self.daemon = True
        self.video_path = video_path
        self.output_dir = output_dir
        self.status_callback = status_callback      # Function to update status label in GUI
        self.progress_callback = progress_callback  # Function to update progress bar in GUI
        self.completion_callback = completion_callback # Function called when done (success or fail)
        self.temp_audio_original = None
        self.temp_audio_censored = None
        self._stop_requested = threading.Event()    # Flag to signal stopping

    def run(self):
        """Executes the processing steps."""
        final_output_path = None
        success = False
        start_time_workflow = time.time()
        logging.info(f"Workflow started for: {self.video_path}")

        try:
            # --- Input Validation ---
            if not self.video_path or not os.path.isfile(self.video_path):
                self.update_status("Error: Input video file not found or invalid.")
                return # Exit thread early

            if not self.output_dir or not os.path.isdir(self.output_dir):
                self.update_status("Error: Output directory is invalid.")
                return # Exit thread early

            # --- Step 1: Extract Audio ---
            step_start_time = time.time()
            self.update_status("Step 1/5: Extracting audio...")
            self.update_progress(5)
            self.temp_audio_original = extract_audio(self.video_path)
            if self._check_stop_or_error(self.temp_audio_original, "Error: Failed to extract audio."): return
            logging.info(f"Step 1 finished in {time.time() - step_start_time:.2f} seconds.")

            # --- Step 2: Transcribe Audio ---
            step_start_time = time.time()
            self.update_status("Step 2/5: Transcribing audio (this may take time)...")
            self.update_progress(20)
            word_timestamps = transcribe_audio(self.temp_audio_original, self.update_status) # Pass status callback
            # Check specifically for None return on transcription error
            if self._check_stop_or_error(word_timestamps, "Error: Failed to transcribe audio.", check_none=True): return
            logging.info(f"Step 2 finished in {time.time() - step_start_time:.2f} seconds.")

            # --- Step 3: Classify Words ---
            step_start_time = time.time()
            num_words = len(word_timestamps) if word_timestamps else 0
            self.update_status(f"Step 3/5: Classifying {num_words} words...")
            self.update_progress(50)
            segments_to_censor = []

            if num_words == 0:
                logging.warning("Transcription resulted in zero words. Skipping classification.")
            else:
                for i, item in enumerate(word_timestamps):
                    if self._stop_requested.is_set(): return # Check frequently during long loops
                    word = item.get('word', '') # Use .get for safety
                    start_time = item.get('start')
                    end_time = item.get('end')

                    if word and start_time is not None and end_time is not None:
                        if classify_word(word): # classify_word handles its own logging
                            segments_to_censor.append((start_time, end_time))

                    # Update progress more granularly
                    if i % 50 == 0 or i == num_words - 1: # Update every 50 words or at the end
                        # Progress for this step goes from 50% to 70%
                        progress = 50 + int(20 * (i + 1) / num_words) if num_words > 0 else 70
                        self.update_progress(progress)

            self.update_status(f"Identified {len(segments_to_censor)} segments to censor.")
            self.update_progress(70)
            if self._stop_requested.is_set(): return
            logging.info(f"Step 3 finished in {time.time() - step_start_time:.2f} seconds.")

            # --- Step 4: Censor Audio ---
            step_start_time = time.time()
            self.update_status("Step 4/5: Generating censored audio...")
            self.temp_audio_censored = get_temp_filepath("audio_censored.wav")
            if self._check_stop_or_error(self.temp_audio_censored, "Error: Could not create path for censored audio."): return

            censored_path = censor_audio(self.temp_audio_original, segments_to_censor, self.temp_audio_censored)
            if self._check_stop_or_error(censored_path, "Error: Failed to censor audio."): return
            self.update_progress(85)
            logging.info(f"Step 4 finished in {time.time() - step_start_time:.2f} seconds.")

            # --- Step 5: Reassemble Video ---
            step_start_time = time.time()
            self.update_status("Step 5/5: Reassembling video (this may also take time)...")
            base_name = Path(self.video_path).stem
            # Use a safe output filename, avoiding overwriting original
            output_filename = f"{base_name}_censored.mp4"
            final_output_path = os.path.join(self.output_dir, output_filename)

            # Handle potential filename collision simply
            counter = 1
            while os.path.exists(final_output_path):
                output_filename = f"{base_name}_censored_{counter}.mp4"
                final_output_path = os.path.join(self.output_dir, output_filename)
                counter += 1

            reassembly_ok = reassemble_video(self.video_path, self.temp_audio_censored, final_output_path)
            if not reassembly_ok:
                # Error logged inside reassemble_video
                self.update_status("Error: Failed to reassemble video.")
                # Don't return immediately, allow cleanup in finally block
            elif self._stop_requested.is_set():
                # If stop was requested *during* reassembly (less likely to be caught)
                self.update_status("Stop requested during video reassembly.")
                return
            else:
                # --- Done ---
                end_time_workflow = time.time()
                duration = end_time_workflow - start_time_workflow
                self.update_status(f"Processing complete in {duration:.2f} seconds!")
                # No need to log output path here, completion callback handles it
                self.update_progress(100)
                success = True # Mark as success only if reassembly finished ok
                logging.info(f"Step 5 finished in {time.time() - step_start_time:.2f} seconds.")


        except Exception as e:
            # Catch any unexpected exceptions during the process
            logging.exception("An unexpected error occurred in the processing workflow.") # Log full traceback
            self.update_status(f"Error: An unexpected error occurred. Check logs.") # User-friendly message
            success = False # Ensure success is false
        finally:
            # This block executes whether the try block succeeded, failed, or was stopped
            logging.info("Workflow finalizing. Cleaning up temporary files...")
            cleanup_temp_files(self.temp_audio_original, self.temp_audio_censored)
            # Consider if the entire temp directory should be cleaned
            # cleanup_temp_directory()
            if self._stop_requested.is_set():
                # If stop was requested *before* this finally block
                self.update_status("Processing stopped.")
                success = False # Mark as not successful if stopped
                final_output_path = None # No valid output path if stopped

            # Notify the GUI thread about completion status
            self.completion_callback(success, final_output_path)
            logging.info(f"Workflow finished. Success: {success}")

    def update_status(self, message):
        """Safely updates the status label in the GUI and logs the message."""
        logging.info(f"Status Update: {message}") # Log first
        if self.status_callback:
            # The GUI's log_status method should handle thread safety (e.g., using 'after')
            self.status_callback(message)

    def update_progress(self, value):
        """Safely updates the progress bar in the GUI."""
        if self.progress_callback:
            # Ensure value is between 0.0 and 1.0 for progress bar
            progress_float = max(0.0, min(1.0, value / 100.0))
            # The GUI's update_progress method should handle thread safety
            self.progress_callback(progress_float)

    def _check_stop_or_error(self, result, error_message, check_none=False):
        """Checks stop request or if result indicates an error. Returns True if should stop."""
        if self._stop_requested.is_set():
            self.update_status("Processing stopped by user.")
            logging.info("Stop request detected during check.")
            return True # Stop requested

        error_condition = False
        if check_none:
            # Specifically check if the result is None
            if result is None:
                error_condition = True
        elif not result: # Catches False, empty strings, 0, etc.
            error_condition = True

        if error_condition:
            self.update_status(error_message) # Log error status to GUI
            logging.error(f"Workflow Error Check: {error_message} (Result evaluated as False/None)")
            return True # Error occurred, should stop

        return False # No stop, no error, continue

    def stop(self):
        """Signals the thread to stop processing gracefully."""
        # Can be called from the GUI thread
        self.update_status("Stop requested. Attempting to halt after current step...")
        self._stop_requested.set()
