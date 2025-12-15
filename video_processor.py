import logging
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import time # To ensure file handles are released
from utils import CONFIG # Could potentially use config for codecs etc.

def reassemble_video(original_video_path, censored_audio_path, output_video_path):
    """
    Combines the original video's visual stream with the provided (censored) audio track.
    Saves the result to the output video path. Handles resource cleanup.
    Returns True on success, False on failure.
    """
    logging.info(f"Starting video reassembly. Input Video: '{os.path.basename(original_video_path)}', "
                 f"Input Audio: '{os.path.basename(censored_audio_path)}', Output Target: '{output_video_path}'")
    video_clip = None
    audio_clip = None
    final_clip = None # Define outside try block for cleanup scope

    # --- Input validation ---
    if not os.path.isfile(original_video_path):
        logging.error(f"Original video file not found: {original_video_path}")
        return False
    if not os.path.isfile(censored_audio_path):
        logging.error(f"Censored audio file not found: {censored_audio_path}")
        return False

    try:
        # --- Load Clips ---
        # Load the original video *without* its audio to save memory/time
        logging.debug(f"Loading video stream from: {original_video_path}")
        video_clip = VideoFileClip(original_video_path, audio=False)
        # Check video duration after loading
        if video_clip.duration is None:
            logging.error("Could not determine duration of the video stream.")
            # Cannot proceed without video duration for comparison/setting audio
            return False
        video_duration = video_clip.duration
        logging.debug(f"Video stream loaded. Duration: {video_duration:.2f}s")

        # Load the censored audio
        logging.debug(f"Loading censored audio from: {censored_audio_path}")
        audio_clip = AudioFileClip(censored_audio_path)
        # Check audio duration
        if audio_clip.duration is None:
            logging.error("Could not determine duration of the censored audio.")
            # Might still proceed but log warning, or return False? Let's warn.
            logging.warning("Cannot verify audio duration, proceeding anyway.")
        audio_duration = video_duration # Assume it matches if unknown? Risky.
        else:
            audio_duration = audio_clip.duration
            logging.debug(f"Censored audio loaded. Duration: {audio_duration:.2f}s")

        # --- Duration Handling (Optional but Recommended) ---
        # If durations differ significantly, decide how to handle it.
        # Option 1: Warn the user (as implemented below)
        # Option 2: Trim/Pad the audio to match video duration
        duration_diff = abs(video_duration - audio_duration)
        # Set a threshold for warning, e.g., 1 second
        if duration_diff > 1.0:
            logging.warning(f"Video duration ({video_duration:.2f}s) and censored audio duration "
                            f"({audio_duration:.2f}s) differ significantly ({duration_diff:.2f}s). "
                            f"The final video length will be determined by the video stream. Audio might be cut short or padded with silence.")
            # Optional: Trim audio if it's longer than video
            if audio_duration > video_duration:
                logging.info(f"Trimming audio to video duration: {video_duration:.2f}s")
                audio_clip = audio_clip.subclip(0, video_duration)

        # --- Combine Video and Audio ---
        logging.debug("Setting new audio to video clip...")
        # The set_audio method creates a *new* clip object that needs cleanup
        final_clip = video_clip.set_audio(audio_clip)
        # Ensure the duration of the final clip is set correctly (usually inherits from video)
        if final_clip.duration is None: final_clip.duration = video_duration

        # --- Write Output File ---
        logging.info(f"Writing final combined video to: {output_video_path}")

        # Define codecs and parameters (could be moved to config.json)
        video_codec = 'libx264' # H.264, widely compatible
        audio_codec = 'aac'     # Common audio codec for MP4
        preset = 'medium'       # Speed/quality tradeoff: ultrafast, superfast, medium, slow, etc.
        threads = os.cpu_count() if os.cpu_count() else 4 # Use available cores
        # Generate a temporary filename for muxing audio that's less likely to collide
        temp_audio_mux_file = output_video_path + f".{int(time.time())}.temp_audio.mp3" # Adding timestamp

        final_clip.write_videofile(
            output_video_path,
            codec=video_codec,
            audio_codec=audio_codec,
            temp_audiofile=temp_audio_mux_file, # Recommended for stability
            remove_temp=True,                   # Clean up the temp muxing file
            threads=threads,
            preset=preset,
            logger=None                         # Suppress moviepy console progress bars
            # ffmpeg_params=["-metadata", "title=Redactor Censored Video"] # Example ffmpeg param
        )

        # Verify output file exists after writing attempt
        if not os.path.exists(output_video_path):
            logging.error("Video writing process seemed to complete but output file not found.")
            return False # Treat as failure if file isn't there

        logging.info(f"Video reassembly completed successfully. Output: {output_video_path}")
        return True

    except Exception as e:
        logging.error(f"An error occurred during video reassembly: {e}", exc_info=True)
        # Attempt to remove partial output file if creation failed
        if os.path.exists(output_video_path):
            try:
                logging.warning(f"Attempting to remove incomplete output file due to error: {output_video_path}")
                os.remove(output_video_path)
            except OSError as e_rem:
                logging.error(f"Could not remove incomplete output file {output_video_path}: {e_rem}")
        return False # Indicate failure
    finally:
        # --- Resource Cleanup ---
        # Ensure all clip objects are closed to release file handles
        logging.debug("Closing video/audio clip resources...")
        closed_count = 0
        # Use hasattr to be safe, though MoviePy clips should have close()
        # Close in reverse order of dependency? final -> audio -> video
        for clip_name, clip_obj in [("final_clip", final_clip), ("audio_clip", audio_clip), ("video_clip", video_clip)]:
            if clip_obj and hasattr(clip_obj, 'close'):
                try:
                    clip_obj.close()
                    closed_count += 1
                    logging.debug(f"Closed {clip_name}.")
                except Exception as e_close:
                    # Log error but continue trying to close others
                    logging.warning(f"Error closing media clip '{clip_name}': {e_close}")
            # else: (Optional log if clip was None)
            #     logging.debug(f"Clip '{clip_name}' was None or had no close method.")
        logging.debug(f"Attempted to close {closed_count} media clips.")
        # Short pause might sometimes help OS release handles, but usually not required
        # time.sleep(0.1)
