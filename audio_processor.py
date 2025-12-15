from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.generators import Sine
from moviepy.editor import VideoFileClip
import os
import numpy as np # For potential alternative beep generation
import shutil # For copying files
import logging # ADDED: Missing in original PDF
from pathlib import Path # For path manipulation

from utils import get_temp_filepath, seconds_to_milliseconds, CONFIG

def extract_audio(video_path):
    """Extracts audio from video file and saves as 16kHz mono WAV."""
    logging.info(f"Attempting to extract audio from: {video_path}")
    temp_audio_path = get_temp_filepath("audio_extracted.wav")
    if not temp_audio_path:
        logging.error("Could not generate temporary audio file path.")
        return None

    video_clip_instance = None # Define outside try for potential cleanup
    try:
        logging.debug("Loading video clip for audio extraction...")
        video_clip_instance = VideoFileClip(video_path)

        if video_clip_instance.audio is None:
            logging.error(f"Video file '{Path(video_path).name}' has no audio track.")
            return None # Return None early

        logging.debug(f"Writing audio to temporary file: {temp_audio_path}")
        # Explicitly set parameters for consistency (Whisper prefers 16kHz mono)
        # Use logger=None to prevent moviepy progress bars in console/log
        # nbytes=2 corresponds to 16-bit depth (pcm_s16le)
        video_clip_instance.audio.write_audiofile(
            temp_audio_path,
            fps=16000,      # Target sample rate for Whisper
            nbytes=2,       # 16-bit depth
            codec='pcm_s16le', # Standard WAV codec for 16-bit
            logger=None,
            ffmpeg_params=["-ac", "1"] # Force mono channel using ffmpeg param
        )
        # Verify file exists after writing
        if not os.path.exists(temp_audio_path):
            logging.error("Audio file writing process completed but output file not found.")
            # Attempt cleanup just in case?
            if video_clip_instance: video_clip_instance.close()
            return None

        logging.info(f"Audio extracted successfully to: {temp_audio_path}")
        return temp_audio_path

    except Exception as e:
        # Provide more context in error logging
        logging.error(f"Error extracting audio from '{Path(video_path).name}': {e}", exc_info=True)
        # Clean up partial file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as e_rem:
                logging.error(f"Could not remove partial temp audio {temp_audio_path}: {e_rem}")
        return None
    finally:
        # Ensure video clip resources are released if instance was created
        if video_clip_instance:
            try:
                video_clip_instance.close()
                logging.debug("Video clip resources released.")
            except Exception as e_close:
                logging.warning(f"Exception while closing video clip: {e_close}")


def generate_beep(duration_ms, frequency_hz=440):
    """Generates a beep sound using pydub. Returns an AudioSegment."""
    # Ensure duration is a positive integer
    duration_ms = max(0, int(duration_ms))
    logging.debug(f"Generating beep: duration={duration_ms}ms, frequency={frequency_hz}Hz")

    if duration_ms <= 0:
        # Return silence of the requested (invalid) duration, which effectively is duration 0
        return AudioSegment.silent(duration=0)
    try:
        # Reduce amplitude to avoid harsh clipping (-12 dBFS is often reasonable)
        beep = Sine(frequency_hz).to_audio_segment(duration=duration_ms, volume=-12)
        # Apply a short fade in/out to prevent clicks (e.g., 5ms)
        fade_duration = min(5, duration_ms // 2) # Fade shouldn't exceed half duration
        if fade_duration > 0:
            beep = beep.fade_in(fade_duration).fade_out(fade_duration)
        return beep
    except Exception as e:
        logging.error(f"Error generating beep sound: {e}", exc_info=True)
        # Fallback: return silence of the same duration
        logging.warning("Falling back to generating silence due to beep generation error.")
        return AudioSegment.silent(duration=duration_ms)


def censor_audio(original_audio_path, segments_to_censor, output_path):
    """Replaces specified time segments in an audio file with beep sounds."""
    logging.info(f"Starting audio censoring for: {original_audio_path}")
    if not segments_to_censor:
        logging.info("No segments identified for censoring. Copying original audio.")
        try:
            shutil.copyfile(original_audio_path, output_path)
            logging.info(f"Copied original audio to: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error copying original audio file: {e}", exc_info=True)
            return None

    try:
        logging.debug(f"Loading original audio from {original_audio_path}...")
        # Load the original WAV audio
        audio = AudioSegment.from_wav(original_audio_path)
        original_duration_ms = len(audio)
        logging.info(f"Original audio loaded. Duration: {original_duration_ms / 1000.0:.2f}s")

        # Use beep frequency from config or default
        beep_frequency = CONFIG.get("beep_frequency_hz", 440)

        # Start with a copy to modify or build the new audio segment by segment
        final_audio = audio # Start with original, will overlay beeps

        applied_count = 0
        # Sort segments by start time to process sequentially and handle potential overlaps
        segments_to_censor.sort(key=lambda x: x[0])
        logging.debug(f"Sorted segments to censor: {segments_to_censor}")

        for i, (start_sec, end_sec) in enumerate(segments_to_censor):
            start_ms = seconds_to_milliseconds(start_sec)
            end_ms = seconds_to_milliseconds(end_sec)

            # --- Sanity checks and clamping ---
            start_ms = max(0, start_ms) # Ensure start isn't negative
            end_ms = max(start_ms, end_ms) # Ensure end isn't before start
            # Clamp to the actual audio duration
            end_ms = min(original_duration_ms, end_ms)
            start_ms = min(original_duration_ms, start_ms) # Clamp start too, in case end was clamped below start

            duration_ms = end_ms - start_ms

            if duration_ms <= 0:
                logging.warning(f"Skipping zero/negative duration segment {i+1} after clamping: {start_sec:.3f}s - {end_sec:.3f}s (Effective: {start_ms}ms - {end_ms}ms)")
                continue

            logging.debug(f"Processing segment {i+1}: Start={start_ms}ms, End={end_ms}ms, Duration={duration_ms}ms")

            # --- Generate Beep ---
            beep_sound = generate_beep(duration_ms, beep_frequency)
            # generate_beep now returns silence on error, so check duration might be better
            if beep_sound is None: # Check if beep generation failed critically
                logging.error(f"Failed to generate beep/silence for segment {i+1}. Skipping segment.")
                continue
            if len(beep_sound) == 0 and duration_ms > 0: # Check if generated silence is unexpectedly zero length
                logging.warning(f"Generated empty audio segment for non-zero duration {duration_ms}ms. Using generated silence.")
                # Proceed with the zero-length silence, effectively doing nothing, but log it.

            # --- Apply Beep (Overlay Method) ---
            # Pydub's overlay replaces the section of the first sound with the second.
            # Overlaying the 'beep_sound' onto the 'final_audio' at the calculated position.
            final_audio = final_audio.overlay(beep_sound, position=start_ms)
            applied_count += 1

        if applied_count > 0:
            logging.info(f"Applied beeps/silence to {applied_count} segments.")
            # Export the modified audio
            logging.debug(f"Exporting censored audio to {output_path}")
            final_audio.export(output_path, format="wav")
            logging.info(f"Censored audio successfully saved to: {output_path}")
            return output_path
        else:
            # If no segments were valid or beep generation failed for all
            logging.warning("No valid segments were censored. Copying original audio instead.")
            shutil.copyfile(original_audio_path, output_path)
            logging.info(f"Copied original audio to: {output_path} as no censoring was applied.")
            return output_path

    except CouldntDecodeError:
        logging.error(f"Pydub could not decode the audio file: {original_audio_path}. It might be corrupted or not a valid WAV file.")
        return None
    except FileNotFoundError:
        logging.error(f"Original audio file not found at: {original_audio_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during audio censoring process: {e}", exc_info=True)
        return None
