import logging
import whisper # Using openai-whisper package
import time
import torch # Check for GPU availability
import gc # Garbage collector

from utils import CONFIG

# Global variables to hold the loaded model and device (avoids reloading)
_whisper_model = None
_model_device = None

def _check_device():
    """Determine the appropriate device for model execution (GPU or CPU)."""
    global _model_device
    if _model_device is None:
        # Prioritize CUDA
        if torch.cuda.is_available():
            _model_device = "cuda"
            logging.info("CUDA (NVIDIA GPU) available. Using GPU for Whisper.")
            # Optionally log GPU details
            try:
                gpu_index = 0 # Assuming first GPU
                logging.info(f"GPU Name: {torch.cuda.get_device_name(gpu_index)}")
                total_mem_gb = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
                logging.info(f"GPU Memory: {total_mem_gb:.2f} GB")
            except Exception as e_gpu:
                logging.warning(f"Could not get GPU details: {e_gpu}")
        # Check for Apple Silicon (MPS)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Check if MPS is usable, sometimes requires specific PyTorch versions
            try:
                # Test MPS availability more directly
                torch.ones(1, device="mps")
                _model_device = "mps"
                logging.info("MPS (Apple Silicon GPU) available and usable. Using MPS for Whisper.")
            except Exception as e_mps:
                logging.warning(f"MPS detected but test failed ({e_mps}). Falling back to CPU.")
                _model_device = "cpu"
                logging.info("Using CPU for Whisper.")
        # Fallback to CPU
        else:
            _model_device = "cpu"
            logging.info("No compatible GPU (CUDA/MPS) found. Using CPU for Whisper. This will be slower.")
    return _model_device


def load_whisper_model():
    """Loads the Whisper model specified in the config onto the determined device."""
    global _whisper_model
    if _whisper_model is None:
        model_size = CONFIG.get("whisper_model_size", "base")
        # Ensure device is checked before loading
        device = _check_device()
        logging.info(f"Attempting to load Whisper model: '{model_size}' onto device: '{device}'...")
        try:
            start_time = time.time()
            # Load model onto the selected device
            _whisper_model = whisper.load_model(model_size, device=device)
            end_time = time.time()
            logging.info(f"Whisper model '{model_size}' loaded successfully in {end_time - start_time:.2f} seconds.")
        except RuntimeError as e:
            # Catch potential CUDA OOM errors
            if device == "cuda" and "CUDA out of memory" in str(e):
                logging.error(f"CUDA out of memory trying to load model '{model_size}' on GPU. "
                              f"Try a smaller model size (e.g., 'tiny', 'base') in config.json "
                              f"or ensure sufficient GPU VRAM is available.", exc_info=False)
                _whisper_model = None # Ensure model is None
            else:
                logging.error(f"Runtime error loading Whisper model '{model_size}' on device '{device}': {e}", exc_info=True)
                _whisper_model = None
        except Exception as e:
            logging.error(f"Failed to load Whisper model '{model_size}' on device '{device}': {e}", exc_info=True)
            _whisper_model = None # Ensure it's None if loading failed

        # Optional: Force garbage collection after model loading if memory is tight
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return _whisper_model

def transcribe_audio(audio_path, progress_callback=None):
    """
    Transcribes audio using the loaded Whisper model.
    Returns a list of word timestamp dictionaries or None on failure.
    """
    model = load_whisper_model()
    if not model:
        logging.error("Whisper model is not loaded. Cannot transcribe.")
        if progress_callback: progress_callback("Error: Whisper model not loaded.")
        return None

    logging.info(f"Starting transcription for audio file: {audio_path}")
    word_timestamps = []
    try:
        start_time = time.time()
        if progress_callback: progress_callback("Transcription started...")

        # Determine if fp16 should be used (generally yes for CUDA, no for CPU/MPS)
        # fp16 might cause issues on older GPUs or certain setups.
        # MPS support for fp16 in Whisper can be inconsistent. Default to False for MPS.
        use_fp16 = (_model_device == "cuda")

        # Configure transcription options
        transcribe_options = {
            "language": 'hi',           # Set language to Hindi
            "word_timestamps": True,    # Get word-level timestamps
            "fp16": use_fp16,
            "verbose": None             # Set to False for less console noise, True/None for more detailed whisper logs
            # Consider adding beam_size or temperature if needed for quality tuning
            # "beam_size": 5,
            # "temperature": 0.0, # 0.0 for most deterministic output
            # "condition_on_previous_text": False # Keep default generally
        }
        logging.debug(f"Whisper transcribe options: {transcribe_options}")

        # Run transcription
        result = model.transcribe(audio_path, **transcribe_options)

        end_time = time.time()
        logging.info(f"Raw transcription completed in {end_time - start_time:.2f} seconds.")

        if progress_callback:
            progress_callback("Parsing transcription results...")

        # --- Robust Extraction of Word Timestamps ---
        text_result = result.get('text', '').strip()
        if not text_result:
            logging.warning("Transcription result contains no text.")
            # Return empty list but log warning, as it's not strictly an error
            return [] # Return empty list, processing can continue maybe

        logging.info(f"Full Transcription Text (first 100 chars): {text_result[:100]}...")

        # Check segments exist and iterate
        if 'segments' in result and isinstance(result['segments'], list) and result['segments']:
            for segment in result['segments']:
                # Check if segment is a dict and has words
                if isinstance(segment, dict) and 'words' in segment and isinstance(segment['words'], list):
                    for word_info in segment['words']:
                        # Check if word_info is a dictionary and has the required keys
                        if isinstance(word_info, dict) and 'word' in word_info and 'start' in word_info and 'end' in word_info:
                            word = word_info['word'].strip() # Strip whitespace from word itself
                            start = word_info['start']
                            end = word_info['end']
                            # Only add if word is not empty and timestamps are valid floats
                            if word: # Ensure word is not just whitespace
                                try:
                                    # Convert timestamps to float, handle potential errors
                                    start_f = float(start)
                                    end_f = float(end)
                                    # Basic validation: end time should not be before start time
                                    if end_f >= start_f:
                                        word_timestamps.append({'word': word, 'start': start_f, 'end': end_f})
                                    else:
                                        logging.warning(f"Skipping word '{word}' due to invalid timestamp order: start={start_f}, end={end_f}")
                                except (ValueError, TypeError) as e_ts:
                                    logging.warning(f"Skipping word '{word}' due to invalid timestamp format in word_info '{word_info}': {e_ts}")
                else:
                    # Log if the word_info structure is unexpected
                    logging.warning(f"Skipping word segment info due to unexpected format or missing keys: {word_info}")
        else:
            logging.warning("Transcription result has text but no 'segments' list or segments list is empty.")
            # Cannot extract word timestamps if segments are missing or invalid

        logging.info(f"Extracted {len(word_timestamps)} valid word timestamps.")
        if not word_timestamps:
            # Log if text exists but no timestamps were extracted
            logging.warning("No word timestamps were successfully extracted from the transcription result, although text was generated.")

        return word_timestamps

    except FileNotFoundError:
        logging.error(f"Audio file not found at path: {audio_path}")
        if progress_callback: progress_callback("Error: Audio file not found.")
        return None
    except RuntimeError as e:
        # Catch potential OOM errors during transcription more specifically
        if _model_device == "cuda" and "CUDA out of memory" in str(e):
            logging.error(f"CUDA out of memory during transcription. Try a smaller Whisper model or ensure sufficient GPU VRAM.", exc_info=False)
            if progress_callback: progress_callback("Error: Out of memory during transcription.")
        else:
            # Log other runtime errors with traceback
            logging.error(f"Runtime error during Whisper transcription: {e}", exc_info=True)
            if progress_callback: progress_callback("Error: Transcription runtime error.")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred during Whisper transcription: {e}", exc_info=True)
        if progress_callback: progress_callback("Error: Unexpected transcription error.")
        return None
    finally:
        # Optional: Force garbage collection after transcription if memory is tight
        gc.collect()
        if _model_device == "cuda":
            torch.cuda.empty_cache() # Clear PyTorch CUDA cache
