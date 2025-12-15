import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import os
import torch # To check device for transformers pipeline
import gc # Garbage collector

from utils import CONFIG

# Global variables to hold loaded models/pipelines/data
_classifier_pipeline = None
_keyword_list = None
_classifier_device = None # For transformers pipeline device ('cpu', 'cuda:0', etc.)

def _get_classifier_device():
    """Determine the appropriate device for the classification model (GPU or CPU)."""
    global _classifier_device
    if _classifier_device is None:
        if torch.cuda.is_available():
            _classifier_device_index = 0 # Use first available CUDA device
            _classifier_device = f"cuda:{_classifier_device_index}"
            logging.info(f"CUDA available. Using device '{_classifier_device}' for classification model.")
            # Optionally log GPU details
            try:
                gpu_name = torch.cuda.get_device_name(_classifier_device_index)
                logging.info(f"GPU Name: {gpu_name}")
            except Exception as e_gpu:
                logging.warning(f"Could not get GPU details: {e_gpu}")
        # Add MPS check if needed (device = "mps")
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     # Verify MPS usability
        #     try:
        #         torch.ones(1, device="mps")
        #         _classifier_device = "mps"
        #         logging.info("MPS available and usable. Using Apple Silicon GPU for classification.")
        #     except Exception as e_mps:
        #         logging.warning(f"MPS detected but test failed ({e_mps}). Falling back to CPU.")
        #         _classifier_device = "cpu"
        #         logging.info("Using CPU for classification model.")
        else:
            _classifier_device = "cpu" # Use CPU if no GPU
            logging.info("No compatible GPU (CUDA/MPS) found. Using CPU for classification model.")
    return _classifier_device


def load_classification_model():
    """
    Loads the classification model/pipeline based on config.json.
    Handles model download and caches the pipeline object.
    Returns the pipeline object or None on failure.
    """
    global _classifier_pipeline
    mode = CONFIG.get("classification_mode", "keyword") # Default to keyword

    if mode == "model" and _classifier_pipeline is None:
        model_name = CONFIG.get("classification_model_name")
        if not model_name:
            logging.error("Classification model name ('classification_model_name') not specified in config.json for 'model' mode.")
            return None

        # Get the target device ('cpu', 'cuda:0', etc.)
        target_device_str = _get_classifier_device()
        # Convert device string to device index for pipeline if it's CUDA
        target_device_index = -1 # Default to CPU
        if target_device_str.startswith("cuda"):
            try:
                target_device_index = int(target_device_str.split(":")[1])
            except (IndexError, ValueError):
                logging.warning(f"Could not parse CUDA device index from '{target_device_str}'. Defaulting to GPU 0.")
                target_device_index = 0
        # Add MPS handling if implemented
        # elif target_device_str == "mps":
        #     target_device_index = "mps" # Pipeline might accept the string "mps"

        logging.info(f"Attempting to load classification model: '{model_name}' onto device index: {target_device_index}...")

        try:
            start_time = time.time()
            # Load model and tokenizer explicitly first - helps with debugging sometimes
            # Consider adding trust_remote_code=True if required by the specific model hub entry
            # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=...)
            # model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=...)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Create the pipeline, specifying the device using integer index or -1 for CPU
            _classifier_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=target_device_index # Use determined device index (-1 for CPU, 0+ for GPU)
                # framework="pt" # Explicitly use PyTorch backend
            )
            end_time = time.time()
            logging.info(f"Classification model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")

        except OSError as e:
            # Handle common HuggingFace model download/not found errors
            logging.error(f"Failed to load model '{model_name}'. Ensure the model identifier is correct, you have internet access for the first download, and sufficient disk space. Error: {e}", exc_info=False)
            _classifier_pipeline = None
        except RuntimeError as e:
            if target_device_index >= 0 and "CUDA out of memory" in str(e): # Check if GPU was attempted
                logging.error(f"CUDA out of memory trying to load classification model '{model_name}' on GPU. Try a smaller model or run on CPU.", exc_info=False)
            else:
                logging.error(f"Runtime error loading classification model '{model_name}': {e}", exc_info=True)
            _classifier_pipeline = None
        except Exception as e:
            logging.error(f"An unexpected error occurred loading classification model '{model_name}': {e}", exc_info=True)
            _classifier_pipeline = None # Ensure it's None if loading failed

        # Optional: Garbage collection after loading
        gc.collect()
        if target_device_index >= 0: # If GPU was used
            torch.cuda.empty_cache()

    return _classifier_pipeline


def load_keyword_list():
    """
    Loads the keyword list from the file specified in config.json.
    Caches the loaded set of keywords (lowercase).
    Returns the set of keywords (can be empty).
    """
    global _keyword_list
    mode = CONFIG.get("classification_mode", "keyword")

    # Only load if in keyword mode and list hasn't been loaded yet
    if mode == "keyword" and _keyword_list is None:
        filepath = CONFIG.get("keyword_file", "keywords.txt")
        logging.info(f"Attempting to load keyword list from: {filepath}")
        keyword_set = set() # Initialize as empty set
        try:
            # Ensure the path exists before trying to open
            if not os.path.isfile(filepath):
                logging.warning(f"Keyword file not found: '{filepath}'. Keyword matching will be inactive.")
                # Fall through to assign empty set
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Read lines, strip whitespace, convert to lower case, remove empty lines and lines starting with '#'
                    keywords_from_file = {line.strip().lower() for line in f if line.strip() and not line.strip().startswith('#')}
                    keyword_set = keywords_from_file
                logging.info(f"Loaded {len(keyword_set)} keywords from '{filepath}'.")

        except Exception as e:
            logging.error(f"Failed to load or process keyword list from '{filepath}': {e}", exc_info=True)
            # Ensure it's an empty set on error, preventing NoneType issues later
            keyword_set = set()
        finally:
            # Cache the result (even if it's an empty set)
            _keyword_list = keyword_set

    # Return the cached (or newly loaded) list/set
    # If mode is not 'keyword', this might return None initially, but classify_word checks mode first
    return _keyword_list


def classify_word(word_text):
    """
    Classifies a single word as abusive or not based on the mode in config.json.
    Handles basic preprocessing and checks configured mode.
    Returns True if abusive, False otherwise.
    """
    # --- Input Validation and Preprocessing ---
    if not isinstance(word_text, str) or not word_text:
        # logging.debug("classify_word received empty or non-string input.")
        return False # Cannot classify non-string or empty string

    # Basic preprocessing: remove common leading/trailing punctuation and convert to lowercase
    # Note: Lowercasing might affect models sensitive to case, but simplifies keyword matching
    # and is often handled by model tokenizers anyway. Adjust if model requires case.
    processed_word = word_text.strip(".,!?;:'\"()[]{}<>-").strip().lower()
    if not processed_word:
        # logging.debug(f"Word '{word_text}' became empty after stripping punctuation.")
        return False # Ignore if word consists only of punctuation or whitespace

    # --- Classification Logic ---
    mode = CONFIG.get("classification_mode", "keyword")

    if mode == "model":
        # Load the pipeline (or get cached one)
        pipeline = load_classification_model()
        if not pipeline:
            logging.error("Classification pipeline is not available. Cannot classify using model.")
            return False # Fail safe: assume not abusive if model isn't loaded

        abuse_label = CONFIG.get("abuse_label")
        try:
            threshold = float(CONFIG.get("abuse_confidence_threshold", 0.75))
        except ValueError:
            logging.warning("Invalid 'abuse_confidence_threshold' in config. Using default 0.75.")
            threshold = 0.75

        if not abuse_label:
            logging.error("Abuse label ('abuse_label') not configured in config.json for model classification mode.")
            return False

        try:
            # Run inference using the pipeline
            # Pass the *original* case word if model is case-sensitive?
            # Let's stick with lowercase for now as it's often handled by tokenizer.
            # Use truncation=True for safety.
            results = pipeline(processed_word, truncation=True, max_length=64) # Adjust max_length if needed

            # Pipeline might return a list even for single input; take the first result safely
            result = results[0] if isinstance(results, list) and results else None

            if not result or not isinstance(result, dict) or 'label' not in result or 'score' not in result:
                logging.warning(f"Classification model returned unexpected result format for word '{processed_word}': {result}")
                return False

            # logging.debug(f"Classify(Model): Word='{processed_word}', Result={result}")

            # Check if the predicted label matches the configured abuse label
            # and the score meets the threshold
            predicted_label = result['label']
            score = result['score']
            if predicted_label == abuse_label and score >= threshold:
                logging.info(f"Abusive word detected (Model): '{word_text}' -> '{processed_word}' (Label: {predicted_label}, Score: {score:.4f})")
                return True
            else:
                # Log if prediction was made but didn't meet criteria
                # logging.debug(f"Word '{processed_word}' not classified as abusive (Label: {predicted_label}, Score: {score:.4f})")
                return False
        except Exception as e:
            # Log errors during the prediction phase for a specific word
            logging.error(f"Error during model classification for word '{processed_word}': {e}", exc_info=False) # Keep log concise
            return False # Fail safe

    elif mode == "keyword":
        # Load keywords (or get cached list)
        keywords = load_keyword_list()
        if keywords is None: # Check if loading failed completely
            logging.error("Keyword list is None (loading may have failed). Cannot classify using keywords.")
            return False
        if not keywords: # Check if list is empty
            # This case happens if file is missing or empty
            # logging.debug("Keyword list is empty. Cannot classify using keywords.")
            return False

        # Keyword matching uses the preprocessed (lowercase) word
        is_abusive = processed_word in keywords
        if is_abusive:
            logging.info(f"Abusive word detected (Keyword): '{word_text}' -> '{processed_word}'")
        return is_abusive

    else:
        logging.warning(f"Unknown classification mode specified in config: '{mode}'. Assuming not abusive.")
        return False
