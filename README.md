# Redactor: Local Hindi Video Censor

Redactor is a desktop application that automatically detects and censors potentially abusive spoken words within Hindi video files. 

**Key Feature:** The entire process—transcription, classification, and censorship—runs **locally** on your machine using AI models (OpenAI Whisper and Hugging Face Transformers). No data is sent to the cloud.

## Features

*   **Offline Privacy:** Videos are processed entirely on your computer.
*   **Hindi Speech Recognition:** Uses OpenAI Whisper to transcribe Hindi audio.
*   **Abuse Detection:** Uses a local BERT model (or keyword matching) to identify offensive terms.
*   **Smart Censoring:** Automatically beeps out the specific timestamps where bad words occur.
*   **GUI:** Simple, user-friendly interface built with CustomTkinter.

## Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **FFmpeg:** This is **mandatory** for video/audio processing.
    *   **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your System PATH environment variable.
    *   **macOS:** `brew install ffmpeg`
    *   **Linux:** `sudo apt install ffmpeg`

## Installation

1.  **Clone or Download** this repository.
2.  **Navigate** to the project folder:
    ```bash
    cd RedactorApp
    ```
3.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```
4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you have an NVIDIA GPU, install the CUDA version of PyTorch separately for much faster performance).*

## Configuration

### `config.json`
You can adjust settings in the `config.json` file:
*   `whisper_model_size`: "tiny", "base", "small", "medium" (Larger = more accurate but slower).
*   `classification_mode`: "model" (AI detection) or "keyword" (exact match from text file).
*   `abuse_confidence_threshold`: How confident the AI must be to censor a word (0.0 - 1.0).

### `keywords.txt`
If using `classification_mode`: "keyword" (or if the AI misses something), add words to `keywords.txt`:
*   Add one Hindi word per line.
*   Ensure the file is saved with **UTF-8 encoding**.

## Usage

1.  **Run the Application**:
    ```bash
    python main.py
    ```
2.  **First Run Notice**: On the very first run, the app will download the AI models (approx. 1GB+). The interface might hang momentarily; check your terminal for download progress.
3.  **Select Input**: Click "Browse Video..." to choose your `.mp4` file.
4.  **Select Output**: Click "Output Folder..." to choose where to save the censored video.
5.  **Process**: Click "Start Processing". 
6.  **Done**: The censored video will be saved as `filename_censored.mp4`.

## Troubleshooting

*   **"FileNotFoundError: [WinError 2] The system cannot find the file specified"**:
    *   This usually means **FFmpeg** is not installed or not in your system PATH. Restart your terminal after installing FFmpeg.
*   **Application Freezes on Start**:
    *   It is likely downloading the Whisper/BERT models. This happens only once.
*   **"CUDA out of memory"**:
    *   If using a GPU, try switching `whisper_model_size` to "tiny" or "base" in `config.json`.

## License

MIT License.
