#!/usr/bin/env python
"""
Voice-activated LLM Chat via Whisper & Piper TTS with Tool Calling, Config, and Session Logging,
with added text input override.
-----------------------------------------------------------------------------------------------
This script now integrates:
  - Real-time noise reduction via the denoiser (version 0.1.5) package.
  - Loading of both Whisper base and medium models.
  - A consensus transcription strategy: The base model transcribes the audio while the medium model validates it.
    Both are run concurrently in separate threads; if their outputs agree (based on a similarity threshold),
    the transcription is accepted; otherwise, it’s discarded.
  - Chat history packaging into JSON with timestamps.
  - Additional functionalities including EQ enhancement, debug audio playback, tool calling, etc.
  - A new parallel text override mode: you can simply type your input and hit enter and it will be processed as if it were spoken.
  - **New! Image inference support:** You can now pass images to the model (e.g. "./image.png").
  - **New! Screen Capture Tool:** A tool call is available to capture the screen (using pyautogui) and save it as temp_screen.png.
  - **New! Image Query Conversion:** An internal function converts the user query into a more relevant image-based query, 
    whose output is then passed to the primary model.
"""

import sys, os, subprocess, platform, re, json, time, threading, queue, datetime, inspect, difflib
from datetime import datetime

# Define ANSI terminal colors for verbose logging
COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[94m"       # Blue for general info
COLOR_SUCCESS = "\033[92m"    # Green for success messages
COLOR_WARNING = "\033[93m"    # Yellow for warnings
COLOR_ERROR = "\033[91m"      # Red for error messages
COLOR_DEBUG = "\033[95m"      # Magenta for debug messages
COLOR_PROCESS = "\033[96m"    # Cyan for process steps

def log_message(message, category="INFO"):
    if category.upper() == "INFO":
         color = COLOR_INFO
    elif category.upper() == "SUCCESS":
         color = COLOR_SUCCESS
    elif category.upper() == "WARNING":
         color = COLOR_WARNING
    elif category.upper() == "ERROR":
         color = COLOR_ERROR
    elif category.upper() == "DEBUG":
         color = COLOR_DEBUG
    elif category.upper() == "PROCESS":
         color = COLOR_PROCESS
    else:
         color = COLOR_RESET
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{timestamp}] {category.upper()}: {message}{COLOR_RESET}")

# ----- VENV BOOTSTRAP CODE -----
def in_virtualenv():
    return sys.prefix != sys.base_prefix

def create_and_activate_venv():
    venv_dir = os.path.join(os.getcwd(), "whisper_env")
    if not os.path.isdir(venv_dir):
        log_message("Virtual environment 'whisper_env' not found. Creating it...", "PROCESS")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        log_message("Virtual environment created.", "SUCCESS")
    if not in_virtualenv():
        new_python = (os.path.join(venv_dir, "Scripts", "python.exe") 
                      if os.name == "nt" else os.path.join(venv_dir, "bin", "python"))
        log_message("Re-launching script inside the virtual environment...", "PROCESS")
        subprocess.check_call([new_python] + sys.argv)
        sys.exit()

if not in_virtualenv():
    create_and_activate_venv()

# ----- Automatic Piper and ONNX Files Setup -----
def setup_piper_and_onnx():
    """
    Ensure Piper executable and ONNX files are available.
    Downloads and extracts files if missing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe = os.path.join(piper_folder, "piper")
    log_message(f"Checking for Piper executable at {piper_exe}", "INFO")
    
    os_name = platform.system()
    machine = platform.machine()
    release_filename = ""
    log_message(f"Detected OS: {os_name} and Architecture: {machine}", "DEBUG")
    if os_name == "Linux":
        if machine == "x86_64":
            release_filename = "piper_linux_x86_64.tar.gz"
        elif machine == "aarch64":
            release_filename = "piper_linux_aarch64.tar.gz"
        elif machine == "armv7l":
            release_filename = "piper_linux_armv7l.tar.gz"
        else:
            log_message(f"Unsupported Linux architecture: {machine}", "ERROR")
            sys.exit(1)
    elif os_name == "Darwin":
        if machine in ("arm64", "aarch64"):
            release_filename = "piper_macos_aarch64.tar.gz"
        elif machine in ("x86_64", "AMD64"):
            release_filename = "piper_macos_x64.tar.gz"
        else:
            log_message(f"Unsupported macOS architecture: {machine}", "ERROR")
            sys.exit(1)
    elif os_name == "Windows":
        release_filename = "piper_windows_amd64.zip"
    else:
        log_message(f"Unsupported OS: {os_name}", "ERROR")
        sys.exit(1)
    
    if not os.path.isfile(piper_exe):
        log_message(f"Piper executable not found at {piper_exe}.", "WARNING")
        download_url = f"https://github.com/rhasspy/piper/releases/download/2023.11.14-2/{release_filename}"
        archive_path = os.path.join(script_dir, release_filename)
        log_message(f"Downloading Piper release from {download_url}...", "PROCESS")
        try:
            subprocess.check_call(["wget", "-O", archive_path, download_url])
        except Exception as e:
            log_message(f"Error downloading Piper archive: {e}", "ERROR")
            sys.exit(1)
        log_message("Download complete.", "SUCCESS")
        os.makedirs(piper_folder, exist_ok=True)
        if release_filename.endswith(".tar.gz"):
            subprocess.check_call(["tar", "-xzvf", archive_path, "-C", piper_folder, "--strip-components=1"])
        elif release_filename.endswith(".zip"):
            subprocess.check_call(["unzip", "-o", archive_path, "-d", piper_folder])
        else:
            log_message("Unsupported archive format.", "ERROR")
            sys.exit(1)
        log_message(f"Piper extracted to {piper_folder}", "SUCCESS")
    else:
        log_message(f"Piper executable found at {piper_exe}", "SUCCESS")
    
    # Check for ONNX files.
    onnx_json = os.path.join(script_dir, "glados_piper_medium.onnx.json")
    onnx_model = os.path.join(script_dir, "glados_piper_medium.onnx")
    if not os.path.isfile(onnx_json):
        log_message("ONNX JSON file not found. Downloading...", "WARNING")
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json"
        subprocess.check_call(["wget", "-O", onnx_json, url])
        log_message("Downloaded ONNX JSON file.", "SUCCESS")
    else:
        log_message("ONNX JSON file exists.", "SUCCESS")
    if not os.path.isfile(onnx_model):
        log_message("ONNX model file not found. Downloading...", "WARNING")
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx"
        subprocess.check_call(["wget", "-O", onnx_model, url])
        log_message("Downloaded ONNX model file.", "SUCCESS")
    else:
        log_message("ONNX model file exists.", "SUCCESS")

setup_piper_and_onnx()

# ----- Automatic Dependency Installation -----
SETUP_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".setup_complete")
if not os.path.exists(SETUP_MARKER):
    log_message("Installing required dependencies...", "PROCESS")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "sounddevice", "numpy", "scipy",
        "openai-whisper",   # For Whisper transcription
        "ollama",           # For Ollama Python API
        "python-dotenv",    # For environment variables
        "beautifulsoup4",   # For BS4 scraping
        "html5lib",         # Parser for BS4
        "pywifi",           # For WiFi scanning
        "psutil",           # For system utilization
        "num2words",        # For converting numbers to words
        "noisereduce",      # For noise cancellation (fallback, not used here)
        "denoiser",         # For real-time speech enhancement via denoiser
        "pyautogui",        # For screen capture
        "pillow",           # For image handling
    ])
    with open(SETUP_MARKER, "w") as f:
        f.write("Setup complete")
    log_message("Dependencies installed. Restarting script...", "SUCCESS")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- Configuration Loading -----
def load_config():
    """
    Load configuration from config.json (in the script directory). If not present, create it with default values.
    New keys include settings for primary/secondary models, temperatures, RMS threshold, debug audio playback,
    noise reduction, consensus threshold, and now image support.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    defaults = {
        "primary_model": "gemma3:12b",  # For conversation (used by other parts of the system)
        "secondary_model": "gemma3:4b",  # For tool calling (unused in current flow)
        "primary_temperature": 0.7,
        "secondary_temperature": 0.3,
        "onnx_json": "glados_piper_medium.onnx.json",
        "onnx_model": "glados_piper_medium.onnx",
        "whisper_model_primary": "base",      # Use "base" model for primary transcription
        "whisper_model_secondary": "medium",  # Use "medium" model for validation
        "stream": True,
        "raw": False,
        "images": None,        # Now support passing images to the primary model
        "options": {},
        "system": "You are a real‑time listener and responder for a voice‑activated assistant. Follow these rules strictly:\n\n1. Wait and Listen: You will receive small fragments of text (transcription “chunks”) from Whisper. Do not interrupt, summarize, or acknowledge each fragment. Instead, collect them silently until a complete utterance is formed.\n\n2. Decide When to Respond: If a fragment clearly ends a user’s thought (e.g., ends with a full sentence and pause), craft a response. If it looks like the user is talking to someone else, or you only have partial context, remain silent—do not respond with “okay” or “I understand.” Only speak when you’re confident you have a fully formed question or request.\n\n3. Maintain Conversational Flow: When you do respond, reply in a friendly, conversational tone—as if you were a helpful human assistant. Keep answers concise and to the point, but never robotic. Use natural phrasing and, when appropriate, gentle questions (“Could you tell me more about…?”, “What would you like me to do next?”).\n\n4. Handle Images: If an input comes with an image, describe what you see in detail: objects, colors, people, and their inferred actions or expressions. After describing, ask any clarifying questions if something is unclear.\n\n5. Avoid Hallucinations: Never invent facts or pretend to know something you don’t. If you’re unsure, ask the user for clarification. Do not end with “OK I understand” or similar placeholders. Silence is better than a false confirmation.\n\n6. Ask for Context When Needed: If you detect confusion or missing information, proactively ask a follow‑up: “I’m missing the end of that sentence—could you repeat it?” If multiple people are speaking, you may ask, “Who would you like me to address?”.\n\n7. Speaker Diarization Usage: Use internal speaker labels (e.g. SPEAKER_1, SPEAKER_2) only to understand turn‑taking and context. Under no circumstances should you echo, repeat, or reference these labels in your response—always speak naturally as a single assistant voice.\n\n8. Stay Focused on the User’s Intent: Your sole goal is to help the user with their spoken (or typed) requests. Do not drift off into tangents, unsolicited advice, or long self‑references.\n\n9. Keep It Human: Use contractions (I’m, you’re) and simple, clear language. Occasionally mirror the user’s phrasing to show you’re engaged.\n\n10. No Apologies or Placeholder Replies: If you do not understand, cannot respond, or lack sufficient context, do NOT apologize or say anything at all—simply remain silent until clear input is received.\n\n11. Music Detection: If you detect the exact same fragment repeated more than three times in succession, assume it is background music or noise and remain completely silent—do not attempt to transcribe or respond.\n\nBy following these numbered rules, you’ll respond naturally, only when appropriate, with clarity and empathy. Always prioritize listening and understanding over eager or misguided replies.",
        "conversation_id": "default_convo",
        "rms_threshold": 0.01,
        "debug_audio_playback": False,
        "enable_noise_reduction": True,
        "consensus_threshold": 0.8       # Similarity ratio required for consensus between models
    }
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump(defaults, f, indent=4)
        log_message("Created default config.json", "SUCCESS")
        return defaults
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
        log_message("Configuration loaded from config.json", "INFO")
        return config

config = load_config()

# ----- Chat Session Logging Setup -----
def setup_chat_session():
    """
    Create a session folder inside "chat_sessions", named by the current datetime.
    Each entry in the chat history includes a timestamp.
    Returns the session folder path and an open log file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sessions_folder = os.path.join(script_dir, "chat_sessions")
    os.makedirs(sessions_folder, exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(sessions_folder, session_name)
    os.makedirs(session_folder, exist_ok=True)
    session_log_path = os.path.join(session_folder, "session.txt")
    session_log = open(session_log_path, "a", encoding="utf-8")
    log_message(f"Chat session started: {session_name}", "SUCCESS")
    return session_folder, session_log

session_folder, session_log = setup_chat_session()

# ----- Import Additional Packages (after venv initialization and dependency installation) -----
from sounddevice import InputStream
from scipy.io.wavfile import write
import whisper
from ollama import chat, embed
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pywifi
from pywifi import const
import psutil
from num2words import num2words
import sounddevice as sd  # For debug audio playback
import numpy as np
from scipy.signal import butter, lfilter  # For EQ enhancement
import noisereduce as nr
import torch
from denoiser import pretrained  # For speech enhancement via denoiser
from PIL import ImageGrab  # Ensure Pillow is installed

load_dotenv()
brave_api_key = os.environ.get("BRAVE_API_KEY")
print("brave key loaded from .env: " + brave_api_key)
log_message("Environment variables loaded using dotenv", "INFO")

# ----- Load Whisper Models -----
log_message("Loading Whisper models...", "PROCESS")
try:
    whisper_model_primary = whisper.load_model(config["whisper_model_primary"])
    whisper_model_secondary = whisper.load_model(config["whisper_model_secondary"])
    log_message("Both base and medium Whisper models loaded.", "SUCCESS")
except torch.cuda.OutOfMemoryError as e:
    log_message(f"CUDA OutOfMemoryError while loading Whisper models: {e}", "ERROR")
    # free up all cached GPU memory
    torch.cuda.empty_cache()
    log_message("Cleared CUDA cache. Restarting script to recover...", "INFO")
    # re‑execute this script from scratch
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- Load Denoiser Model -----
log_message("Loading denoiser model (DNS64)...", "PROCESS")
denoiser_model = pretrained.dns64()   # Loads a pretrained DNS64 model from denoiser
log_message("Denoiser model loaded.", "SUCCESS")

# ----- Global Settings and Queues -----
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024
tts_queue = queue.Queue()
audio_queue = queue.Queue()
current_tts_process = None
tts_lock = threading.Lock()
log_message("Global settings and queues initialized.", "DEBUG")

def flush_current_tts():
    log_message("Flushing current TTS queue and processes...", "DEBUG")
    with tts_lock:
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
                tts_queue.task_done()
            except queue.Empty:
                break
        global current_tts_process
        if current_tts_process is not None:
            pproc, aproc = current_tts_process
            try:
                pproc.kill()
                log_message("Killed current Piper process.", "DEBUG")
            except Exception as e:
                log_message(f"Error killing Piper process: {e}", "ERROR")
            try:
                aproc.kill()
                log_message("Killed current aplay process.", "DEBUG")
            except Exception as e:
                log_message(f"Error killing aplay process: {e}", "ERROR")
            current_tts_process = None

# ----- TTS Processing -----
def process_tts_request(text):
    volume = config.get("tts_volume", 1.0)  # <-- new volume config

    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_exe = os.path.join(script_dir, "piper", "piper")
    onnx_json = os.path.join(script_dir, config["onnx_json"])
    onnx_model = os.path.join(script_dir, config["onnx_model"])

    # Verify that necessary files exist
    for path, desc in [
        (piper_exe, "Piper executable"),
        (onnx_json, "ONNX JSON file"),
        (onnx_model, "ONNX model file"),
    ]:
        if not os.path.isfile(path):
            log_message(f"Error: {desc} not found at {path}", "ERROR")
            return

    payload = {"text": text, "config": onnx_json, "model": onnx_model}
    payload_str = json.dumps(payload)

    cmd_piper = [piper_exe, "-m", onnx_model, "--debug", "--json-input", "--output_raw"]
    cmd_aplay = ["aplay", "--buffer-size=777", "-r", "22050", "-f", "S16_LE"]

    log_message(f"[TTS] Synthesizing: '{text}'", "INFO")

    try:
        with tts_lock:
            proc = subprocess.Popen(
                cmd_piper,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Intercept Piper stdout to adjust volume before sending to aplay
            proc_aplay = subprocess.Popen(cmd_aplay, stdin=subprocess.PIPE)
            global current_tts_process
            current_tts_process = (proc, proc_aplay)
            log_message("TTS processes started.", "DEBUG")

        # Send TTS request to Piper
        proc.stdin.write(payload_str.encode("utf-8"))
        proc.stdin.close()

        # Function to adjust volume on raw 16-bit PCM data
        def adjust_volume(data, vol):
            samples = np.frombuffer(data, dtype=np.int16)
            samples = (samples.astype(np.float32) * vol).clip(-32768, 32767).astype(np.int16)
            return samples.tobytes()

        # Read Piper's output, adjust volume, and send it to aplay
        chunk_size = 4096
        while True:
            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                break
            if volume != 1.0:
                chunk = adjust_volume(chunk, volume)
            proc_aplay.stdin.write(chunk)

        proc_aplay.stdin.close()
        proc_aplay.wait()
        proc.wait()

        with tts_lock:
            current_tts_process = None

        stderr_output = proc.stderr.read().decode("utf-8")
        if stderr_output:
            log_message("[Piper STDERR]: " + stderr_output, "ERROR")

    except Exception as e:
        log_message("Error during TTS processing: " + str(e), "ERROR")

def tts_worker(q):
    log_message("TTS worker thread started.", "DEBUG")
    while True:
        text = q.get()
        if text is None:
            log_message("TTS worker received shutdown signal.", "DEBUG")
            q.task_done()
            break
        process_tts_request(text)
        q.task_done()

def assemble_mixed_data(data):
    """
    Assembles a mixed set of variables, strings, and other data types into a single string.

    Args:
        data: An arbitrary collection (list, tuple, set, etc.) containing variables,
              strings, and other data types.  The order of elements in the input
              collection determines the order in the output string.

    Returns:
        A single string formed by converting each element in the input collection
        to a string and concatenating them in the original order.
    """

    result = ""
    for item in data:
        result += str(item)  # Convert each item to a string and append

    return result

# ----- Microphone Audio Capture -----
def audio_callback(indata, frames, time_info, status):
    if status:
        log_message("Audio callback status: " + str(status), "WARNING")
    audio_queue.put(indata.copy())

def start_audio_capture():
    log_message("Starting microphone capture...", "PROCESS")
    stream = InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
    stream.start()
    log_message("Microphone capture started.", "SUCCESS")
    return stream

def dynamic_range_normalize(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 20,
    hop_ms: float = 10,
    target_rms: float = 0.1,
    eps: float = 1e-6,
    smoothing_coef: float = 0.9
) -> np.ndarray:
    """
    1) Split into overlapping frames.
    2) Compute per-frame RMS.
    3) Compute gain = target_rms / (frame_rms + eps).
    4) Smooth gains across time.
    5) Apply and overlap-add back.
    """
    frame_len = int(sr * frame_ms/1000)
    hop_len   = int(sr * hop_ms/1000)
    # pad to fit an integer number of hops
    pad = (frame_len - (len(audio) - frame_len) % hop_len) % hop_len
    audio_p = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
    gains = []
    # analysis
    for start in range(0, len(audio_p)-frame_len+1, hop_len):
        frame = audio_p[start:start+frame_len]
        rms = np.sqrt(np.mean(frame**2))
        gain = np.sqrt(target_rms**2 / (rms**2 + eps))
        gains.append(gain)
    gains = np.array(gains, dtype=np.float32)
    # simple smoothing (IIR)
    for i in range(1, len(gains)):
        gains[i] = smoothing_coef * gains[i-1] + (1-smoothing_coef)*gains[i]
    # synthesis
    out = np.zeros_like(audio_p)
    win = np.hanning(frame_len)
    idx = 0
    for i, g in enumerate(gains):
        start = i*hop_len
        out[start:start+frame_len] += g * (audio_p[start:start+frame_len] * win)
    # compensate for the window overlap
    norm = np.zeros_like(audio_p)
    for i in range(len(gains)):
        norm[i*hop_len:i*hop_len+frame_len] += win
    out /= (norm + eps)
    return out[:len(audio)]

def apply_eq_and_denoise(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float = 300.0,
    highcut: float = 4000.0,
    eq_gain: float = 2.0,
    pre_emphasis_coef: float = 0.99,
    compress_thresh: float = 0.1,
    compress_ratio: float = 4.0
) -> np.ndarray:
    """
    0) Dynamic range normalization to boost distant/quiet speech.
    1) Noise reduction via spectral gating.
    2) Pre‑emphasis filter.
    3) Band‑pass EQ boost.
    4) Simple dynamic range compression.

    Args:
        audio:       1‑D float32 waveform (–1.0 to +1.0).
        sample_rate: Sampling rate in Hz.
        lowcut:      Low frequency cutoff for band‑pass.
        highcut:     High frequency cutoff for band‑pass.
        eq_gain:     Multiplier for the band‑passed component.
        pre_emphasis_coef:
                     Coefficient for pre‑emphasis filter.
        compress_thresh:
                     Threshold (fraction of max) above which to compress.
        compress_ratio:
                     Compression ratio (e.g. 4:1).

    Returns:
        Enhanced audio, float32 in [–1, +1].
    """
    # --- 0) Dynamic range normalization ---
    try:
        log_message("Enhancement: Normalizing dynamic range...", "PROCESS")
        audio = dynamic_range_normalize(
            audio,
            sample_rate,
            frame_ms=20,
            hop_ms=10,
            target_rms=0.3,
            smoothing_coef=0.9
        )
        log_message("Enhancement: Dynamic range normalization complete.", "SUCCESS")
    except Exception as e:
        log_message(f"Enhancement: Dynamic range normalization failed: {e}", "WARNING")

    # --- 1) Noise reduction ---
    try:
        log_message("Enhancement: Reducing noise via spectral gating...", "PROCESS")
        denoised = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=1.0,
            stationary=False
        )
        log_message("Enhancement: Noise reduction complete.", "SUCCESS")
    except Exception as e:
        log_message(f"Enhancement: Noise reduction failed: {e}", "WARNING")
        denoised = audio

    # --- 2) Pre‑emphasis filter (boost highs) ---
    log_message("Enhancement: Applying pre‑emphasis filter...", "PROCESS")
    emphasized = np.append(
        denoised[0],
        denoised[1:] - pre_emphasis_coef * denoised[:-1]
    )

    # --- 3) Band‑pass EQ boost ---
    log_message("Enhancement: Applying band‑pass EQ...", "PROCESS")
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype="band")
    band = lfilter(b, a, emphasized)
    eq_boosted = emphasized + (eq_gain - 1.0) * band

    # Normalize to prevent clipping before compression
    max_val = np.max(np.abs(eq_boosted))
    if max_val > 1.0:
        eq_boosted = eq_boosted / max_val

    # --- 4) Simple dynamic range compression ---
    log_message("Enhancement: Applying dynamic range compression...", "PROCESS")
    thresh = compress_thresh * np.max(np.abs(eq_boosted))
    compressed = np.copy(eq_boosted)
    over_thresh = np.abs(eq_boosted) > thresh
    compressed[over_thresh] = (
        np.sign(eq_boosted[over_thresh]) *
        (thresh + (np.abs(eq_boosted[over_thresh]) - thresh) / compress_ratio)
    )

    # Final normalization
    final_max = np.max(np.abs(compressed))
    if final_max > 1.0:
        compressed = compressed / final_max

    log_message("Enhancement: Audio enhancement complete.", "DEBUG")
    return compressed.astype(np.float32)


def clean_text(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    cleaned = text.replace("*", "").replace("```tool_output", "").replace("tool_call", "").replace("tool_output", "")
    return cleaned

# ----- Consensus Transcription Helper -----
def consensus_whisper_transcribe_helper(audio_array, language="en", rms_threshold=0.01, consensus_threshold=0.8):
    rms = np.sqrt(np.mean(np.square(audio_array)))
    if rms < rms_threshold:
        log_message("Audio RMS too low (RMS: {:.5f}). Skipping transcription.".format(rms), "WARNING")
        return ""
    
    transcription_base = ""
    transcription_medium = ""
    
    def transcribe_with_base():
        nonlocal transcription_base
        try:
            log_message("Starting base model transcription...", "PROCESS")
            result = whisper_model_primary.transcribe(audio_array, language=language)
            transcription_base = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
            log_message("Base transcription completed.", "SUCCESS")
        except Exception as e:
            log_message("Error during base transcription: " + str(e), "ERROR")
    
    def transcribe_with_medium():
        nonlocal transcription_medium
        try:
            log_message("Starting medium model transcription...", "PROCESS")
            result = whisper_model_secondary.transcribe(audio_array, language=language)
            transcription_medium = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
            log_message("Medium transcription completed.", "SUCCESS")
        except Exception as e:
            log_message("Error during medium transcription: " + str(e), "ERROR")
    
    thread_base = threading.Thread(target=transcribe_with_base)
    thread_medium = threading.Thread(target=transcribe_with_medium)
    thread_base.start()
    thread_medium.start()
    thread_base.join()
    thread_medium.join()
    
    if not transcription_base or not transcription_medium:
        log_message("One of the models returned no transcription.", "WARNING")
        return ""
    
    similarity = difflib.SequenceMatcher(None, transcription_base, transcription_medium).ratio()
    log_message(f"Transcription similarity: {similarity:.2f}", "INFO")
    if similarity >= consensus_threshold:
        log_message("Consensus reached between Whisper models.", "SUCCESS")
        return transcription_base
    else:
        log_message("No consensus between base and medium models; ignoring transcription.", "WARNING")
        return ""

# ----- Transcription Validation Helper -----
def validate_transcription(text):
    if not any(ch.isalpha() for ch in text):
        log_message("Transcription validation failed: no alphabetic characters.", "WARNING")
        return False
    words = text.split()
    if len(words) < 2:
        log_message("Transcription validation failed: fewer than 2 words.", "WARNING")
        return False
    log_message("Transcription validated successfully.", "SUCCESS")
    return True

# ----- Manager & Helper Classes for Tool-Calling Chat System -----
class ConfigManager:
    def __init__(self, config):
        config["model"] = config["primary_model"]
        self.config = config
        log_message("ConfigManager initialized with config.", "DEBUG")

class HistoryManager:
    def __init__(self):
        self.history = []
        log_message("HistoryManager initialized.", "DEBUG")
    def add_entry(self, role, content):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        log_message(f"History entry added for role '{role}'.", "INFO")

class TTSManager:
    def enqueue(self, text):
        log_message(f"Enqueuing text for TTS: {text}", "INFO")
        tts_queue.put(text)
    def stop(self):
        log_message("Stopping TTS processes.", "PROCESS")
        flush_current_tts()
    def start(self):
        log_message("Starting TTS processes.", "PROCESS")
        pass

class MemoryManager:
    def store_message(self, conversation_id, role, message, embedding):
        log_message("Storing message in MemoryManager.", "DEBUG")
        pass
    def retrieve_similar(self, conversation_id, embedding, top_n=3, mode="conversational"):
        log_message("Retrieving similar messages from MemoryManager.", "DEBUG")
        return []
    def retrieve_latest_summary(self, conversation_id):
        log_message("Retrieving latest summary from MemoryManager.", "DEBUG")
        return None

class ModeManager:
    def detect_mode(self, history):
        log_message("Detecting conversation mode.", "DEBUG")
        return "conversational"

class DisplayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_tokens = ""
        self.current_request = ""
        self.current_tool_calls = ""
        log_message("DisplayState initialized.", "DEBUG")
        
display_state = DisplayState()

class Utils:
    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        result = emoji_pattern.sub(r'', text)
        log_message("Emojis removed from text.", "DEBUG")
        return result
    @staticmethod
    def convert_numbers_to_words(text):
        def replace_num(match):
            number_str = match.group(0)
            try:
                return num2words(int(number_str))
            except ValueError:
                return number_str
        converted = re.sub(r'\b\d+\b', replace_num, text)
        log_message("Numbers converted to words in text.", "DEBUG")
        return converted
    @staticmethod
    def get_current_time():
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message("Current time retrieved.", "DEBUG")
        return current_time
    @staticmethod
    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            log_message("One of the vectors has zero norm in cosine similarity calculation.", "WARNING")
            return 0.0
        similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        log_message("Cosine similarity computed.", "DEBUG")
        return similarity
    @staticmethod
    def safe_load_json_file(path, default):
        if not path:
            return default
        if not os.path.exists(path):
            if default == []:
                with open(path, 'w') as f:
                    json.dump([], f)
            return default
        try:
            with open(path, 'r') as f:
                result = json.load(f)
            log_message(f"JSON file loaded from {path}.", "DEBUG")
            return result
        except Exception:
            log_message(f"Failed to load JSON file from {path}, returning default.", "ERROR")
            return default
    @staticmethod
    def load_format_schema(fmt):
        if not fmt:
            return None
        if fmt.lower() == "json":
            log_message("JSON format schema detected.", "DEBUG")
            return "json"
        if os.path.exists(fmt):
            try:
                with open(fmt, 'r') as f:
                    result = json.load(f)
                log_message("Format schema loaded from file.", "DEBUG")
                return result
            except Exception:
                log_message("Error loading format schema from file.", "ERROR")
                return None
        log_message("No valid format schema found.", "WARNING")
        return None
    @staticmethod
    def monitor_script(interval=5):
        script_path = os.path.abspath(__file__)
        last_mtime = os.path.getmtime(script_path)
        log_message("Monitoring script for changes...", "PROCESS")
        while True:
            time.sleep(interval)
            try:
                new_mtime = os.path.getmtime(script_path)
                if new_mtime != last_mtime:
                    log_message("Script change detected. Restarting...", "INFO")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                pass
    @staticmethod
    def embed_text(text):
        try:
            log_message("Embedding text for context.", "PROCESS")
            response = embed(model="nomic-embed-text", input=text)
            embedding = response['embeddings']
            vec = np.array(embedding, dtype=float)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            normalized = vec / norm
            log_message("Text embedding computed and normalized.", "SUCCESS")
            return normalized
        except Exception as e:
            log_message("Error during text embedding: " + str(e), "ERROR")
            return np.zeros(768)

import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
import psutil

class Tools:
    @staticmethod
    def parse_tool_call(text):
        pattern = r"```tool_(?:code|call)\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        result = match.group(1).strip() if match else None
        log_message("Parsed tool call from text.", "DEBUG")
        return result

    @staticmethod
    def capture_screen():
        try:
            import pyautogui
            log_message("Capturing screen using pyautogui...", "PROCESS")
            screenshot = pyautogui.screenshot()
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_screen.png")
            screenshot.save(file_path)
            log_message(f"Screen captured and saved to {file_path}", "SUCCESS")
            return file_path
        except Exception as e:
            log_message("Error capturing screen: " + str(e), "ERROR")
            return f"Error: {e}"

    @staticmethod
    def capture_screenshot():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "screen_states")
        if not os.path.exists(folder):
            os.makedirs(folder)
            log_message(f"Created folder for screenshots: {folder}", "SUCCESS")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        file_path = os.path.join(folder, filename)
        try:
            from PIL import ImageGrab
            log_message("Capturing screenshot using Pillow's ImageGrab...", "PROCESS")
            screenshot = ImageGrab.grab()
            screenshot.save(file_path)
            log_message(f"Screenshot captured and saved to {file_path}", "SUCCESS")
            return f"./screen_states/{filename}"
        except Exception as e:
            log_message("Error capturing screenshot: " + str(e), "ERROR")
            return f"Error: {e}"

    @staticmethod
    def convert_query_for_image(query, image_path):
        prompt = (f"Given the user query: '{query}', and the context of the image at '{image_path}', "
                  "convert this query into a more precise information query related to the image content.")
        log_message(f"Converting user query for image using prompt: {prompt}", "PROCESS")
        response = Tools.secondary_agent_tool(prompt, temperature=0.5)
        log_message("Image query conversion response: " + response, "SUCCESS")
        return response

    @staticmethod
    def load_image(image_path):
        full_path = os.path.abspath(image_path)
        if os.path.isfile(full_path):
            log_message(f"Image found at {full_path}", "SUCCESS")
            return full_path
        else:
            log_message(f"Image file not found: {full_path}", "ERROR")
            return f"Error: Image file not found: {full_path}"
        
    @staticmethod
    async def see_whats_around():
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            log_message(f"Images directory created at {images_dir}.", "SUCCESS")
        url = "http://127.0.0.1:8234/camera/default_0"
        import httpx
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_message(f"[Attempt {attempt+1}/{max_retries}] Attempting to capture image from {url}", "PROCESS")
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"camera_{timestamp}.jpg"
                        file_path = os.path.join(images_dir, filename)
                        with open(file_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                        log_message(f"Image captured and saved at: {file_path}", "SUCCESS")
                        return file_path
                    else:
                        log_message(f"Unexpected status code {response.status_code} while capturing image.", "WARNING")
            except httpx.RequestError as e:
                log_message(f"Network or request error occurred: {e}", "WARNING")
            log_message("Retrying after short delay...", "INFO")
            await asyncio.sleep(1)
        error_msg = "Error: Unable to capture image after multiple attempts."
        log_message(error_msg, "ERROR")
        return error_msg

    @staticmethod
    def get_battery_voltage():
        try:
            home_dir = os.path.expanduser("~")
            file_path = os.path.join(home_dir, "voltage.txt")
            with open(file_path, "r") as f:
                voltage = float(f.readline().strip())
            log_message("Battery voltage retrieved.", "SUCCESS")
            return voltage
        except Exception as e:
            log_message("Error reading battery voltage: " + str(e), "ERROR")
            raise RuntimeError(f"Error reading battery voltage: {e}")

    @staticmethod
    def brave_search(topic):
        api_key = brave_api_key
        if not api_key:
            log_message("BRAVE_API_KEY not set for brave search.", "ERROR")
            return "Error: BRAVE_API_KEY not set."
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "x-subscription-token": api_key
        }
        params = {"q": topic, "count": 3}
        try:
            import requests
            log_message(f"Performing brave search for topic: {topic}", "PROCESS")
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                log_message("Brave search successful.", "SUCCESS")
                return response.text
            else:
                log_message(f"Error in brave search: {response.status_code}", "ERROR")
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            log_message("Error in brave search: " + str(e), "ERROR")
            return f"Error: {e}"
        
    @staticmethod
    def search_internet(topic):
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            log_message("BRAVE_API_KEY not set for brave search.", "ERROR")
            return "Error: BRAVE_API_KEY not set."
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "x-subscription-token": api_key
        }
        params = {"q": topic, "count": 3}
        try:
            import requests
            log_message(f"Performing brave search for topic: {topic}", "PROCESS")
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                log_message("Brave search successful.", "SUCCESS")
                return response.text
            else:
                log_message(f"Error in brave search: {response.status_code}", "ERROR")
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            log_message("Error in brave search: " + str(e), "ERROR")
            return f"Error: {e}"

    @staticmethod
    def bs4_scrape(url):
        headers = {
            'User-Agent': ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
        }
        try:
            import requests
            log_message(f"Scraping URL: {url}", "PROCESS")
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            log_message("Webpage scraped successfully.", "SUCCESS")
            return soup.prettify()
        except Exception as e:
            log_message("Error during scraping: " + str(e), "ERROR")
            return f"Error during scraping: {e}"

    @staticmethod
    def find_file(filename, search_path="."):
        log_message(f"Searching for file: {filename} in path: {search_path}", "PROCESS")
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                log_message(f"File found in directory: {root}", "SUCCESS")
                return root
        log_message("File not found.", "WARNING")
        return None

    @staticmethod
    def get_current_location():
        try:
            import requests
            log_message("Retrieving current location based on IP.", "PROCESS")
            response = requests.get("http://ip-api.com/json", timeout=5)
            if response.status_code == 200:
                log_message("Current location retrieved.", "SUCCESS")
                return response.json()
            else:
                log_message("Error retrieving location: HTTP " + str(response.status_code), "ERROR")
                return {"error": f"HTTP error {response.status_code}"}
        except Exception as e:
            log_message("Error retrieving location: " + str(e), "ERROR")
            return {"error": str(e)}

    @staticmethod
    def get_system_utilization():
        utilization = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        log_message("System utilization retrieved.", "DEBUG")
        return utilization

    @staticmethod
    def secondary_agent_tool(prompt: str, temperature: float = 0.7) -> str:
        secondary_model = config["secondary_model"]
        payload = {
            "model": secondary_model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            log_message("Calling secondary agent tool.", "PROCESS")
            response = chat(model=secondary_model, messages=payload["messages"], stream=False)
            log_message("Secondary agent tool responded.", "SUCCESS")
            return response["message"]["content"]
        except Exception as e:
            log_message("Error in secondary agent: " + str(e), "ERROR")
            return f"Error in secondary agent: {e}"

class ChatManager:
    def __init__(self, config_manager: ConfigManager, history_manager: HistoryManager,
                 tts_manager: TTSManager, tools_data, format_schema,
                 memory_manager: MemoryManager, mode_manager: ModeManager):
        self.config_manager = config_manager
        self.history_manager = history_manager
        self.tts_manager = tts_manager
        self.tools_data = tools_data
        self.format_schema = format_schema
        self.memory_manager = memory_manager
        self.mode_manager = mode_manager
        self.current_tokens = ""
        self.current_tool_calls = ""
        self.stop_flag = False
        self.inference_lock = threading.Lock()
        self.current_thread = None
        self.conversation_id = self.config_manager.config.get("conversation_id", "default_convo")
        log_message("ChatManager initialized.", "DEBUG")
    
    def build_payload(self):
        log_message("ChatManager.build_payload: Building payload...", "DEBUG")
        cfg = self.config_manager.config
        system_prompt = cfg.get("system", "")
        tools_source = ""
        for attr in dir(Tools):
            if not attr.startswith("_"):
                method = getattr(Tools, attr)
                if callable(method):
                    try:
                        tools_source += "\n" + inspect.getsource(method)
                    except Exception:
                        pass
        tool_instructions = (
            "At each turn, if you decide to invoke any function, wrap the call in triple backticks with the label `tool_code`.\n\n"
            "Review the following Python methods (source provided for context) to decide if a function call is appropriate:\n\n"
            "```python\n" + tools_source + "\n```\n\n"
            "When a function call is executed, its response will be wrapped in triple backticks with the label `tool_output`."
        )
        system_message = {"role": "system", "content": system_prompt + "\n\n" + tool_instructions}
        log_message("ChatManager.build_payload: System message constructed.", "DEBUG")
        if self.history_manager.history:
            last_user_msg = next((msg["content"] for msg in reversed(self.history_manager.history) if msg["role"] == "user"), "")
            _ = Utils.embed_text(last_user_msg)
            mem_context = ""
            log_message("ChatManager.build_payload: Memory context extracted from history.", "DEBUG")
        else:
            mem_context = ""
        summary_text = ""
        memory_context = f"Memory Context:\n{mem_context}\n\nSummary Narrative:\n{summary_text}\n"
        memory_message = {"role": "system", "content": memory_context}
        messages = [system_message, memory_message] + self.history_manager.history
        payload = {
            "model": cfg["primary_model"],
            "temperature": cfg["primary_temperature"],
            "messages": messages,
            "stream": cfg["stream"]
        }
        if self.format_schema:
            payload["format"] = self.format_schema
        if cfg["raw"]:
            payload["raw"] = True
        if cfg["images"]:
            if self.history_manager.history and self.history_manager.history[-1].get("role") == "user":
                self.history_manager.history[-1]["images"] = cfg["images"]
        if self.tools_data:
            payload["tools"] = self.tools_data
        if cfg["options"]:
            payload["options"] = cfg["options"]
        log_message("ChatManager.build_payload: Payload built successfully.", "DEBUG")
        return payload

    def chat_completion_stream(self, processed_text):
        log_message("ChatManager.chat_completion_stream: Starting streaming chat completion...", "DEBUG")
        payload = self.build_payload()
        tokens = ""
        model = self.config_manager.config["primary_model"]
        try:
            stream = chat(model=model,
                          messages=payload["messages"],
                          stream=self.config_manager.config["stream"])
            for part in stream:
                if self.stop_flag:
                    log_message("ChatManager.chat_completion_stream: Stop flag detected.", "WARNING")
                    yield "", True
                    return
                content = part["message"]["content"]
                tokens += content
                with display_state.lock:
                    display_state.current_tokens = tokens
                log_message(f"ChatManager.chat_completion_stream: Received token: {content}", "DEBUG")
                yield content, part.get("done", False)
                if part.get("done", False):
                    log_message("ChatManager.chat_completion_stream: Streaming chat completion finished.", "DEBUG")
                    break
        except Exception as e:
            if "not found" in str(e).lower():
                log_message(f"ChatManager.chat_completion_stream: Model {model} not found, pulling it now...", "WARNING")
                try:
                    from ollama import pull
                    pull(model)
                    log_message(f"ChatManager.chat_completion_stream: Model {model} pulled successfully. Retrying...", "DEBUG")
                    stream = chat(model=model,
                                  messages=payload["messages"],
                                  stream=self.config_manager.config["stream"])
                    for part in stream:
                        if self.stop_flag:
                            log_message("ChatManager.chat_completion_stream: Stop flag detected during retry.", "WARNING")
                            yield "", True
                            return
                        content = part["message"]["content"]
                        tokens += content
                        with display_state.lock:
                            display_state.current_tokens = tokens
                        log_message(f"ChatManager.chat_completion_stream (retry): Received token: {content}", "DEBUG")
                        yield content, part.get("done", False)
                        if part.get("done", False):
                            log_message("ChatManager.chat_completion_stream (retry): Finished.", "DEBUG")
                            break
                except Exception as e2:
                    log_message("ChatManager.chat_completion_stream: Error during model pull or retry: " + str(e2), "ERROR")
                    yield "", True
            else:
                log_message("ChatManager.chat_completion_stream: Error during streaming chat completion: " + str(e), "ERROR")
                yield "", True

    def chat_completion_nonstream(self, processed_text):
        log_message("ChatManager.chat_completion_nonstream: Starting non-streaming chat completion...", "DEBUG")
        payload = self.build_payload()
        model = self.config_manager.config["primary_model"]
        try:
            response = chat(model=model,
                            messages=payload["messages"],
                            stream=False)
            log_message("ChatManager.chat_completion_nonstream: Non-streaming chat completion finished.", "DEBUG")
            return response["message"]["content"]
        except Exception as e:
            if "not found" in str(e).lower():
                log_message(f"ChatManager.chat_completion_nonstream: Model {model} not found, pulling it now...", "WARNING")
                try:
                    from ollama import pull
                    pull(model)
                    log_message(f"ChatManager.chat_completion_nonstream: Model {model} pulled successfully. Retrying...", "DEBUG")
                    response = chat(model=model,
                                    messages=payload["messages"],
                                    stream=False)
                    log_message("ChatManager.chat_completion_nonstream: Finished after retry.", "DEBUG")
                    return response["message"]["content"]
                except Exception as e2:
                    log_message("ChatManager.chat_completion_nonstream: Error during model pull or retry: " + str(e2), "ERROR")
                    return ""
            else:
                log_message("ChatManager.chat_completion_nonstream: Error during non-streaming chat completion: " + str(e), "ERROR")
                return ""

    def process_text(self, text, skip_tts=False):
        log_message("ChatManager.process_text: Starting processing of text.", "DEBUG")
        processed_text = Utils.convert_numbers_to_words(text)
        sentence_endings = re.compile(r'[.?!]+')
        tokens = ""
        if self.config_manager.config["stream"]:
            buffer = ""
            for content, done in self.chat_completion_stream(processed_text):
                log_message(f"ChatManager.process_text: Received content chunk: {content}", "DEBUG")
                buffer += content
                tokens += content
                with display_state.lock:
                    display_state.current_tokens = tokens
                while True:
                    match = sentence_endings.search(buffer)
                    if not match:
                        break
                    end_index = match.end()
                    sentence = buffer[:end_index].strip()
                    buffer = buffer[end_index:].lstrip()
                    sentenceCleaned = clean_text(sentence)
                    log_message(f"ChatManager.process_text: Extracted sentence: {sentenceCleaned}", "DEBUG")
                    if sentence and not skip_tts:
                        self.tts_manager.enqueue(sentenceCleaned)
                        log_message(f"ChatManager.process_text: Enqueued sentence for TTS: {sentenceCleaned}", "DEBUG")
                if done:
                    break
            if buffer.strip():
                tokens += buffer.strip()
                with display_state.lock:
                    display_state.current_tokens = tokens
            log_message("ChatManager.process_text: Completed processing in streaming mode.", "DEBUG")
            return tokens
        else:
            result = self.chat_completion_nonstream(processed_text)
            tokens = result
            with display_state.lock:
                display_state.current_tokens = tokens
            log_message("ChatManager.process_text: Completed processing in non-streaming mode.", "DEBUG")
            return tokens

    def inference_thread(self, user_message, result_holder, skip_tts):
        log_message("ChatManager.inference_thread: Started for message: " + user_message, "DEBUG")
        result = self.process_text(user_message, skip_tts)
        result_holder.append(result)
        log_message("ChatManager.inference_thread: Completed processing.", "DEBUG")

    def run_inference(self, prompt, skip_tts=False):
        log_message("ChatManager.run_inference: Starting inference for prompt: " + prompt, "DEBUG")
        result_holder = []
        with self.inference_lock:
            if self.current_thread and self.current_thread.is_alive():
                log_message("ChatManager.run_inference: Existing inference thread is still running; stopping it.", "WARNING")
                self.stop_flag = True
                self.current_thread.join()
                self.stop_flag = False
            self.tts_manager.stop()
            self.tts_manager.start()
            self.current_thread = threading.Thread(
                target=self.inference_thread,
                args=(prompt, result_holder, skip_tts)
            )
            log_message("ChatManager.run_inference: Starting new inference thread.", "DEBUG")
            self.current_thread.start()
        self.current_thread.join()
        log_message("ChatManager.run_inference: Inference thread joined. Result obtained.", "DEBUG")
        return result_holder[0] if result_holder else ""

    def run_tool(self, tool_code):
        log_message("ChatManager.run_tool: Executing tool call: " + tool_code, "DEBUG")
        allowed_tools = {}
        for attr in dir(Tools):
            if not attr.startswith("_"):
                method = getattr(Tools, attr)
                if callable(method):
                    allowed_tools[attr] = method
        try:
            result = eval(tool_code, {"__builtins__": {}}, allowed_tools)
            log_message("ChatManager.run_tool: Tool call executed successfully.", "DEBUG")
            return str(result)
        except Exception as e:
            log_message("ChatManager.run_tool: Error executing tool: " + str(e), "ERROR")
            return f"Error executing tool: {e}"

    def new_request(self, user_message, skip_tts=False):
        log_message("ChatManager.new_request: Received new request: " + user_message, "DEBUG")
        self.history_manager.add_entry("user", user_message)
        _ = Utils.embed_text(user_message)
        with display_state.lock:
            display_state.current_request = user_message
            display_state.current_tool_calls = ""
        result = self.run_inference(user_message, skip_tts)
        log_message("ChatManager.new_request: Inference result obtained: " + result, "DEBUG")
        tool_code = Tools.parse_tool_call(result)
        if tool_code:
            log_message("ChatManager.new_request: Tool call detected.", "DEBUG")
            tool_output = self.run_tool(tool_code)
            formatted_output = f"```tool_output\n{tool_output}\n```"
            combined_prompt = f"{user_message}\n{formatted_output}"
            self.history_manager.add_entry("user", combined_prompt)
            _ = Utils.embed_text(combined_prompt)
            log_message("ChatManager.new_request: Recursively processing tool output.", "DEBUG")
            final_result = self.new_request(combined_prompt, skip_tts=False)
            return final_result
        else:
            self.history_manager.add_entry("assistant", result)
            _ = Utils.embed_text(result)
            log_message("ChatManager.new_request: Assistant response recorded in history.", "DEBUG")
            return result
        
# ----- Voice-to-LLM Loop (for microphone input) -----
def voice_to_llm_loop(chat_manager: ChatManager, playback_lock, output_stream):
    import re
    log_message("Voice-to-LLM loop started. Listening for voice input...", "INFO")
    last_response = None
    max_words = config.get("max_response_words", 100)
    while True:
        time.sleep(3)
        if audio_queue.empty():
            continue

        # Gather all pending audio chunks
        chunks = []
        while not audio_queue.empty():
            chunks.append(audio_queue.get())
            audio_queue.task_done()
        if not chunks:
            continue

        log_message("Voice-to-LLM loop: Audio chunks collected for transcription.", "DEBUG")
        audio_data = np.concatenate(chunks, axis=0)
        audio_array = audio_data.flatten().astype(np.float32)

        # Combined noise reduction, EQ boost, and dynamic compression
        if config.get("enable_noise_reduction", True):
            log_message("Voice-to-LLM loop: Enhancing audio (denoise, EQ, compression)...", "PROCESS")
            audio_array = apply_eq_and_denoise(audio_array, SAMPLE_RATE)
            log_message("Voice-to-LLM loop: Audio enhancement complete.", "SUCCESS")

        # Prepare audio for transcription (already enhanced)
        audio_to_transcribe = audio_array

        # Async debug playback of the enhanced audio
        if config.get("debug_audio_playback", False):
            volume = config.get("debug_volume", 1.0)
            playback_chunk = audio_to_transcribe * volume
            log_message("Voice-to-LLM loop: Queueing async playback of enhanced audio...", "INFO")

            def _play_chunk_async(data_chunk):
                with playback_lock:
                    output_stream.write(data_chunk)

            threading.Thread(
                target=_play_chunk_async,
                args=(playback_chunk,),
                daemon=True
            ).start()

        # Transcription via Whisper with built-in timestamps
        log_message("Voice-to-LLM loop: Transcribing with Whisper segments...", "PROCESS")
        whisper_result = whisper_model_primary.transcribe(
            audio_to_transcribe,
            language="en",
            temperature=0.0
        )
        segments = whisper_result.get("segments", [])
        if not segments:
            log_message("Voice-to-LLM loop: No segments from Whisper.", "WARNING")
            continue

        # Assemble transcription with alternating speaker labels
        transcription = ""
        for idx, seg in enumerate(segments):
            speaker = f"SPEAKER_{(idx % 2) + 1}"
            transcription += f"[{speaker}] {seg['text'].strip()} "
        transcription = transcription.strip()

        if not validate_transcription(transcription):
            log_message("Voice-to-LLM loop: Transcription validation failed. Skipping.", "WARNING")
            continue

        log_message(f"Voice-to-LLM loop: Transcribed prompt: {transcription}", "INFO")
        session_log.write(json.dumps({
            "role": "user",
            "content": transcription,
            "timestamp": datetime.now().isoformat()
        }) + "")
        session_log.flush()

        flush_current_tts()
        response = chat_manager.new_request(transcription)
        # strip tool_output markers
        response = re.sub(r"```tool_output.*?```", "", response, flags=re.DOTALL).strip()
        # skip empty or repeated
        if not response or response == last_response:
            continue
        # detect hallucination: overly long
        if len(response.split()) > max_words:
            log_message(
                f"Voice-to-LLM loop: Hallucination detected (>{max_words} words). Discarding response and restarting inference.",
                "WARNING"
            )
            last_response = None
            continue
        last_response = response

        log_message("Voice-to-LLM loop: LLM response received.", "INFO")
        log_message("Voice-to-LLM loop: LLM response: " + response, "INFO")

        session_log.write(json.dumps({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        }) + "")
        session_log.flush()

        log_message("Voice-to-LLM loop: Awaiting further voice input...", "INFO")


# ----- New: Text Input Override Loop -----
def text_input_loop(chat_manager: ChatManager):
    log_message("Text input override mode is active.", "INFO")
    print("\nText override mode is active. Type your message and press Enter to send it to the LLM.")
    while True:
        try:
            user_text = input()  # Blocking call in its own thread.
            if not user_text.strip():
                continue
            log_message(f"Text input override: Received text: {user_text}", "INFO")
            session_log.write(json.dumps({"role": "user", "content": user_text, "timestamp": datetime.now().isoformat()}) + "\n")
            session_log.flush()
            flush_current_tts()
            response = chat_manager.new_request(user_text)
            log_message("Text input override: LLM response received.", "INFO")
            print("LLM response:", response)
            session_log.write(json.dumps({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}) + "\n")
            session_log.flush()
        except Exception as e:
            log_message("Error in text input loop: " + str(e), "ERROR")
# ----- Main Function -----
def main():
    # file watcher to restart on code or config.json changes
    def _monitor_files(interval=1):
        paths = [
            os.path.abspath(__file__),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        ]
        last_mtimes = {p: os.path.getmtime(p) for p in paths if os.path.exists(p)}
        while True:
            time.sleep(interval)
            for p in paths:
                try:
                    m = os.path.getmtime(p)
                    if last_mtimes.get(p) != m:
                        log_message(f"File change detected for '{os.path.basename(p)}', restarting...", "INFO")
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception:
                    continue

    monitor_thread = threading.Thread(target=_monitor_files, daemon=True)
    monitor_thread.start()

    log_message("Main function starting.", "INFO")

    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,))
    tts_thread.daemon = True
    tts_thread.start()
    log_message("Main: TTS worker thread launched.", "DEBUG")

    # One persistent output stream for debug playback
    playback_lock = threading.Lock()
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    output_stream.start()

    # Start microphone capture
    try:
        mic_stream = start_audio_capture()
    except Exception as e:
        log_message("Main: Error starting microphone capture: " + str(e), "ERROR")
        sys.exit(1)

    # Initialize managers
    config_manager = ConfigManager(config)
    history_manager = HistoryManager()
    tts_manager = TTSManager()
    memory_manager = MemoryManager()
    mode_manager = ModeManager()

    # Build ChatManager
    chat_manager = ChatManager(
        config_manager,
        history_manager,
        tts_manager,
        tools_data=True,
        format_schema=None,
        memory_manager=memory_manager,
        mode_manager=mode_manager
    )

    # Launch voice loop (pass in playback primitives)
    voice_thread = threading.Thread(
        target=voice_to_llm_loop,
        args=(chat_manager, playback_lock, output_stream)
    )
    voice_thread.daemon = True
    voice_thread.start()
    log_message("Main: Voice-to-LLM loop thread started.", "DEBUG")

    # Launch text‑override loop
    text_thread = threading.Thread(
        target=text_input_loop,
        args=(chat_manager,)
    )
    text_thread.daemon = True
    text_thread.start()
    log_message("Main: Text input override thread started.", "DEBUG")

    print(f"\nVoice-activated LLM mode (primary model: {config['primary_model']}) is running.")
    print("Speak into the microphone or type your prompt. LLM responses are streamed and spoken via Piper. Press Ctrl+C to exit.")

    # Keep alive until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log_message("Main: KeyboardInterrupt detected. Exiting...", "INFO")
        print("\nExiting...")

    # Clean shutdown
    mic_stream.stop()
    tts_queue.put(None)
    tts_queue.join()
    tts_thread.join()
    voice_thread.join()
    # Note: text_thread may remain blocked on input(), so we don’t join it
    session_log.close()

    # Save chat history
    history_path = os.path.join(session_folder, "chat_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_manager.history, f, indent=4)

    log_message(f"Chat session saved in folder: {session_folder}", "SUCCESS")
    log_message("Main: System terminated.", "INFO")
    print("Chat session saved in folder:", session_folder)
    print("System terminated.")

if __name__ == "__main__":
    main()
