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

import sys, os, subprocess, platform, re, json, time, threading, queue, datetime, inspect, difflib, random, copy, statistics, ast
from datetime import datetime, timezone

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
        "opencv-python",             # For image processing
        "mss",             # For screen capture
        "selenium",
        "webdriver-manager",
        "flask_cors",
        "flask",            # For web server
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
        "system": "You are the Primary Controller in a modular, tool-enabled agent pipeline. You have at your disposal a set of discrete stages (context_analysis, intent_clarification, external_knowledge_retrieval, planning_summary, tool_chaining, assemble_prompt, final_inference, chain_of_thought, flow_health_check, etc.) and a rich toolkit (web search, file operations, screen/webcam capture, subtask management, chat history queries, etc.).\n- **Be aggressively curious**: constantly probe for missing information, but always **self-clarify** internally first before asking the user.\n- **Streamline data flow**: each stage consumes exactly its predecessor’s output (via ctx.stage_outputs) and must explicitly pass along what downstream stages will need.\n- **Dynamically assemble and reorder** stages on every turn based on the user’s intent, dependencies, and watchdog feedback.\n- **Health-check every run** (flow_health_check): compute a reliability score, log it, and if it dips below your threshold, propose concrete adjustments to the stage list as a Python list of stage names.\n- **Chain-of-thought**: after each response, append a concise 2–3 sentence reflection explaining how you arrived at your answer.\n- **Idle mull**: when you’ve been idle, generate a context-aware “what should I work on next?” internally, summarize your pending tasks, and only speak if truly necessary.\n- **Tool accountability**: when you call a tool—especially file writes or searches—confirm in your final_response that the operation succeeded and follow through end-to-end.\n- **Never get stuck in loops**: if you detect recursion or repetition, immediately propose stack tweaks (“run external_knowledge earlier”, “merge html_filtering+chunk_and_summarize”, etc.) and test them.\n- **Learn and evolve**: search online, use your tools, and keep optimizing your own pipeline—upgrade, test, and repeat without stopping.\nYour sole focus is to use these instruments to deliver the most accurate, context-rich, and self-improving assistance possible. Respond only with your reasoning and final answer—no filler, no apologies, just action and insight.",
        "conversation_id": "default_convo",
        "rms_threshold": 0.01,
        "tts_volume": 0.5,  # Volume for TTS playback
        "debug_audio_playback": False,
        "enable_noise_reduction": False,
        "consensus_threshold": 0.5,       # Similarity ratio required for consensus between models
        "speak_chain_of_thought": False,
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
import sounddevice as sd                       # audio playback / capture

from scipy.io.wavfile import write
from scipy.signal import butter, lfilter       # EQ enhancement
import numpy as np

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import whisper

from ollama import chat, embed
from dotenv import load_dotenv
from ollama._types import ResponseError

# ── Selenium stack
from selenium import webdriver
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# ── Parsers / utilities
from bs4 import BeautifulSoup
import html, textwrap                          # (html imported only once)
import os, shutil, random, platform, json
import inspect

# ── Wi-Fi control
import pywifi
from pywifi import const

# ── System + numerics
import psutil
from num2words import num2words
from itertools import count
from collections import defaultdict

# ── Audio / speech enhancement
import noisereduce as nr
import torch
from denoiser import pretrained

# ── Screen & vision
from PIL import ImageGrab
import cv2
import mss

load_dotenv()
brave_api_key = os.environ.get("BRAVE_API_KEY")
if brave_api_key:
    print(f"brave key loaded from .env: {brave_api_key}")
else:
    log_message("BRAVE_API_KEY not set; falling back to DuckDuckGo search", "WARNING")
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
    
log_message("Warming up Ollama primary model...", "PROCESS")
try:
    # a minimal dummy chat to force the primary model to load & cache weights
    dummy_messages = [{"role": "user", "content": "Hi"}]
    _ = chat(
        model=config["primary_model"],
        messages=dummy_messages,
        stream=False
    )
    log_message("Ollama primary model warm‑up complete.", "SUCCESS")

    log_message("Warming up Ollama secondary model...", "PROCESS")
    _ = chat(
        model=config["secondary_model"],
        messages=dummy_messages,
        stream=False
    )
    log_message("Ollama secondary model warm‑up complete.", "SUCCESS")

except Exception as e:
    log_message(f"Error during Ollama model warm‑up: {e}", "ERROR")
    # if it's an OOM or other recoverable error, you could clear cache and restart here
    try:
        torch.cuda.empty_cache()
        log_message("Cleared CUDA cache after Ollama warm‑up failure.", "INFO")
    except NameError:
        pass
    log_message("Restarting script to recover...", "INFO")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- Load Denoiser Model -----
log_message("Loading denoiser model (DNS64)...", "PROCESS")
denoiser_model = pretrained.dns64()   # Loads a pretrained DNS64 model from denoiser
log_message("Denoiser model loaded.", "SUCCESS")


# ─── OBSERVABILITY & FLASK IMPORTS ─────────────────────────────────────
import threading, time, json
from datetime import datetime

from flask import Flask, Response, render_template_string, request, jsonify
from flask_cors import CORS
from werkzeug.serving import make_server

# ----- Global Settings and Queues -----

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024
tts_queue = queue.Queue()
audio_queue = queue.Queue()
current_tts_process = None
tts_lock = threading.Lock()
log_message("Global settings and queues initialized.", "DEBUG")
# at the top of your module, after you load `config`:
TTS_DEBUG = config.get("tts_debug", False)

def _tts_log(msg: str, level: str = "DEBUG"):
    """
    Internal helper: only emit DEBUG logs if TTS_DEBUG is True.
    INFO/ERROR always go through.
    """
    if level.upper() == "DEBUG" and not TTS_DEBUG:
        return
    log_message(msg, level)

# At top of your module, after loading `config`:
TTS_DEBUG = config.get("tts_debug", False)

def _tts_debug(msg: str, category: str = "DEBUG"):
    """Only log if TTS_DEBUG is True."""
    if TTS_DEBUG:
        log_message(msg, category)

def flush_current_tts():
    _tts_debug("Flushing current TTS queue and processes...")
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
                _tts_debug("Killed current Piper process.")
            except Exception as e:
                log_message(f"Error killing Piper process: {e}", "ERROR")
            try:
                aproc.kill()
                _tts_debug("Killed current aplay process.")
            except Exception as e:
                log_message(f"Error killing aplay process: {e}", "ERROR")
            current_tts_process = None

def process_tts_request(text: str):
    volume    = config.get("tts_volume", 0.2)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_exe  = os.path.join(script_dir, "piper", "piper")
    onnx_json  = os.path.join(script_dir, config["onnx_json"])
    onnx_model = os.path.join(script_dir, config["onnx_model"])

    # Verify that necessary files exist
    for path, desc in [
        (piper_exe,  "Piper executable"),
        (onnx_json,  "ONNX JSON file"),
        (onnx_model, "ONNX model file"),
    ]:
        if not os.path.isfile(path):
            log_message(f"Error: {desc} not found at {path}", "ERROR")
            return

    payload     = {"text": text, "config": onnx_json, "model": onnx_model}
    payload_str = json.dumps(payload)

    # Build Piper command; include --debug only if TTS_DEBUG
    cmd_piper = [piper_exe, "-m", onnx_model, "--json-input", "--output_raw"]
    if TTS_DEBUG:
        cmd_piper.insert(3, "--debug")

    cmd_aplay = ["aplay", "--buffer-size=777", "-r", "22050", "-f", "S16_LE"]

    log_message(f"[TTS] Synthesizing: '{text}'", "INFO")
    try:
        with tts_lock:
            stderr_dest = subprocess.PIPE if TTS_DEBUG else subprocess.DEVNULL
            proc = subprocess.Popen(
                cmd_piper,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_dest
            )
            proc_aplay = subprocess.Popen(cmd_aplay, stdin=subprocess.PIPE)
            global current_tts_process
            current_tts_process = (proc, proc_aplay)
            _tts_debug("TTS processes started.")

        # send payload
        proc.stdin.write(payload_str.encode("utf-8"))
        proc.stdin.close()

        def adjust_volume(data: bytes, vol: float) -> bytes:
            samples   = np.frombuffer(data, dtype=np.int16)
            adjusted  = samples.astype(np.float32) * vol
            clipped   = np.clip(adjusted, -32768, 32767)
            return clipped.astype(np.int16).tobytes()

        # stream audio through aplay
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

        # only read & log Piper’s stderr if debugging
        if TTS_DEBUG:
            stderr_output = proc.stderr.read().decode("utf-8", errors="ignore").strip()
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
    # 1) Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # 2) Remove any code-fence markers or tool tags
    text = text.replace("```tool_output", "") \
               .replace("tool_call", "") \
               .replace("tool_output", "")

    # 3) Strip out all asterisks, backticks, and underscores
    text = re.sub(r'[*_`]', "", text)

    # 4) Trim whitespace
    return text.strip()


# ----- Consensus Transcription Helper -----
def consensus_whisper_transcribe_helper(audio_array, language="en", rms_threshold=0.01, consensus_threshold=0.8):
    rms = np.sqrt(np.mean(np.square(audio_array)))
    if rms < rms_threshold:
        #log_message("Audio RMS too low (RMS: {:.5f}). Skipping transcription.".format(rms), "WARNING")
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
        log_message("The Primary Model transcription is: " + transcription_base, "DEBUG")
        log_message("The Secondary Model transcription is: " + transcription_medium, "DEBUG")
        log_message("Consensus reached between Whisper models.", "SUCCESS")
        return transcription_base
    else:
        log_message("Primary Whisper model transcription: " + transcription_base, "DEBUG")
        log_message("Secondary Whisper model transcription: " + transcription_medium, "DEBUG")
        log_message("No consensus between base and medium models; ignoring transcription.", "WARNING")
        return ""

# ----- Transcription Validation Helper -----
def validate_transcription(text):
    """
    Accept any transcript that contains at least one letter or digit
    and at least one non‐whitespace token.
    """
    # must contain at least one alpha or digit character
    if not any(ch.isalpha() or ch.isdigit() for ch in text):
        log_message("Transcription validation failed: no alphanumeric characters.", "WARNING")
        return False

    # split on whitespace to count words/tokens
    words = text.strip().split()
    if len(words) < 1:
        log_message("Transcription validation failed: no words detected.", "WARNING")
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
        """
        Embed into a 1-D numpy array of shape (768,).
        """
        try:
            #log_message("Embedding text for context.", "PROCESS")
            response = embed(model="nomic-embed-text", input=text)
            vec = np.array(response["embeddings"], dtype=float)
            # ensure 1-D
            vec = vec.flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            #log_message("Text embedding computed and normalized.", "SUCCESS")
            return vec
        except Exception as e:
            log_message("Error during text embedding: " + str(e), "ERROR")
            return np.zeros(768, dtype=float)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine‐similarity between two 1-D vectors.
        """
        # flatten just in case
        v1 = np.array(vec1).flatten()
        v2 = np.array(vec2).flatten()
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            log_message("One of the vectors has zero norm in cosine similarity calculation.", "WARNING")
            return 0.0
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        #log_message("Cosine similarity computed.", "DEBUG")
        return sim


class Tools:
    _driver = None          # always present, even before first browser launch
    _poll   = 0.05                     
    _short  = 3
    
    @staticmethod
    def parse_tool_call(text: str) -> str | None:
        """
        Extract either:
          1) A fenced ```tool_code``` block
          2) An inline-backtick call: `func(arg1, arg2)`
          3) A bare call on its own line: func(arg1, arg2)

        Strips out Python‐style type annotations (e.g. filename: str="x")
        so that `file_info(filename: str = "F.gig")` becomes
        `file_info(filename="F.gig")`.
        """
        import re

        t = text.strip()

        # 0) Remove type annotations before the '='
        #    e.g. 'filename: str = "F.gig"' → 'filename = "F.gig"'
        t = re.sub(
            r'([A-Za-z_]\w*)\s*:\s*[A-Za-z_]\w*(?=\s*=)',
            r'\1',
            t
        )

        # 1) Normalize key: value → key=value (for older-style “key: 'val'”)
        t = re.sub(
            r'(\b[A-Za-z_]\w*\b)\s*:\s*'                 # key:
            r'("(?:[^"\\]|\\.)*"|'                      #   "quoted string"
            r'\'(?:[^\'\\]|\\.)*\'|'                    #   'quoted string'
            r'\d+|None)',                               #   integer or None
            r'\1=\2',
            t
        )

        # 2) Unwrap inline single-backticks if present
        if t.startswith("`") and t.endswith("`"):
            t = t[1:-1].strip()

        # 3) Prepare literal patterns
        LIT = r'(?:\"[^\"]*\"|\'[^\']*\'|\d+|None)'
        KV  = rf'(?:[A-Za-z_]\w*\s*=\s*)?{LIT}'

        # 4) Try fenced ```tool_code``` block
        fenced_re = re.compile(
            rf"```tool_code\s*"
            rf"([A-Za-z_]\w*\s*\(\s*"
            rf"(?:{KV}(?:\s*,\s*{KV})*)?\s*"
            rf"\)\s*)```",
            re.DOTALL
        )
        m = fenced_re.search(t)

        # 5) If not found, fall back to a bare call on its own line
        if not m:
            bare_re = re.compile(
                rf"^([A-Za-z_]\w*\s*\(\s*"
                rf"(?:{KV}(?:\s*,\s*{KV})*)?\s*"
                rf"\))\s*$",
                re.DOTALL
            )
            m = bare_re.match(t)

        if not m:
            log_message("Parsed tool call from text: None", "DEBUG")
            return None

        code = m.group(1).strip()
        log_message(f"Parsed tool call from text: {code!r}", "DEBUG")
        return code
    
    @staticmethod
    def add_subtask(parent_id: int, text: str) -> dict:
        """
        Create a new subtask under an existing task.
        If the parent_id does not exist (or is <= 0), the subtask will be top-level (parent=None).
        Returns the created subtask object.
        """
        import os, json

        path = os.path.join(WORKSPACE_DIR, "tasks.json")

        # 1) Safely load existing tasks (empty or invalid → [])
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tasks = json.load(f)
                if not isinstance(tasks, list):
                    tasks = []
            except (json.JSONDecodeError, ValueError):
                tasks = []
        else:
            tasks = []

        # 2) Determine whether the parent exists
        parent = None
        if isinstance(parent_id, int) and parent_id > 0:
            parent = next((t for t in tasks if t.get("id") == parent_id), None)

        # 3) Assign new ID
        new_id = max((t.get("id", 0) for t in tasks), default=0) + 1

        # 4) Build the subtask record
        sub = {
            "id":     new_id,
            "text":   text,
            "status": "pending",
            # if parent was found use its id, otherwise None (top-level)
            "parent": parent.get("id") if parent else None
        }

        # 5) Append and save
        tasks.append(sub)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2)
        except Exception as e:
            # if saving fails, return an error dict
            return {"error": f"Failed to save subtask: {e}"}

        return sub


    @staticmethod
    def list_subtasks(parent_id: int) -> list:
        """
        Return all subtasks for the given parent task.
        """
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        return [t for t in tasks if t.get("parent") == parent_id]

    @staticmethod
    def set_task_status(task_id: int, status: str) -> dict:
        """
        Set status = 'pending'|'in_progress'|'done' on a task or subtask.
        """
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return {"error": "No tasks yet."}
        tasks = json.loads(open(path).read())
        # find the task
        t = next((t for t in tasks if t.get("id") == task_id), None)
        if not t:
            return {"error": f"No such task {task_id}"}
        if status not in ("pending", "in_progress", "done"):
            return {"error": f"Invalid status {status}"}
        t["status"] = status
        # save list back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return t

    # internal helpers for persistence
    @staticmethod
    def _load_tasks_dict() -> dict:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if os.path.isfile(path):
            return json.load(open(path, "r"))
        return {}

    @staticmethod
    def _save_tasks_dict(tasks: dict) -> None:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2)

    
    _process_registry: dict[int, subprocess.Popen] = {}
    
    @staticmethod
    def add_task(text: str) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        new_id = max((t["id"] for t in tasks), default=0) + 1
        tasks.append({"id": new_id, "text": text})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return f"Task {new_id} added."

    @staticmethod
    def update_task(task_id: int, text: str) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return "No tasks yet."
        tasks = json.loads(open(path).read())
        for t in tasks:
            if t["id"] == task_id:
                t["text"] = text
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=2)
                return f"Task {task_id} updated."
        return f"No task with id={task_id}."

    @staticmethod
    def remove_task(task_id: int) -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        if not os.path.exists(path):
            return "No tasks yet."
        tasks = json.loads(open(path).read())
        new = [t for t in tasks if t["id"] != task_id]
        if len(new) == len(tasks):
            return f"No task with id={task_id}."
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new, f, indent=2)
        return f"Task {task_id} removed."

    @staticmethod
    def list_tasks() -> str:
        import json, os
        path = os.path.join(WORKSPACE_DIR, "tasks.json")
        tasks = json.loads(open(path).read()) if os.path.exists(path) else []
        return json.dumps(tasks)
    
    # ── Tools.run_python_snippet  (drop-in) ──────────────────────────────
    @staticmethod
    def run_python_snippet(
        code: str,
        *,
        stdin: str = "",
        timeout: int = 10,
        dedent: bool = True,
    ) -> dict:
        """
        Execute an arbitrary Python snippet in a fresh subprocess.

        Parameters
        ----------
        code : str
            The snippet to run.  It may already be wrapped in
            ```python …```   or   ```py …``` fences – these are stripped
            automatically so callers don’t have to worry.
        stdin : str
            Text piped to the child process’ STDIN.
        timeout : int
            Hard wall-clock limit (seconds).
        dedent : bool
            If True, run `textwrap.dedent()` on the snippet after stripping
            fences – makes copy-pasted indented code work.

        Returns
        -------
        dict
            {
            "stdout":      <captured STDOUT str>,
            "stderr":      <captured STDERR str>,
            "returncode":  <int>,
            }
            On failure an ``"error"`` key is present instead.
        """
        import re, subprocess, sys, tempfile, textwrap, os

        # 1) unwrap optional back-tick fences -----------------------------------
        fence_rx = re.compile(
            r"```(?:python|py)?\s*([\s\S]*?)\s*```",
            re.IGNORECASE,
        )
        m = fence_rx.search(code)
        if m:
            code = m.group(1)

        if dedent:
            code = textwrap.dedent(code)

        # 2) write to a temporary .py file --------------------------------------
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        # 3) run it in a clean subprocess ---------------------------------------
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Timed-out after {timeout}s"}

        except Exception as e:                       # any other unexpected error
            return {"error": str(e)}

        finally:                                     # always clean up the temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


    @staticmethod
    def run_tool_once(tool_call:str) -> dict:
        """
        Convenience: parse `tool_call` (e.g. "hello('world')")
        → execute the referenced *tool function* in-process
        → return { "output":…, "exception": str|None }.
        Useful for micro-benchmarks when launching a subprocess is overkill.
        """
        import ast, inspect, traceback
        m = ast.parse(tool_call.strip(), mode="eval")
        if not isinstance(m.body, ast.Call):
            return {"exception": "Not a function call"}
        name = m.body.func.id
        fn   = getattr(Tools, name, None)
        if not callable(fn):
            return {"exception": f"Unknown tool {name}"}
        # rebuild positional / kw from the AST
        args   = [ast.literal_eval(ast.get_source_segment(tool_call,a)) for a in m.body.args]
        kwargs = {kw.arg: ast.literal_eval(ast.get_source_segment(tool_call,kw.value))
                  for kw in m.body.keywords}
        try:
            out = fn(*args, **kwargs)
            return {"output": out, "exception": None}
        except Exception as e:
            return {"output": None, "exception": traceback.format_exc(limit=2)}


    @staticmethod
    def run_script(
        script_path: str,
        args: str = "",
        base_dir: str = WORKSPACE_DIR,
        capture_output: bool = False,
        window_title: str | None = None
    ) -> dict:
        """
        Launch or run a Python script.

        • If capture_output=False, opens in a new terminal (per OS) and registers PID.
        • If capture_output=True, runs synchronously, captures stdout/stderr/rc.
        Returns a dict:
          - on new terminal: {"pid": <pid>} or {"error": "..."}
          - on capture: {"stdout": "...", "stderr": "...", "returncode": <int>} or {"error": "..."}
        """
        import os, sys, platform, subprocess, shlex

        full_path = script_path if os.path.isabs(script_path) \
                    else os.path.join(base_dir, script_path)
        if not os.path.isfile(full_path):
            return {"error": f"Script not found at {full_path}"}

        cmd = [sys.executable, full_path] + (shlex.split(args) if args else [])
        system = platform.system()

        try:
            if capture_output:
                proc = subprocess.run(
                    cmd, cwd=base_dir, text=True, capture_output=True
                )
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode
                }

            # non-capture → new terminal window
            if system == "Windows":
                flags = subprocess.CREATE_NEW_CONSOLE
                if window_title:
                    title_cmd = ["cmd", "/c", f"title {window_title} &&"] + cmd
                    p = subprocess.Popen(title_cmd, creationflags=flags)
                else:
                    p = subprocess.Popen(cmd, creationflags=flags)

            elif system == "Darwin":
                osa = f'tell application "Terminal" to do script "{ " ".join(cmd) }"'
                p = subprocess.Popen(["osascript", "-e", osa])

            else:
                term = os.getenv("TERMINAL", "xterm")
                p = subprocess.Popen([term, "-hold", "-e"] + cmd, cwd=base_dir)

            pid = p.pid
            Tools._process_registry[pid] = p
            return {"pid": pid}

        except Exception as e:
            return {"error": str(e)}


    @staticmethod
    def stop_script(pid: int) -> dict:
        """
        Terminate a previously launched script by its PID.
        Returns {"stopped": pid} or {"error": "..."}.
        """
        proc = Tools._process_registry.get(pid)
        if not proc:
            return {"error": f"No managed process with PID {pid}"}
        try:
            proc.terminate()
            proc.wait(timeout=5)
            del Tools._process_registry[pid]
            return {"stopped": pid}
        except Exception as e:
            return {"error": str(e)}


    @staticmethod
    def script_status(pid: int) -> dict:
        """
        Check status of a managed PID.
        Returns {"running": pid} if alive, or {"exit_code": <int>} if done,
        or {"error": "..."} if unknown.
        """
        proc = Tools._process_registry.get(pid)
        if not proc:
            return {"error": f"No managed process with PID {pid}"}
        rc = proc.poll()
        if rc is None:
            return {"running": pid}
        # finished
        del Tools._process_registry[pid]
        return {"exit_code": rc}

    @staticmethod
    def explore_tools(action: str = "list", tool: str | None = None) -> str:
        """
        Tiny dispatcher so agents can:
            • explore_tools("list")          –> same as list_tools(detail=True)
            • explore_tools("source","foo")  –> same as get_tool_source("foo")
        """
        if action == "list":
            return Tools.list_tools(detail=True)
        if action == "source" and tool:
            return Tools.get_tool_source(tool)
        return "Usage: explore_tools('list')  or  explore_tools('source','tool_name')"


    # ───────────────────────────  SELF-IMPROVEMENT HELPERS  ───────────────────────────
    @staticmethod
    def list_tools(detail: bool = False) -> str:
        """
        Return JSON metadata for every callable tool currently on Tools.

        detail=False → ["name", ...]  
        detail=True  → [{"name","signature","doc"}, ...]
        """
        import inspect, json, textwrap

        tools: list = []
        for name, fn in inspect.getmembers(Tools, predicate=callable):
            if name.startswith("_"):
                continue
            if detail:
                sig = str(inspect.signature(fn))
                doc = textwrap.shorten(inspect.getdoc(fn) or "", width=140)
                tools.append({"name": name, "signature": sig, "doc": doc})
            else:
                tools.append(name)
        return json.dumps(tools, indent=2)

    @staticmethod
    def get_tool_source(tool_name: str) -> str:
        """
        Return the *source code* for `tool_name`, or an error string.
        """
        import inspect

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return f"Error: tool '{tool_name}' not found."
        try:
            return inspect.getsource(fn)
        except Exception as e:                     # pragma: no-cover
            return f"Error retrieving source: {e}"

    @staticmethod
    def create_tool(
        tool_name: str,
        code: str | None = None,
        *,
        description: str | None = None,           # ← tolerated, but ignored
        tool_call: str | None = None,             # ← dito (compat shim)
        overwrite: bool = False,
        auto_reload: bool = True,
    ) -> str:
        """
        Persist a new *external tool* and optionally hot-reload it.

        Parameters
        ----------
        tool_name    name of the Python file **and** the function inside it
        code         full `def tool_name(...):` **OR** None when you just want
                     to reserve the name (rare – normally provide real code)
        description  ignored → kept for backward compatibility with agents
        tool_call    ignored → ditto
        overwrite    allow clobbering an existing file
        auto_reload  immediately import the new module and attach the function
        """
        import os, re, textwrap

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        os.makedirs(ext_dir, exist_ok=True)
        file_path = os.path.join(ext_dir, f"{tool_name}.py")

        # ── guard rails ────────────────────────────────────────────────────
        if os.path.exists(file_path) and not overwrite:
            return f"Error: {file_path} already exists (use overwrite=True)."

        if code is None:
            return ("Error: `code` is required.  Pass the full function body "
                    "as a string under the `code=` parameter.")

        if not re.match(rf"^\s*def\s+{re.escape(tool_name)}\s*\(", code):
            return (f"Error: `code` must start with `def {tool_name}(` so that "
                    "the module exposes exactly one top-level function.")

        # ── write the module ───────────────────────────────────────────────
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                header = textwrap.dedent(f'''\
                    """
                    Auto-generated external tool  –  {tool_name}
                    Created via Tools.create_tool()
                    {('Description: ' + description) if description else ''}
                    """
                    ''')
                fh.write(header.rstrip() + "\n\n" + code.strip() + "\n")
        except Exception as e:
            return f"Error writing file: {e}"

        log_message(f"Created external tool {tool_name} at {file_path}", "SUCCESS")

        if auto_reload:
            Tools.reload_external_tools()

        return f"Tool '{tool_name}' created ✔"

    @staticmethod
    def list_external_tools() -> list[str]:
        """
        List *.py files currently present in *external_tools/*.
        """
        import os
        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        if not os.path.isdir(ext_dir):
            return []
        return sorted(f for f in os.listdir(ext_dir) if f.endswith(".py"))

    @staticmethod
    def remove_external_tool(tool_name: str) -> str:
        """
        Delete *external_tools/<tool_name>.py* and detach it from Tools.
        """
        import os, sys, importlib

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        path = os.path.join(ext_dir, f"{tool_name}.py")

        try:
            if os.path.isfile(path):
                os.remove(path)
                # also nuke any stale .pyc
                pyc = path + "c"
                if os.path.isfile(pyc):
                    os.remove(pyc)
            if hasattr(Tools, tool_name):
                delattr(Tools, tool_name)
            sys.modules.pop(f"external_tools.{tool_name}", None)
            log_message(f"External tool {tool_name} removed.", "INFO")
            return f"Tool '{tool_name}' removed."
        except Exception as e:                     # pragma: no-cover
            return f"Error removing tool: {e}"

    @staticmethod
    def reload_external_tools() -> str:
        """
        Detach any previously-loaded external tools, purge their modules, then
        re-import everything found in *external_tools/*.
        """
        import sys, inspect

        # 1️⃣  Detach current externals from Tools
        for name, fn in list(inspect.getmembers(Tools, predicate=callable)):
            if getattr(fn, "__module__", "").startswith("external_tools."):
                delattr(Tools, name)

        # 2️⃣  Purge from sys.modules so the next import is fresh
        for mod in list(sys.modules):
            if mod.startswith("external_tools."):
                sys.modules.pop(mod, None)

        # 3️⃣  Re-import
        Tools.load_external_tools()
        return "External tools reloaded."

    @staticmethod
    def test_tool(tool_name: str, test_cases: list[dict]) -> dict:
        """
        Quick-and-dirty unit-test harness.

        Each *test_case* dict may contain:  
          • args   : list – positional args  
          • kwargs : dict – keyword args  
          • expect : any  – expected value (optional)  
          • compare: "eq" | "contains" | "custom" (default "eq")  
          • custom : a **lambda** as string, evaluated only when compare="custom"
        """
        import traceback

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return {"error": f"Tool '{tool_name}' not found."}

        passed = failed = 0
        results = []

        for idx, case in enumerate(test_cases, 1):
            args   = case.get("args", []) or []
            kwargs = case.get("kwargs", {}) or {}
            expect = case.get("expect")
            mode   = case.get("compare", "eq")
            try:
                out = fn(*args, **kwargs)

                if mode == "eq":
                    ok = (out == expect)
                elif mode == "contains":
                    ok = str(expect) in str(out)
                elif mode == "custom":
                    ok = bool(eval(case.get("custom", "lambda *_: False"))(out))
                else:
                    ok = False

                passed += ok
                failed += (not ok)
                results.append({"case": idx, "passed": ok, "output": out})
            except Exception as e:
                failed += 1
                results.append({
                    "case": idx,
                    "passed": False,
                    "error": f"{e}",
                    "trace": traceback.format_exc(limit=2),
                })

        return {"tool": tool_name, "passed": passed, "failed": failed, "results": results}

    @staticmethod
    def evaluate_tool(
        tool_name: str,
        metric_code: str,
        sample_inputs: list[dict],
    ) -> dict:
        """
        Evaluate a tool with an arbitrary metric.

        • *metric_code* must be a **single-line λ**:  
          `lambda output, **inputs: <float between 0-1>` (higher = better).

        • *sample_inputs*  → list of **kwargs** dicts supplied to the tool.

        Returns {"scores": [...], "mean_score": <float>, "details": [...]}
        """
        import statistics, traceback

        fn = getattr(Tools, tool_name, None)
        if not fn or not callable(fn):
            return {"error": f"Tool '{tool_name}' not found."}

        try:
            scorer = eval(metric_code)
            assert callable(scorer)          # noqa: S101
        except Exception as e:
            return {"error": f"Invalid metric_code: {e}"}

        scores, details = [], []
        for inp in sample_inputs:
            try:
                out = fn(**inp)
                score = float(scorer(out, **inp))
            except Exception as e:
                score = 0.0
                details.append({"input": inp, "error": str(e),
                                "trace": traceback.format_exc(limit=1)})
            scores.append(score)

        mean = statistics.mean(scores) if scores else 0.0
        return {"scores": scores, "mean_score": mean, "details": details}

    # ───────────────────────────  EXTERNAL-TOOL LOADER  ───────────────────────────
    @staticmethod
    def load_external_tools() -> None:
        """
        Import every *.py* file in *external_tools/* and attach its **public**
        callables as @staticmethods on Tools.
        """
        import os, inspect, importlib.machinery, importlib.util

        ext_dir = os.path.join(os.path.dirname(__file__), "external_tools")
        os.makedirs(ext_dir, exist_ok=True)

        for fname in os.listdir(ext_dir):
            if fname.startswith("_") or not fname.endswith(".py"):
                continue

            mod_name = f"external_tools.{fname[:-3]}"
            path     = os.path.join(ext_dir, fname)

            loader = importlib.machinery.SourceFileLoader(mod_name, path)
            spec   = importlib.util.spec_from_loader(mod_name, loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)               # actual import

            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_") or hasattr(Tools, name):
                    continue
                setattr(Tools, name, staticmethod(fn))
                log_message(f"Loaded external tool: {name}()", "INFO")

        # keep the public manifest in sync for other agents
        Tools.discover_agent_stack()



    @staticmethod
    def create_file(filename: str, content: str, base_dir: str = WORKSPACE_DIR) -> str:
        """
        Create or overwrite a file under base_dir.
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Created file: {path}"
        except Exception as e:
            return f"Error creating file {path!r}: {e}"


    @staticmethod
    def append_file(filename: str, content: str, base_dir: str = WORKSPACE_DIR) -> str:
        """
        Append to or create a file under base_dir.
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended to file: {path}"
        except Exception as e:
            return f"Error appending to file {path!r}: {e}"


    @staticmethod
    def delete_file(filename: str, base_dir: str = WORKSPACE_DIR) -> str:
        """
        Delete a file under base_dir.
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.remove(path)
            return f"Deleted file: {path}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error deleting file {path!r}: {e}"


    @staticmethod
    def list_workspace(base_dir: str = WORKSPACE_DIR) -> str:
        """
        List entries under base_dir.
        """
        import os, json
        try:
            entries = os.listdir(base_dir)
            return json.dumps(entries)
        except Exception as e:
            return json.dumps({"error": str(e)})


    @staticmethod
    def find_files(pattern: str, path: str = WORKSPACE_DIR) -> str:
        """
        Recursively glob under path for files matching pattern.
        """
        import os, fnmatch, json
        matches = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    matches.append({"file": fname, "dir": root})
        return json.dumps(matches)


    @staticmethod
    def list_dir(path: str = WORKSPACE_DIR) -> str:
        """
        JSON list of entries at path.
        """
        import os, json
        try:
            return json.dumps(os.listdir(path))
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def list_files(path: str = WORKSPACE_DIR, pattern: str = "*") -> list:
        """Alias for find_files(pattern, path)."""
        return Tools.find_files(pattern, path)

    @staticmethod
    def read_files(path: str, *filenames: str) -> dict:
        """
        Read multiple files under `path` and return a dict { filename: content }.
        """
        out = {}
        for fn in filenames:
            out[fn] = Tools.read_file(fn, path)
        return out

    @staticmethod
    def read_file(filepath: str, base_dir: str | None = None) -> str:
        """
        Read and return the file’s contents.
        """
        import os
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path!r}: {e}"


    @staticmethod
    def write_file(filepath: str, content: str, base_dir: str | None = None) -> str:
        """
        Write content to filepath.
        """
        import os
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path!r}"
        except Exception as e:
            return f"Error writing {path!r}: {e}"


    @staticmethod
    def rename_file(old: str, new: str, base_dir: str = WORKSPACE_DIR) -> str:
        """
        Rename old→new under base_dir.
        """
        import os
        safe_old = os.path.normpath(old)
        safe_new = os.path.normpath(new)
        if safe_old.startswith("..") or safe_new.startswith(".."):
            return "Error: Invalid path"
        src = os.path.join(base_dir, safe_old)
        dst = os.path.join(base_dir, safe_new)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            return f"Renamed {safe_old} → {safe_new}"
        except Exception as e:
            return f"Error renaming file: {e}"


    @staticmethod
    def copy_file(src: str, dst: str, base_dir: str = WORKSPACE_DIR) -> str:
        """
        Copy src→dst under base_dir.
        """
        import os, shutil
        safe_src = os.path.normpath(src)
        safe_dst = os.path.normpath(dst)
        if safe_src.startswith("..") or safe_dst.startswith(".."):
            return "Error: Invalid path"
        src_path = os.path.join(base_dir, safe_src)
        dst_path = os.path.join(base_dir, safe_dst)
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
            return f"Copied {safe_src} → {safe_dst}"
        except Exception as e:
            return f"Error copying file: {e}"


    @staticmethod
    def file_exists(filename: str, base_dir: str = WORKSPACE_DIR) -> bool:
        """
        Check if filename exists under base_dir.
        """
        import os
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return False
        return os.path.exists(os.path.join(base_dir, safe))


    @staticmethod
    def file_info(filename: str, base_dir: str = WORKSPACE_DIR) -> dict:
        """
        Return metadata for filename.
        """
        import os
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return {"error": "Invalid path"}
        path = os.path.join(base_dir, safe)
        try:
            st = os.stat(path)
            return {"size": st.st_size, "modified": st.st_mtime}
        except Exception as e:
            return {"error": str(e)}


    @staticmethod
    def get_workspace_dir() -> str:
        """
        Absolute path of the workspace.
        """
        return WORKSPACE_DIR


    @staticmethod
    def get_cwd() -> str:
        """
        Return current working directory.
        """
        import os
        return os.getcwd()


    @staticmethod
    def discover_agent_stack() -> str:
        """
        Introspect Tools + Agent classes, write agent_stack.json.
        """
        import os, sys, json, inspect
        from datetime import datetime

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        tools = [
            name for name, fn in inspect.getmembers(Tools, predicate=callable)
            if not name.startswith("_")
        ]
        agents = [
            name for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
            if name.endswith("Manager") or name.endswith("ChatManager")
        ]

        default_stages = [
            "summary_request",
            "timeframe_history_query",
            "record_user_message",
            "context_analysis",
            "intent_clarification",
            "external_knowledge_retrieval",
            "memory_summarization",
            "planning_summary",
            "tool_self_improvement", 
            "tool_chaining",
            "assemble_prompt",
            "final_inference",
            "chain_of_thought",
            "notification_audit"
        ]

        config = {
            "tools":   tools,
            "agents":  agents,
            "stages":  default_stages,
            "updated": datetime.now().isoformat()
        }

        with open(stack_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return f"agent_stack.json created with {len(tools)} tools & {len(agents)} agents."
    @staticmethod
    def load_agent_stack() -> dict:
        """
        Return the contents of *agent_stack.json*.

        • If the file is missing or unreadable ⇒ call
          `Tools.discover_agent_stack()` to (re)create it, then reload.
        • If the JSON loads but is missing any of the canonical
          top-level keys (“tools”, “agents”, “stages”) ⇒ we *merge-patch*
          those keys from a fresh discovery result.
        """
        import os, json, copy

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        # helper: (re)generate the stack file on disk
        def _regen() -> dict:
            Tools.discover_agent_stack()          # writes the file
            with open(stack_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 1️⃣  Ensure the file exists – otherwise create from scratch
        if not os.path.isfile(stack_path):
            return _regen()

        # 2️⃣  Try to load it; regenerate on corruption
        try:
            with open(stack_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return _regen()

        # 3️⃣  Patch any missing keys without nuking user additions
        required_keys = {"tools", "agents", "stages"}
        if not required_keys.issubset(data.keys()):
            fresh = _regen()
            merged = copy.deepcopy(fresh)          # start with complete set
            merged.update(data)                    # user keys / overrides win
            # ensure required keys exist after merge
            for k in required_keys:
                merged.setdefault(k, fresh[k])
            # write the patched file back
            with open(stack_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            return merged

        return data


    @staticmethod
    def update_agent_stack(changes: dict, justification: str | None = None) -> str:
        """
        Merge `changes` into agent_stack.json, and if `justification` is provided,
        append an entry to `change_history` with timestamp, changes, and justification.
        """
        import os, json
        from datetime import datetime

        module_path = os.path.dirname(os.path.abspath(__file__))
        stack_path  = os.path.join(module_path, "agent_stack.json")

        # load existing stack (or discover a fresh one)
        config = Tools.load_agent_stack()

        # apply the user‐requested changes
        config.update(changes)
        now = datetime.now().isoformat()
        config["updated"] = now

        # record a change_history entry
        if justification:
            entry = {
                "timestamp":     now,
                "changes":       changes,
                "justification": justification
            }
            config.setdefault("change_history", []).append(entry)

        # write back to disk
        with open(stack_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return "agent_stack.json updated."
    
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    @staticmethod
    def _find_system_chromedriver() -> str | None:
        """
        Return a path to a *working* chromedriver executable on this machine
        (correct CPU arch + executable).  We try common locations first;
        each candidate must:
        1. Exist and be executable for the current user.
        2. Run `--version` without raising OSError (catches x86-64 vs arm64).
        If none pass, we return None so the caller can fall back to
        webdriver-manager or raise.
        """
        candidates: list[str | None] = [
            shutil.which("chromedriver"),                         # Anything already in PATH
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/snap/bin/chromium.chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
            "/opt/homebrew/bin/chromedriver",                     # macOS arm64
        ]

        for path in filter(None, candidates):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                try:
                    # If this fails with Exec format error, we skip it.
                    subprocess.run([path, "--version"],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
                    return path
                except Exception:
                    continue
        return None
    
    # ── internal helpers ──────────────────────────────────────────────
    @staticmethod
    def _wait_for_ready(drv, timeout=6):
        """Block until document.readyState == 'complete'."""
        WebDriverWait(drv, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    @staticmethod
    def _first_present(drv, selectors: list[str], timeout=4):
        """
        Return the first WebElement present+enabled out of a list of CSS selectors.
        Returns None if none arrive within `timeout`.
        """
        for sel in selectors:
            try:
                return WebDriverWait(drv, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
            except TimeoutException:
                continue
        return None

    @staticmethod
    def _visible_and_enabled(locator):
        """condition: element is displayed *and* not disabled."""
        def _cond(drv):
            try:
                el = drv.find_element(*locator)
                return el.is_displayed() and el.is_enabled()
            except Exception:
                return False
        return _cond

    # ── session bootstrap ────────────────────────────────────────────
    @staticmethod
    def open_browser(headless: bool = False, force_new: bool = False) -> str:
        """
        Launch Chrome/Chromium on x86-64 **and** ARM.
        Tries Selenium-Manager → system chromedriver → webdriver-manager.
        """
        import os, platform, random, shutil
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager

        if force_new and getattr(Tools, "_driver", None):
            try:
                Tools._driver.quit()
            except Exception:
                pass
            Tools._driver = None

        if getattr(Tools, "_driver", None):
            return "Browser already open"

        arch = platform.machine().lower()
        log_message(f"[open_browser] Detected CPU arch = {arch}", "DEBUG")

        chrome_bin = (
            os.getenv("CHROME_BIN")
            or shutil.which("google-chrome")
            or shutil.which("chromium")
            or "/usr/bin/chromium"
        )

        opts = Options()
        opts.binary_location = chrome_bin
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--remote-allow-origins=*")
        opts.add_argument(f"--remote-debugging-port={random.randint(45000, 65000)}")

        # 1️⃣  Selenium-Manager (Chrome ≥115 ships its own driver)
        try:
            log_message("[open_browser] Trying Selenium-Manager auto-driver…", "DEBUG")
            Tools._driver = webdriver.Chrome(options=opts)
            log_message("[open_browser] Launched via Selenium-Manager.", "SUCCESS")
            return "Browser launched (selenium-manager)"
        except WebDriverException as e_mgr:
            log_message(f"[open_browser] Selenium-Manager failed: {e_mgr}", "WARNING")

        # 2️⃣  system-wide chromedriver?
        sys_drv = Tools._find_system_chromedriver() if hasattr(Tools, "_find_system_chromedriver") else None
        if sys_drv:
            try:
                log_message(f"[open_browser] Trying system chromedriver at {sys_drv}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(sys_drv), options=opts)
                log_message("[open_browser] Launched via system chromedriver.", "SUCCESS")
                return "Browser launched (system chromedriver)"
            except WebDriverException as e_sys:
                log_message(f"[open_browser] System chromedriver failed: {e_sys}", "WARNING")

        try:
            # ── figure out which driver we need ───────────────────────────────
            # 1) detect CPU architecture
            arch_alias = {
                "aarch64": "arm64",
                "arm64":   "arm64",
                "armv8l":  "arm64",
                "armv7l":  "arm",            # 32-bit
            }
            wdm_arch = arch_alias.get(arch)          # None on x86-64

            # 2) detect the *installed* Chrome/Chromium major version
            try:
                raw_ver = (
                    subprocess.check_output([chrome_bin, "--version"])
                    .decode()
                    .strip()
                )                       # e.g. "Chromium 136.0.7103.92"
                browser_major = raw_ver.split()[1].split(".")[0]
            except Exception:
                browser_major = ""      # let webdriver-manager pick “latest”

            # 3) ask webdriver-manager for the matching driver
            from webdriver_manager.chrome import ChromeDriverManager

            log_message(
                f"[open_browser] Requesting ChromeDriver {browser_major or 'latest'} "
                f"for arch={wdm_arch or 'x86_64'}",
                "DEBUG",
            )

            drv_path = ChromeDriverManager(
                driver_version=browser_major or "latest",
                arch=wdm_arch,
                os_type="linux",
            ).install()

            log_message(f"[open_browser] webdriver-manager driver at {drv_path}", "DEBUG")
            Tools._driver = webdriver.Chrome(service=Service(drv_path), options=opts)
            log_message("[open_browser] Launched via webdriver-manager.", "SUCCESS")
            return "Browser launched (webdriver-manager)"

        except Exception as e_wdm:
            log_message(f"[open_browser] webdriver-manager failed: {e_wdm}", "ERROR")
            raise RuntimeError(
                "All driver acquisition strategies failed. "
                "Install a matching chromedriver and set PATH or CHROME_BIN."
            ) from e_wdm


    @staticmethod
    def close_browser() -> str:
        if Tools._driver:
            try:
                Tools._driver.quit()
                log_message("[close_browser] Browser closed.", "DEBUG")
            except Exception:
                pass
            Tools._driver = None
            return "Browser closed"
        return "No browser to close"

    # ── low-level DOM helpers ────────────────────────────────────────
    @staticmethod
    def navigate(url: str) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        log_message(f"[navigate] → {url}", "DEBUG")
        Tools._driver.get(url)
        return f"Navigated to {url}"

    @staticmethod
    def click(selector: str, timeout: int = 8) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
            focused = drv.execute_script("return document.activeElement === arguments[0];", el)
            log_message(f"[click] {selector} clicked (focused={focused})", "DEBUG")
            return f"Clicked {selector}"
        except Exception as e:
            log_message(f"[click] Error clicking {selector}: {e}", "ERROR")
            return f"Error clicking {selector}: {e}"

    @staticmethod
    def input(selector: str, text: str, timeout: int = 8) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.clear()
            el.send_keys(text + Keys.RETURN)
            log_message(f"[input] Sent {text!r} to {selector}", "DEBUG")
            return f"Sent {text!r} to {selector}"
        except Exception as e:
            log_message(f"[input] Error typing into {selector}: {e}", "ERROR")
            return f"Error typing into {selector}: {e}"

    @staticmethod
    def get_html() -> str:
        if not Tools._driver:
            return "Error: browser not open"
        return Tools._driver.page_source

    @staticmethod
    def screenshot(filename: str = "screenshot.png") -> str:
        if not Tools._driver:
            return "Error: browser not open"
        Tools._driver.save_screenshot(filename)
        return filename

    # ───────────────────────────  DUCKDUCKGO SEARCH  ──────────────────────────
    @staticmethod
    def duckduckgo_search(        # ← new canonical name
        topic: str,
        num_results: int = 5,
        wait_sec: int = 1,
        deep_scrape: bool = True,
    ) -> list:
        """
        Ultra-quick DuckDuckGo search (event-driven, JS injection).
        • Opens the first *num_results* links in separate tabs and deep-scrapes each.
        • Returns: title, url, snippet, summary, and full page HTML (`content`).
        • Never blocks on long sleeps – every wait is event-based and aggressively-polled.
        """
        import html, traceback
        from bs4 import BeautifulSoup
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        log_message(f"[duckduckgo_search] ▶ {topic!r}", "INFO")

        # fresh browser session ─────────────────────────────────────────────
        Tools.close_browser()
        Tools.open_browser(headless=False, force_new=True)
        drv   = Tools._driver
        wait  = WebDriverWait(drv, wait_sec, poll_frequency=0.10)   # very snappy polling
        results = []

        try:
            # 1️⃣  Home page ------------------------------------------------------------------
            drv.get("https://duckduckgo.com/")
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            log_message("[duckduckgo_search] Home page ready.", "DEBUG")

            # Cookie banner (if any) ----------------------------------------------------------
            try:
                btn = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler")
                ))
                btn.click()
                log_message("[duckduckgo_search] Cookie banner dismissed.", "DEBUG")
            except TimeoutException:
                pass

            # 2️⃣  Search box -----------------------------------------------------------------
            selectors = (
                "input#search_form_input_homepage",
                "input#searchbox_input",
                "input[name='q']",
            )
            box = next((drv.find_element(By.CSS_SELECTOR, sel)
                        for sel in selectors if drv.find_elements(By.CSS_SELECTOR, sel)),
                    None)
            if not box:
                raise RuntimeError("Search box not found on DuckDuckGo!")

            # 3️⃣  Inject query & submit (instant) --------------------------------------------
            drv.execute_script(
                "arguments[0].value = arguments[1];"
                "arguments[0].dispatchEvent(new Event('input'));"
                "arguments[0].form.submit();",
                box, topic
            )
            log_message("[duckduckgo_search] Query submitted via JS.", "DEBUG")

            # 4️⃣  Wait until results appear --------------------------------------------------
            try:
                wait.until(lambda d: "?q=" in d.current_url)
                wait.until(lambda d:
                    d.find_elements(By.CSS_SELECTOR,
                                    "#links .result, #links [data-nr]")
                )
                log_message("[duckduckgo_search] Results detected.", "DEBUG")
            except TimeoutException:
                log_message("[duckduckgo_search] Timeout waiting for results.", "WARNING")

            # 5️⃣  Collect anchors for the first N results ------------------------------------
            anchors = drv.find_elements(
                By.CSS_SELECTOR,
                "a.result__a, a[data-testid='result-title-a']"
            )[:num_results]

            if not anchors:
                log_message("[duckduckgo_search] No anchors found — layout change?", "ERROR")

            main_handle = drv.current_window_handle

            for a in anchors:
                try:
                    href  = a.get_attribute("href")
                    title = a.text.strip() or html.unescape(
                        drv.execute_script("return arguments[0].innerText;", a)
                    )

                    # locate snippet in the same result container
                    try:
                        parent   = a.find_element(By.XPATH, "./ancestor::*[contains(@class,'result')][1]")
                        sn_el    = parent.find_element(By.CSS_SELECTOR,
                                ".result__snippet, span[data-testid='result-snippet']")
                        snippet  = sn_el.text.strip()
                    except NoSuchElementException:
                        snippet = ""

                    summary       = snippet
                    page_content  = ""

                    # 6️⃣  Deep-scrape in **new tab** ----------------------------------------
                    if deep_scrape and href:
                        drv.switch_to.new_window("tab")          # Selenium-4 native tab
                        drv.get(href)
                        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

                        # First try our lightweight requests helper
                        page_content = Tools.bs4_scrape(href)
                        if page_content.startswith("Error"):
                            # fallback to Selenium DOM if remote site blocks requests
                            page_content = drv.page_source

                        pg    = BeautifulSoup(page_content, "html5lib")
                        meta  = pg.find("meta", attrs={"name": "description"})
                        p_tag = pg.find("p")

                        if meta and meta.get("content"):
                            summary = meta["content"].strip()
                        elif p_tag:
                            summary = p_tag.get_text(strip=True)

                        drv.close()                 # close the tab
                        drv.switch_to.window(main_handle)

                    # record ---------------------------------------------------------------
                    if page_content:
                        clean_content = BeautifulSoup(
                            page_content,
                            "html.parser"
                        ).get_text(separator=" ", strip=True)
                    else:
                        clean_content = ""

                    results.append({
                        "title":   title,
                        "url":     href,
                        "snippet": snippet,
                        "summary": summary,
                        "content": clean_content,
                    })

                    if len(results) >= num_results:
                        break

                except Exception as ex:
                    log_message(f"[duckduckgo_search] Error processing a result: {ex}", "WARNING")
                    continue

        except Exception as e:
            log_message(f"[duckduckgo_search] Fatal error: {e}\n{traceback.format_exc()}", "ERROR")
        finally:
            Tools.close_browser()

        log_message(f"[duckduckgo_search] Collected {len(results)} results.", "SUCCESS")
        return results


    # ▸ optional backward-compat alias
    ddg_search = duckduckgo_search



    # ────────────────────────────────────────────────────────────────
    #  selenium_extract_summary  (robust version)
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def selenium_extract_summary(url: str, wait_sec: int = 8) -> str:
        """
        Fast two-stage page summariser:

        1. Try a lightweight `Tools.bs4_scrape()` (no browser).
           If it yields HTML, grab <meta name="description"> or first <p>.
        2. If bs4-scrape fails, fall back to *headless* Selenium.
           Uses same driver logic.
        3. Always cleans up the browser session.
        """
        from bs4 import BeautifulSoup
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        # 1) quick request-based scrape
        html_doc = Tools.bs4_scrape(url)
        if html_doc and not html_doc.startswith("Error"):
            pg = BeautifulSoup(html_doc, "html5lib")
            m = pg.find("meta", attrs={"name": "description"})
            if m and m.get("content"):
                return m["content"].strip()
            p = pg.find("p")
            if p:
                return p.get_text(strip=True)

        # 2) fall back to headless Selenium
        try:
            Tools.open_browser(headless=True)
            drv = Tools._driver
            wait = WebDriverWait(drv, wait_sec)
            drv.get(url)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            pg2 = BeautifulSoup(drv.page_source, "html5lib")
            m2 = pg2.find("meta", attrs={"name": "description"})
            if m2 and m2.get("content"):
                return m2["content"].strip()
            p2 = pg2.find("p")
            if p2:
                return p2.get_text(strip=True)
            return ""
        finally:
            Tools.close_browser()

    # ────────────────────────────────────────────────────────────────
    #  summarize_local_search  (minor tweak → deep_scrape flag)
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def summarize_local_search(topic: str, top_n: int = 3, deep: bool = False) -> str:
        """
        1) Call ddg_search() to get top_n results (optionally deep-scraped)
        2) Return bullet list “1. Title — summary”
        """
        try:
            entries = Tools.ddg_search(topic, num_results=top_n, deep_scrape=deep)
        except Exception as e:
            return f"Search error: {e}"
        if not entries:
            return "No results found."
        return "\n".join(
            f"{i}. {e['title']} — {e['summary'] or '(no summary)'}"
            for i, e in enumerate(entries, 1)
        )

    
    @staticmethod
    def get_chat_history(arg1=None, arg2=None) -> str:
        """
        get_chat_history("today"/"yesterday"/"last N days") -> all messages in that window
        get_chat_history(n) -> last n messages
        get_chat_history(n, "2 days") -> last n messages from the last 2 days
        get_chat_history("query", n) -> top-n by relevance to 'query'

        Also merges on-disk chat_sessions/*/session.txt entries with in-memory history,
        and caps any semantic similarity search to the 100 most recent messages.
        """
        import os, re, json
        from datetime import datetime, timedelta

        # 1) Load on-disk session logs (one session.txt per dated folder)
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        sessions_dir = os.path.join(script_dir, "chat_sessions")
        disk_entries = []
        if os.path.isdir(sessions_dir):
            for sub in os.listdir(sessions_dir):
                session_file = os.path.join(sessions_dir, sub, "session.txt")
                if os.path.isfile(session_file):
                    try:
                        with open(session_file, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    e = json.loads(line)
                                    if all(k in e for k in ("timestamp", "role", "content")):
                                        disk_entries.append(e)
                                except:
                                    continue
                    except:
                        continue

        # 2) Grab in-memory entries
        hm = Tools._history_manager
        mem_entries = hm.history if hm else []

        # 3) Merge (disk first so in-memory can override if identical timestamps)
        all_entries = disk_entries + mem_entries

        # --- 1) TIMEFRAME-ONLY MODE ---
        if isinstance(arg1, str):
            period = arg1.lower().strip()
            now    = datetime.now()
            today  = now.date()
            start = end = None

            if period == "today":
                start = datetime.combine(today, datetime.min.time())
                end   = start + timedelta(days=1)
            elif period == "yesterday":
                start = datetime.combine(today - timedelta(days=1), datetime.min.time())
                end   = datetime.combine(today, datetime.min.time())
            else:
                m = re.match(r'last\s+(\d+)\s+days?', period)
                if m:
                    days = int(m.group(1))
                    start = datetime.combine(today - timedelta(days=days), datetime.min.time())
                    end   = datetime.combine(today + timedelta(days=1), datetime.min.time())

            if start is not None:
                results = []
                for e in all_entries:
                    try:
                        ts = datetime.fromisoformat(e["timestamp"])
                    except:
                        continue
                    if start <= ts < end:
                        results.append({
                            "timestamp": e["timestamp"],
                            "role":      e["role"],
                            "content":   e["content"]
                        })
                return json.dumps({"results": results}, indent=2)

        # --- 2) NUMERIC (+ optional relative period) MODE ---
        top_n    = None
        since_dt = None
        query    = None

        # If arg1 is a number → last N messages (with optional period arg2)
        if isinstance(arg1, (int, str)) and re.match(r'^\d+$', str(arg1)):
            top_n = int(arg1)
            if arg2:
                m = re.match(r'(\d+)\s*(day|hour|minute|week)s?', str(arg2), re.IGNORECASE)
                if m:
                    val, unit = int(m.group(1)), m.group(2).lower()
                    now = datetime.now()
                    if unit.startswith("day"):
                        since_dt = now - timedelta(days=val)
                    elif unit.startswith("hour"):
                        since_dt = now - timedelta(hours=val)
                    elif unit.startswith("minute"):
                        since_dt = now - timedelta(minutes=val)
                    elif unit.startswith("week"):
                        since_dt = now - timedelta(weeks=val)
                else:
                    try:
                        since_dt = datetime.fromisoformat(arg2)
                    except:
                        since_dt = None
        else:
            # Otherwise interpret arg1 as a query string
            if arg1 is not None:
                query = str(arg1)
                if arg2 and re.match(r'^\d+$', str(arg2)):
                    top_n = int(arg2)
            if top_n is None:
                top_n = 5

        # Default if still unset
        if top_n is None:
            top_n = 5

        # 3) Filter by since_dt
        filtered = []
        for e in all_entries:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
            except:
                continue
            if since_dt and ts < since_dt:
                continue
            filtered.append(e)

        # 4) Build results list
        scored = []
        if query:
            # cap to last 100 for embedding efficiency
            candidates = filtered[-100:]
            q_vec = Utils.embed_text(query)
            for e in candidates:
                text  = e.get("content", "")
                score = 1.0 if query.lower() in text.lower() else 0.0
                v     = Utils.embed_text(text)
                score += Utils.cosine_similarity(q_vec, v)
                scored.append((score, e))
            scored.sort(key=lambda x: x[0], reverse=True)
        else:
            # reverse-chronological for no-query
            for e in reversed(filtered):
                scored.append((0.0, e))

        # 5) Take top_n and format
        top = scored[:top_n]
        out = []
        for score, e in top:
            out.append({
                "timestamp": e.get("timestamp"),
                "role":      e.get("role"),
                "content":   e.get("content"),
                "score":     round(score, 3)
            })

        return json.dumps({"results": out}, indent=2)

    @staticmethod
    def get_current_time():
        from datetime import datetime
        # Grab current local time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # Log the exact timestamp we’re returning
        log_message(f"Current time retrieved: {current_time}", "DEBUG")
        return current_time
        
    @staticmethod
    def capture_screen_and_annotate():
        """
        Capture the primary monitor’s screen using mss, save it with a timestamp,
        and return a JSON string containing:
          - 'file': the saved file path
          - 'prompt': an instruction for the model to describe the screenshot.

        Usage:
            ```tool_code
            capture_screen_and_annotate()
            ```
        """

        # 1) Build output path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"screen_{ts}.png"
        path       = os.path.join(script_dir, filename)

        # 2) Capture with mss
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                img     = sct.grab(monitor)
                mss.tools.to_png(img.rgb, img.size, output=path)
            log_message(f"Screen captured and saved to {path}", "SUCCESS")
        except Exception as e:
            log_message(f"Error capturing screen: {e}", "ERROR")
            return json.dumps({"error": str(e)})

        # 3) Return the file path plus a prompt
        return json.dumps({
            "file":   path,
            "prompt": f"Please describe what you see in the screenshot, considering this is a screenshot which is of the computer that you reside on, and activity on the screen may be critical to answering questions, be as verbose as possible and describe any text or images present at '{path}'."
        })


    @staticmethod
    def capture_webcam_and_annotate():
        """
        Capture one frame from the default webcam using OpenCV,
        save it with a timestamp, and return a JSON string containing:
          - 'file': the saved file path
          - 'prompt': an instruction for the model to describe the image.

        Usage:
            ```tool_code
            capture_webcam_and_annotate()
            ```
        """

        # 1) Open the default camera
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cam.isOpened():
            log_message("Webcam not accessible via cv2.VideoCapture", "ERROR")
            return json.dumps({"error": "Webcam not accessible."})

        # 2) Grab a frame
        ret, frame = cam.read()
        cam.release()
        if not ret:
            log_message("Failed to read frame from webcam", "ERROR")
            return json.dumps({"error": "Failed to capture frame."})

        # 3) Build output path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"webcam_{ts}.png"
        path       = os.path.join(script_dir, filename)

        # 4) Save as PNG
        try:
            cv2.imwrite(path, frame)
            log_message(f"Webcam frame saved to {path}", "SUCCESS")
        except Exception as e:
            log_message(f"Error saving webcam frame: {e}", "ERROR")
            return json.dumps({"error": str(e)})

        # 5) Return the file path plus a prompt
        return json.dumps({
            "file":   path,
            "prompt": f"Please describe what you see in the image in great detail, considering the context that this image is coming from a webcam attached to the computer you reside on at '{path}'."
        })


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
    

    # ───────────────────────────  WEB SEARCH HELPERS  ─────────────────────────
    @staticmethod
    def search_internet(topic: str) -> str:
        """
        Search the web.

        1. If BRAVE_API_KEY is set → use Brave Search API.
        2. Otherwise *or* upon any error → fall back to Tools.ddg_search().
        3. Always return **JSON text** shaped as:
           { "engine": "brave" | "duckduckgo",
             "data":   <raw-Brave-JSON or list-of-DDG-dicts> }
        """
        import json, os, requests, traceback

        api_key = os.environ.get("BRAVE_API_KEY", "")
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers  = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "x-subscription-token": api_key
        }
        params   = {"q": topic, "count": 10}

        # ── 1️⃣  Brave Search if possible ───────────────────────────────────
        if api_key:
            try:
                log_message(f"Performing Brave search for topic: {topic}", "PROCESS")
                resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
                if resp.status_code == 200:
                    log_message("Brave search successful.", "SUCCESS")
                    return json.dumps({"engine": "brave", "data": resp.json()})
                else:
                    log_message(f"Brave search error {resp.status_code}", "ERROR")
            except Exception as e:
                log_message(f"Brave search exception: {e}\n{traceback.format_exc()}", "ERROR")

        # ── 2️⃣  Fallback → DuckDuckGo via Selenium helper ───────────────────
        log_message("Falling back to DuckDuckGo search.", "WARNING")
        try:
            ddg_results = Tools.ddg_search(topic, num_results=10, deep_scrape=False)
            return json.dumps({"engine": "duckduckgo", "data": ddg_results})
        except Exception as e:
            log_message(f"DuckDuckGo fallback failed: {e}", "ERROR")
            return json.dumps({"engine": "error", "data": str(e)})

    # ───────────────────────────  SEARCH → SUMMARY  ───────────────────────────
    @staticmethod
    def summarize_search(topic: str, top_n: int = 3) -> str:
        """
        1) Call search_internet() – Brave when available, DDG otherwise.
        2) Take the first `top_n` results.
        3) Scrape each URL with bs4_scrape().
        4) Ask secondary_agent_tool() for a 2-3 sentence summary.
        5) Return a neat bullet-list.
        """
        import json, traceback

        try:
            raw = Tools.search_internet(topic)
            payload = json.loads(raw or "{}")
            engine  = payload.get("engine")
            data    = payload.get("data", {})
        except Exception as e:
            return f"Error parsing search results: {e}"

        # ── normalise to a list of results with .url / .title ──────────────
        if engine == "brave":
            web_results = data.get("web", {}).get("results", [])[:top_n]
        elif engine == "duckduckgo":
            web_results = data[:top_n] if isinstance(data, list) else []
        else:  # some error earlier
            return f"No results – engine reported '{engine}'."

        summaries = []
        for idx, r in enumerate(web_results, start=1):
            url   = r.get("url") or r.get("url", "")
            title = r.get("title") or r.get("title", url)
            if not url:
                continue
            try:
                html_doc = Tools.bs4_scrape(url)
                snippet  = html_doc[:2000].replace("\n", " ") if isinstance(html_doc, str) else ""
                prompt   = (
                    f"Here is the beginning of the page at {url}:\n\n"
                    f"{snippet}\n\n"
                    "Please give me a 2-3 sentence summary of the key points."
                )
                summary = Tools.secondary_agent_tool(prompt, temperature=0.3)
            except Exception as ex:
                log_message(f"Summarisation failed for {url}: {ex}", "ERROR")
                summary = "Failed to scrape or summarise that page."
            summaries.append(f"{idx}. {title} — {summary.strip()}")

        if not summaries:
            return "No web results found."
        return "\n".join(summaries)


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
        """
        Invoke the secondary LLM (for tool processing) with a user prompt
        and an optional temperature. Returns the assistant’s message content
        as a plain string (or an error message on failure).
        """
        secondary_model = config.get("secondary_model")
        payload = {
            "model": secondary_model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            log_message(f"Calling secondary agent tool with prompt={prompt!r} temperature={temperature}", "PROCESS")
            response = chat(
                model=secondary_model,
                messages=payload["messages"],
                stream=False
            )
            # Ollama’s chat(...) returns {"message": {"content": ...}}
            content = response.get("message", {}).get("content")
            if content is None:
                log_message("Secondary agent tool returned no content field", "WARNING")
                return ""
            log_message("Secondary agent tool responded.", "SUCCESS")
            return content
        except Exception as e:
            log_message(f"Error in secondary agent tool: {e}", "ERROR")
            return f"Error in secondary agent: {e}"


class StopStreamException(Exception):
    """Raised to abort a runaway or hallucinating stream."""
    pass


class WatchdogManager:
    # ─── Replacement ───
    def __init__(self, quality_model: str, halluc_model: str = None,
                 quality_threshold: float = 7.5, halluc_threshold: float = 0.5,
                 max_retries: int = 1):
        """
        quality_model: LLM to rate overall run quality (0–10)
        halluc_model: LLM to classify hallucination (if None, uses quality_model)
        quality_threshold: minimum acceptable run score
        halluc_threshold: minimum hallucination probability to flag
        max_retries: how many run‐retries before giving up
        """
        self.quality_model     = quality_model
        self.halluc_model      = halluc_model or quality_model
        self.quality_threshold = quality_threshold
        self.halluc_threshold  = halluc_threshold
        self.max_retries       = max_retries

        # New controls to avoid flagging tiny or too‐frequent snippets:
        self.min_snippet_tokens     = 3    # ignore snippets shorter than this
        self.snippet_check_frequency = 2   # only check every Nth snippet


    def evaluate_run(self, ctx) -> tuple[float, list[str]]:
        """
        After the full pipeline, returns (score, suggestions).
        """
        report = {
            "stages":         ctx.stage_counts,
            "ctx_txt":        ctx.ctx_txt[:2000],
            "final_response": ctx.final_response
        }
        system = (
            "You are a Pipeline Watchdog.  Given a JSON report of a pipeline run "
            "(which stages ran, their counts, the accumulated ctx_txt, and the "
            "final_response),\n"
            " 1) rate overall run quality 0–10 (higher is better)\n"
            " 2) if below threshold, propose 1–3 concrete adjustments to the stage list\n"
            "Output a JSON object: {\"score\": <float>, \"suggestions\": [<string>,…]}."
        )
        user = f"Here is the run report:\n\n```json\n{json.dumps(report, indent=2)}\n```"
        resp = chat(
            model=self.quality_model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            stream=False
        )["message"]["content"]
        try:
            out = json.loads(resp)
            return float(out.get("score", 0.0)), out.get("suggestions", [])
        except Exception:
            return 0.0, []

    # ─── Replacement ───
    def monitor_stream(self, stream_generator, ctx):
        executor       = ThreadPoolExecutor(max_workers=2)
        pending_checks = []
        snippet_count  = 0

        def schedule_check(snippet):
            return executor.submit(self._detect_hallucination, snippet, ctx)

        try:
            for chunk, done in stream_generator:
                # 1) Always pass tokens through immediately
                yield chunk, done

                snippet = chunk.strip()
                if not snippet:
                    continue

                # 2) Throttle: only check every Nth non-empty snippet
                snippet_count += 1
                if snippet_count % self.snippet_check_frequency != 0:
                    continue

                # 3) Ignore very short snippets (e.g. “My”, “Sorry”)
                if len(snippet.split()) < self.min_snippet_tokens:
                    continue

                # 4) Schedule the hallucination check
                pending_checks.append(schedule_check(snippet))

                # 5) Process any completed checks
                for f in pending_checks[:]:
                    if f.done():
                        score = f.result()
                        pending_checks.remove(f)
                        if score >= self.halluc_threshold:
                            log_message(
                                f"[Watchdog] Hallucination flagged "
                                f"(score={score:.2f}) on snippet {snippet!r}; continuing without abort.",
                                "WARNING"
                            )
                            # note: no StopStreamException here
        finally:
            executor.shutdown(wait=False)


    # ─── Replacement ───
    def _detect_hallucination(self, snippet: str, ctx) -> float:
        """
        Returns a hallucination score 0–1 for this snippet, using self.halluc_model.
        We skip very short or trivial snippets entirely.
        """
        # 1) Skip trivial snippets
        if len(snippet.split()) < self.min_snippet_tokens:
            return 0.0

        system = (
            "You are a Hallucination Detector.  Given a short snippet of output and "
            "the conversation context, rate how likely it is that this snippet is "
            "fabricated or unsupported by the context, on a scale from 0.0 to 1.0."
        )
        user = (
            f"Context so far:\n{ctx.ctx_txt[-1000:]}\n\n"
            f"Snippet to evaluate:\n\"\"\"\n{snippet}\n\"\"\"\n\n"
            "Respond with a single float between 0.0 (no hallucination) and 1.0 "
            "(definite hallucination)."
        )
        resp = chat(
            model=self.halluc_model,
            messages=[{"role":"system","content":system},
                      {"role":"user",  "content":user}],
            stream=False
        )["message"]["content"].strip()
        try:
            return float(resp)
        except:
            # on parse failure, assume no hallucination
            return 0.0


class PromptEvaluator:
    """
    Tracks per-stage timing & success, and adjusts the system prompt
    for any stage that repeatedly under-performs.
    """
    WINDOW = 10  # how many recent runs to consider

    def __init__(self):
        # stage → list of (duration, success)
        self.metrics = defaultdict(list)

    def record(self, stage: str, duration: float, success: bool):
        m = self.metrics[stage]
        m.append((duration, success))
        if len(m) > self.WINDOW:
            m.pop(0)

    def should_adjust(self, stage: str) -> bool:
        # if more than 30% failures in WINDOW
        m = self.metrics.get(stage, [])
        if len(m) < self.WINDOW:
            return False
        failures = sum(1 for d,s in m if not s)
        return (failures / len(m)) > 0.3

    def adjust(self, stage: str, current_system: str) -> str | None:
        """
        Returns a modified system prompt if needed, else None.
        E.g. adds a clarifying instruction for that stage.
        """
        if not self.should_adjust(stage):
            return None
        # simple strategy: append a note to system prompt
        note = f"\n[Note] Improve reliability of '{stage}' stage by being more explicit."
        return current_system + note
    
class RLManager:
    """
    Keeps a rolling history of prompts/stage-lists and their
    observed 'rewards' (here: success vs. error + optional metrics).
    Provides simple APIs to suggest variations and pick the best.
    """
    WINDOW = 25        # how many recent runs to keep

    def __init__(self):
        self.history = []   # list[(prompt,str), stages,list[str], reward,float]

    # ——— called by ChatManager at end of each run ———
    def record(self, prompt: str, stages: list[str], reward: float):
        self.history.append((prompt, stages, reward))
        if len(self.history) > self.WINDOW:
            self.history.pop(0)

    # ——— produce a candidate (prompt, stages) tuple ———
    def propose(self, base_prompt: str, base_stages: list[str]) -> tuple[str, list[str]]:
        # 1) mutate the prompt (simple example: shuffle two instructions)
        parts = base_prompt.split("\n")
        if len(parts) > 1 and random.random() < 0.5:
            i, j = random.sample(range(len(parts)), 2)
            parts[i], parts[j] = parts[j], parts[i]
        new_prompt = "\n".join(parts)

        # 2) mutate the stage list: maybe drop or add
        stages = base_stages.copy()
        ops = ["drop", "add", "swap"]
        op = random.choice(ops)
        if op == "drop" and len(stages) > 5:
            stages.pop(random.randrange(len(stages)))
        elif op == "add":
            extras = ["html_filtering", "chunk_and_summarize",
                      "rl_experimentation", "prompt_optimization"]
            stages.insert(random.randrange(len(stages)+1),
                          random.choice(extras))
        elif op == "swap" and len(stages) > 2:
            i, j = random.sample(range(len(stages)), 2)
            stages[i], stages[j] = stages[j], stages[i]
        return new_prompt, stages

    # ——— pick best prompt seen so far ———
    def best_prompt(self, default: str) -> str:
        if not self.history:
            return default
        best = max(self.history, key=lambda rec: rec[2])
        return best[0]

class Observer:
    """
    Lightweight run-time telemetry: tracks each pipeline run, every stage’s
    status and timing, and exposes simple signals (e.g. recent failure rate)
    that ChatManager can use to adjust its behaviour.
    """
    _ids = count(1)

    def __init__(self):
        self.runs: dict[int, dict] = {}

    # ---------------- run lifecycle ----------------
    def start_run(self, user_msg: str) -> int:
        run_id = next(self._ids)
        self.runs[run_id] = {
            "user_msg": user_msg,
            "start":    datetime.now(),
            "stages":   defaultdict(dict),
            "status":   "running"
        }
        return run_id

    def log_stage_start(self, run_id: int, stage: str):
        self.runs[run_id]["stages"][stage]["start"] = datetime.now()

    def log_stage_end(self, run_id: int, stage: str):
        self.runs[run_id]["stages"][stage]["end"]   = datetime.now()
        self.runs[run_id]["stages"][stage]["status"] = "done"

    def log_error(self, run_id: int, stage: str, err: Exception):
        self.runs[run_id]["stages"][stage]["error"]  = str(err)
        self.runs[run_id]["stages"][stage]["status"] = "error"

    def complete_run(self, run_id: int):
        self.runs[run_id]["end"]   = datetime.now()
        self.runs[run_id]["status"] = "done"

    # ---------------- signals ----------------
    def recent_failure_ratio(self, n: int = 10) -> float:
        """Return fraction of the last *n* runs that had any stage error."""
        last = list(self.runs.values())[-n:]
        if not last:
            return 0.0
        fails = sum(
            1 for r in last
            if any(s.get("status") == "error" for s in r["stages"].values())
        )
        return fails / len(last)


class ObservabilityManager:
    """
    Wraps Observer to emit every event as JSON over SSE,
    proxies other calls, and adds per-token streaming.
    """
    def __init__(self, real_observer):
        self.real = real_observer
        self._subscribers = []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def subscribe(self, fn):
        self._subscribers.append(fn)

    def _emit(self, event: dict):
        payload = json.dumps(event)
        for fn in list(self._subscribers):
            try:
                fn(payload)
            except:
                self._subscribers.remove(fn)

    # Run lifecycle
    def start_run(self, user_msg):
        rid = self.real.start_run(user_msg)
        self._emit({
            "runId": rid, "type":"run_start", "user_msg":user_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return rid

    def log_stage_start(self, runId, stage):
        self.real.log_stage_start(runId, stage)
        self._emit({
            "runId": runId, "type":"stage_start", "stage":stage,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def log_stage_token(self, runId, stage, token):
        # new: per‐token streaming
        self._emit({
            "runId": runId, "type":"stage_stream",
            "stage":stage, "token": token,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def log_stage_end(self, runId, stage):
        self.real.log_stage_end(runId, stage)
        self._emit({
            "runId": runId, "type":"stage_end", "stage":stage,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def log_error(self, runId, stage, err):
        self.real.log_error(runId, stage, err)
        self._emit({
            "runId": runId, "type":"stage_error", "stage":stage,
            "error": str(err),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def complete_run(self, runId):
        self.real.complete_run(runId)
        self._emit({
            "runId": runId, "type":"run_end",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    # Tool hooks
    def log_tool_start(self, runId, tool):
        self._emit({
            "runId": runId, "type":"tool_start", "tool":tool,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def log_tool_end(self, runId, tool, output):
        self._emit({
            "runId": runId, "type":"tool_end", "tool":tool,
            "output": output,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# -------------------------------------------------------------------
# Expose two module-level variables for LLM introspection
# -------------------------------------------------------------------
ALL_TOOLS  = Tools.load_agent_stack().get("tools", [])
ALL_AGENTS = Tools.load_agent_stack().get("agents", [])

Tools.load_external_tools()

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Context:
    name: str
    history_manager: HistoryManager
    tts_manager: TTSManager
    workspace_memory: dict

    # per-turn fields (seeded by new_request)
    user_message: str = ""
    sender_role: str  = "user"
    skip_tts: bool    = False

    ctx_txt: str = ""
    tool_summaries: list[str] = field(default_factory=list)
    assembled: str = ""
    final_response: Optional[str] = None

    stage_counts: dict[str,int] = field(default_factory=dict)

    # values that some stages expect to find on ctx
    run_id: Optional[int]     = None
    global_tasks: list[Any]   = field(default_factory=list)
    ALL_TOOLS: list[str]      = field(default_factory=list)
    ALL_AGENTS: list[str]     = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

@dataclass
class RunContext:
    run_id:      str
    user_msg:    str
    context_text: str = ""
    tool_choice: str = ""


class ChatManager:
    # ---------------------------------------------------------------- #
    #                           INITIALISATION                         #
    # ---------------------------------------------------------------- #
    def __init__(self,
                config_manager: ConfigManager,
                history_manager: HistoryManager,
                tts_manager: TTSManager,
                tools_data,
                format_schema,
                memory_manager: MemoryManager,
                mode_manager: ModeManager):

        # Core managers & tools
        self.config_manager   = config_manager
        self.tools_data       = tools_data
        self.format_schema    = format_schema
        self.memory_manager   = memory_manager
        self.mode_manager     = mode_manager

        # Observability & learning
        # self.observer         = Observer()
        # wrap real Observer in SSE‐enabled Observable
        real_obs = Observer()
        self.observer = ObservabilityManager(real_obs)

        self.rl_manager       = RLManager()
        self.prompt_evaluator = PromptEvaluator()
        self._temp_bump       = 0.0

        # Context‐switching registry
        self.contexts = {}
        # Create and activate the default “global” context
        self._activate_context("global", history_manager, tts_manager)

        # Inference streaming lock
        self.stop_flag      = False
        self.inference_lock = threading.Lock()
        self.current_thread = None

        # Idle‐handling
        self.last_interaction = datetime.now()
        self.current_run_id   = None
        self.idle_interval    = self.config_manager.config.get("idle_interval_sec", 300)
        self._idle_stop       = threading.Event()
        self._idle_thread     = threading.Thread(target=self._idle_monitor, daemon=True)
        self._idle_thread.start()

        log_message("ChatManager initialized with context switching and idle monitor.", "DEBUG")


    STAGES = [
        # ── core understanding ───────────────────────────────────────────
        ("context_analysis",             "_stage_context_analysis",             None),
        ("intent_clarification",         "_stage_intent_clarification",         "context_analysis"),
        ("external_knowledge_retrieval", "_stage_external_knowledge_retrieval", "intent_clarification"),
        ("planning_summary",             "_stage_planning_summary",             "external_knowledge_retrieval"),

        # ── 🔧 autonomous tool editing  ──────────────────────────────────
        # will run only when `planning_summary` sets ctx.needs_tool_work = True
        ("tool_self_improvement",        "_stage_tool_self_improvement",        "planning_summary"),

        # ── complex-task fulfilment ──────────────────────────────────────
        ("define_criteria",              "_stage_define_criteria",              "tool_self_improvement"),
        ("task_decomposition",           "_stage_task_decomposition",           "define_criteria"),
        ("plan_validation",              "_stage_plan_validation",              "task_decomposition"),
        ("execute_actions",              "_stage_execute_actions",              "plan_validation"),
        ("verify_results",               "_stage_verify_results",               "execute_actions"),

        # ── task-list management ─────────────────────────────────────────
        ("task_management",              "_stage_task_management",              "verify_results"),
        ("subtask_management",           "_stage_subtask_management",           "task_management"),
        ("execute_tasks",                "_stage_execute_tasks",                "subtask_management"),

        # ── normal reply pipeline ────────────────────────────────────────
        ("tool_chaining",                "_stage_tool_chaining",                "execute_tasks"),
        ("assemble_prompt",              "_stage_assemble_prompt",              "tool_chaining"),
        ("final_inference",              "_stage_final_inference",              "assemble_prompt"),
        ("adversarial_loop",             "_stage_adversarial_loop",             "final_inference"),
        ("notification_audit",           "_stage_notification_audit",           "adversarial_loop"),
        ("flow_health_check",            "_stage_flow_health_check",            "notification_audit"),
    ]



    def _stage_flow_health_check(self, ctx):
        """
        After notification_audit, compute a simple health-score (1 − recent_failure_ratio),
        log it, and—if it’s below 0.8—ask the secondary LLM for a recommended new stage list
        (as a Python list).  If valid, persist via Tools.update_agent_stack (with justification).
        """
        import ast
        from datetime import datetime

        # 1) Measure recent failures
        failure_ratio = self.observer.recent_failure_ratio(n=10)
        score = 1.0 - failure_ratio
        log_message(f"[Watchdog] run score={score:.2f}", "INFO")

        suggestions = []
        # 2) If health poor, solicit LLM advice
        if score < 0.8:
            prompt = (
                f"Our last pipeline run scored {score:.2f}/1.00 (lower is worse).\n"
                f"Here’s the context-analysis + tool outputs + critique:\n{ctx.ctx_txt}\n\n"
                "Which ordering of our full stage list (by exact stage names) would you "
                "recommend to improve reliability?  Reply *exactly* with a Python list "
                "of stage names, in the new desired order."
            )
            resp = chat(
                model=self.config_manager.config["secondary_model"],
                messages=[
                    {"role": "system", "content": "You are a pipeline-health adviser."},
                    {"role": "user",   "content": prompt}
                ],
                stream=False
            )["message"]["content"].strip()

            # 3) Parse the LLM’s proposed list
            try:
                cand = ast.literal_eval(resp)
                if isinstance(cand, list) and all(isinstance(n, str) for n in cand):
                    suggestions = cand
                else:
                    log_message("[Watchdog] ignored non-list suggestion", "WARNING")
            except Exception as e:
                log_message(f"[Watchdog] suggestion parse failed: {e}", "WARNING")

        # 4) If we got a non-empty valid suggestion, persist it
        if suggestions:
            current = Tools.load_agent_stack().get("stages", [])
            valid = [s for s in suggestions if s in current]
            if valid:
                just = f"health_check run at {datetime.now().isoformat()} score={score:.2f}"
                try:
                    Tools.update_agent_stack(
                        {"stages": valid},
                        justification=just
                    )
                    log_message(f"[Watchdog] agent_stack.json updated to: {valid}", "INFO")
                    ctx.ctx_txt += f"\n[Watchdog] Updated pipeline stages to: {valid}"
                except Exception as e:
                    log_message(f"[Watchdog] failed to update agent stack: {e}", "ERROR")
            else:
                log_message(
                    "[Watchdog] no overlap between suggestion and existing stages; skipping update",
                    "WARNING"
                )

        return None

    def _inject_python_results(self, raw_text: str) -> str:
        """
        Scan *raw_text* for ```python ...``` blocks, execute them via
        Tools.run_python_snippet, and append a fenced ```output``` block
        with the captured result **immediately after each code block**.

        The returned string is safe to pass to downstream stages/logs.
        """
        def _runner(match):
            code_block = match.group(0)
            code_only  = match.group(1)

            res = Tools.run_python_snippet(code_only)
            pretty = json.dumps(res, indent=2)

            return (
                f"{code_block}\n"
                f"```output\n{pretty}\n```"
            )

        pattern = re.compile(
            r"```(?:python|py)\s*([\s\S]*?)```",
            re.IGNORECASE
        )
        return pattern.sub(_runner, raw_text)

    
    def _activate_context(self, name: str, history_mgr=None, tts_mgr=None):
        if name not in self.contexts:
            self.contexts[name] = Context(
                name=name,
                history_manager=history_mgr or HistoryManager(),
                tts_manager=tts_mgr or TTSManager(),
                workspace_memory=self._load_workspace_memory()
            )
            log_message(f"Created new context '{name}'", "INFO")
        # switch current context
        self.current_ctx = self.contexts[name]
        # make ChatManager use this context’s managers
        self.history_manager = self.current_ctx.history_manager
        self.tts_manager     = self.current_ctx.tts_manager
        Tools._history_manager = self.current_ctx.history_manager

    def _stream_context(self, user_msg: str):
        """
        Phase 1 – context-analysis helper.
        NOTE:  It receives **just the user message string** and
        pulls the active run-id from self.current_run_id, so it
        can be called safely from any stage.
        """
        run_id = self.current_run_id                     # <─ NEW
        log_message("Phase 1: Starting context-analysis stream…", "INFO")
        self.observer.log_stage_start(run_id, "context_analysis")

        # 1) System prompt
        preamble = (
            "You are the CONTEXT-ASSEMBLY AGENT.  Synthesise the user’s "
            "latest message, conversation history and tool capabilities "
            "into a concise analysis (1–4 sentences)."
        )

        # 2) tool signatures
        import inspect, json
        func_sigs = []
        for name in dir(Tools):
            if name.startswith("_"):
                continue
            fn = getattr(Tools, name)
            if callable(fn):
                try:
                    func_sigs.append(f"  • {name}{inspect.signature(fn)}")
                except (TypeError, ValueError):
                    func_sigs.append(f"  • {name}(…)")
        tools_block = "AVAILABLE_TOOLS:\n" + "\n".join(func_sigs)

        # 3) recent + similar history
        recent = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self.history_manager.history[-5:]
        )
        try:
            sim_json = Tools.get_chat_history(user_msg, 5)
            sim = "\n".join(
                f"{e['role'].upper()}: {e['content']}"
                for e in json.loads(sim_json).get("results", [])
            )
        except Exception:
            sim = ""

        ctx_block = ""
        if sim:
            ctx_block += f"SIMILAR PAST:\n{sim}\n\n"
        if recent:
            ctx_block += f"RECENT:\n{recent}\n\n"

        payload = self.build_payload(
            override_messages=[
                {"role": "system", "content": "\n\n".join([preamble, tools_block, ctx_block])},
                {"role": "user",   "content": user_msg}
            ],
            model_key="secondary_model"
        )

        print("⟳ Context: ", end="", flush=True)
        buf = ""
        with self.inference_lock:
            try:
                for part in chat(model=payload["model"],
                                 messages=payload["messages"],
                                 stream=True):
                    self._tick_activity()
                    tok = part["message"]["content"]
                    buf += tok
                    if run_id is not None:                 # <─ NEW
                        self.observer.log_stage_token(run_id, "context_analysis", tok)
                    print(tok, end="", flush=True)
                    yield tok, False
            except Exception as e:
                log_message(f"Context-analysis streaming failed: {e}", "ERROR")
                yield "", True
                return

        print()
        self.observer.log_stage_end(run_id, "context_analysis")
        log_message("Phase 1 complete.", "INFO")
        yield "", True

    # ───────────────────────────────────────────────────────────────
    #  Phase-2: _stream_tool  ― decide which helper to call
    # ───────────────────────────────────────────────────────────────
    def _stream_tool(self, ctx: "RunContext | str"):
        """
        Phase 2 – pick ONE helper (or NO_TOOL).

        • If the caller passes a RunContext we use it directly.
        • If it passes a plain string we fabricate a shim object that has
          .run_id, .user_message, .tool_choice so downstream code is happy.
        """
        import inspect, ast, re                               # NEW: explicit
        # ─── 0) input normalisation ──────────────────────────────────────────
        if isinstance(ctx, str):                              # NEW
            class _Shim:
                def __init__(self, rid, msg):
                    self.run_id        = rid
                    self.user_message  = msg                  # ← unified name
                    self.tool_choice   = ""
            ctx = _Shim(getattr(self, "current_run_id", None), ctx)

        # guard for first-time RunContext instances ---------- NEW
        if not hasattr(ctx, "tool_choice"):
            ctx.tool_choice = ""

        log_message("Phase 2: Starting tool-decision stream…", "INFO")
        self.observer.log_stage_start(ctx.run_id, "tool_decision")

        # ─── 1) enumerate available helpers ────────────────────────────────
        tool_names, func_sigs = [], []
        for name in dir(Tools):
            if name.startswith("_"):
                continue
            fn = getattr(Tools, name)
            if callable(fn):
                tool_names.append(name)
                try:
                    func_sigs.append(f"{name}{inspect.signature(fn)}")
                except (TypeError, ValueError):
                    func_sigs.append(f"{name}(…)")
        tools_msg = "AVAILABLE_FUNCTIONS:\n" + "\n".join(func_sigs)

        preamble = (
            "You are a TOOL-CALLING agent.  Choose exactly one helper function.\n"
            "If none are needed, output NO_TOOL.\n\n"
            "Output ONE fenced block:\n"
            "```tool_code\n<FunctionName>(…)\n```"
        )
        final_inst = (
            "NOW: read the user message below and immediately output your SINGLE "
            "`tool_code` block."
        )

        payload = self.build_payload(
            override_messages=[
                {"role": "system", "content": preamble},
                {"role": "system", "content": tools_msg},
                {"role": "system", "content": final_inst},
                {"role": "user",   "content": ctx.user_message}
            ],
            model_key="secondary_model"
        )
        payload["temperature"] = self.config_manager.config.get(
            "tool_temperature", payload["temperature"]
        )

        # ─── 2) stream LLM output ───────────────────────────────────────────
        print("⟳ Tool: ", end="", flush=True)
        buf = ""
        with self.inference_lock:
            try:
                for part in chat(model=payload["model"],
                                 messages=payload["messages"],
                                 stream=True):
                    self._tick_activity()                     # NEW
                    if self.stop_flag:
                        log_message("Phase 2 aborted by stop flag.", "WARNING")
                        break
                    tok = part["message"]["content"]
                    buf += tok
                    ctx.tool_choice += tok
                    self.observer.log_stage_token(
                        ctx.run_id, "tool_decision", tok
                    )
                    print(tok, end="", flush=True)
                    yield tok, False
            except Exception as e:
                log_message(f"Tool-decision streaming failed: {e}", "ERROR")

        # ─── 3) finalise / parse the tool_code block ───────────────────────
        print()
        log_message(f"Phase 2 complete: raw tool-decision output:\n{buf}", "DEBUG")

        m = re.search(
            r"(?:```tool_code\s*(.*?)\s*```)|^\s*([A-Za-z_]\w*\(.*?\))\s*$",
            buf, re.DOTALL | re.MULTILINE
        )
        code = (m.group(1) or m.group(2) or "").strip() if m else None

        # strip stray inline type hints like “arg: int” ------------- NEW
        if code:
            code = re.sub(r'(\b[A-Za-z_]\w*)\s*:\s*[A-Za-z_]\w*', r'\1', code)

        if code and code.upper() != "NO_TOOL":
            try:
                expr = ast.parse(code, mode="eval")
                call = expr.body
                # structural validation
                if (not isinstance(expr, ast.Expression) or
                    not isinstance(call, ast.Call) or
                    not isinstance(call.func, ast.Name) or
                    call.func.id not in tool_names):
                    raise ValueError("Invalid helper name")
                for arg in call.args:
                    if not isinstance(arg, ast.Constant):
                        raise ValueError("Args must be literals")
                for kw in call.keywords:
                    if not kw.arg or not isinstance(kw.value, ast.Constant):
                        raise ValueError("Kwargs must be literals")
                log_message(f"Tool selected: {code}", "INFO")
            except Exception as e:
                log_message(f"Invalid tool invocation `{code}`: {e}", "ERROR")
                code = None
        else:
            log_message("No valid tool_code detected; treating as NO_TOOL.", "INFO")
            code = None

        self.observer.log_stage_end(ctx.run_id, "tool_decision")
        yield "", True


    # ------------------------------------------------------------------
    #  FIXED  _stage_task_decomposition
    # ------------------------------------------------------------------
    def _stage_task_decomposition(self, ctx) -> list[str]:
        """
        Build a list of tool calls that will achieve the user request.

        – Accepts JSON / Python-literal / Markdown-fenced output from the LLM.  
        – Silently strips *type hints* (`: int`, `: str = …`) and converts any
          **bare tool names** into “no-arg” calls (`foo` → `foo()`).

        Returns the cleaned list (possibly empty) and stores it on ctx.plan.
        """
        import json, ast, re, inspect

        # ---------- 1.  Prompt the LLM ----------
        tool_names = [
            n for n, f in inspect.getmembers(Tools, callable)
            if not n.startswith("_")
        ]
        prompt = (
            "You are a Task-Decomposer.\n"
            f"Available tools: {', '.join(tool_names)}\n\n"
            f'User request: "{ctx.user_message}"\n\n'
            "Return **only** a JSON array where each item is ONE call, e.g.:\n"
            '  ["search_internet(\\"cats\\")", "summarize_search()", ...]'
        )
        resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[{"role":"system","content":prompt}],
            stream=False
        )["message"]["content"]

        # ---------- 2.  Unfence / detype ----------
        raw = re.sub(r"```(?:json|tool_code)?", "", resp, flags=re.I).strip("` \n")
        raw = re.sub(r"\s*#.*", "", raw)                # strip line comments
        raw = re.sub(r":\s*\w+\s*(?==)", "", raw)       # strip “: int =”
        raw = re.sub(r":\s*\w+\s*\)", ")", raw)         # strip “arg: int)”

        # ---------- 3.  Parse to list ----------
        try:
            plan = json.loads(raw)
        except Exception:
            try:
                plan = ast.literal_eval(raw)
            except Exception:
                plan = []

        if not isinstance(plan, list):
            plan = []

        cleaned: list[str] = []
        call_re = re.compile(r'^[A-Za-z_]\w*\s*\(')      # looks like foo(

        for item in plan:
            if not isinstance(item, str):
                continue
            item = item.strip().rstrip(",")
            if not item:
                continue

            # add () if it’s a bare tool name
            if item in tool_names:
                item = f"{item}()"

            # skip unknown tools
            fn = item.split("(", 1)[0]
            if fn not in tool_names:
                continue

            # reject obvious junk
            if not call_re.match(item):
                continue

            cleaned.append(item)

        if not cleaned:
            ctx.ctx_txt += "\n[Task-Decomposition FAILED] no valid tool calls"
            log_message("Task decomposition failed → no valid tool calls", "ERROR")
            ctx.plan = []
            return []

        ctx.plan = cleaned
        ctx.ctx_txt += f"\n[Task-Decomposition] {cleaned}"
        log_message(f"Task plan accepted: {cleaned}", "INFO")
        return cleaned



    # ------------------------------
    # Stage 9: plan_validation  (upgraded)
    # ------------------------------
    def _stage_plan_validation(self, ctx):
        """
        Validate -- and if needed *refine* -- ctx.plan (list[str] tool calls).

        Adds / mutates context attributes:
            ctx.plan_validated : bool
            ctx.needs_tool_work : bool
            ctx.validation_note : str
        """
        import json, re, ast, inspect, textwrap

        plan: list[str] = getattr(ctx, "plan", [])
        if not plan:
            ctx.plan_validated = False
            ctx.validation_note = "No plan generated."
            log_message("[plan_validation] Empty plan – nothing to validate.", "WARNING")
            return

        # ── helper: single-call semantic guard via LLM ──────────────────────
        def _llm_check(call: str) -> tuple[bool, str, str]:
            """
            Returns (ok:bool, reason:str, suggestion:str|NO_TOOL)
            """
            prompt = textwrap.dedent(f"""
                USER REQUEST:
                «{ctx.user_message}»

                CANDIDATE CALL:
                ```tool_code
                {call}
                ```

                Reply in *one* line of valid JSON:
                {{"valid": true|false, "reason": "...", "suggest": "<call>|NO_TOOL"}}
            """)
            raw = chat(
                model=self.config_manager.config["secondary_model"],
                messages=[{"role":"user","content":prompt}],
                stream=False
            )["message"]["content"].strip()
            raw = re.sub(r"```(?:json)?|```", "", raw, flags=re.I).strip()
            try:
                out = json.loads(raw)
                return bool(out.get("valid")), str(out.get("reason","")).strip(), str(out.get("suggest","NO_TOOL")).strip()
            except Exception as e:
                return False, f"Unparsable validator JSON: {e}", "NO_TOOL"

        # ── pass 1 : syntax / availability check ────────────────────────────
        valid_calls : list[str] = []
        tool_names   = {n for n,_ in inspect.getmembers(Tools, inspect.isfunction)
                        if not n.startswith("_")}

        ctx.needs_tool_work = False
        for call_raw in plan:
            call = call_raw.strip()
            # strip stray code fences if decomposer kept them
            call = re.sub(r"^```tool_code\s*|\s*```$", "", call, flags=re.I).strip()

            # quick syntax validation
            parsed = Tools.parse_tool_call(call)
            if not parsed:
                log_message(f"[plan_validation] Dropped invalid syntax: {call_raw!r}", "ERROR")
                continue

            fn_name = parsed.split("(",1)[0].strip()
            if fn_name not in tool_names:
                ctx.needs_tool_work = True
                log_message(f"[plan_validation] Missing helper {fn_name} – will trigger self-improvement.", "INFO")

            valid_calls.append(parsed)

        if not valid_calls:
            ctx.plan_validated = False
            ctx.validation_note = "All steps failed basic syntax."
            return

        # ── pass 2 : semantic approval per call ─────────────────────────────
        final_plan : list[str] = []
        for call in valid_calls:
            ok, reason, sugg = _llm_check(call)
            if ok:
                final_plan.append(call)
                log_message(f"[plan_validation] ✅ {call} – {reason}", "INFO")
            else:
                log_message(f"[plan_validation] ❌ {call} – {reason}", "WARNING")
                if sugg and sugg.upper() != "NO_TOOL":
                    # ensure suggestion has proper syntax before accepting
                    if Tools.parse_tool_call(sugg):
                        final_plan.append(sugg)
                        log_message(f"[plan_validation]   ↳ replaced with {sugg}", "INFO")

        # ── pass 3 : commit results to context ──────────────────────────────
        if final_plan:
            # only mark validated if every step survived the semantic guard
            ctx.plan_validated  = (len(final_plan) == len(valid_calls))
            ctx.plan            = final_plan
            ctx.validation_note = (
                "Plan approved." if ctx.plan_validated
                else "Plan partially revised during validation."
            )
            ctx.ctx_txt += f"\n[Plan validation] {ctx.validation_note}"
        else:
            ctx.plan_validated  = False
            ctx.validation_note = "All candidate steps were rejected."
            ctx.plan            = []

        # Optional: voice feedback
        if not getattr(ctx, "skip_tts", False):
            self.tts_manager.enqueue(ctx.validation_note)

    def _stage_execute_actions(self, ctx):
        """
        Run the validated plan, step-by-step.

        NEW FEATURES
        ────────────
        1. **Guard-rail** – we refuse to execute unless a previous stage set
        `ctx.plan_validated == True`.
        2. **Idle heartbeat** – `self._tick_activity()` keeps the idle-monitor
        timer from firing while long actions run.
        3. **Rich logging** – success/failure for every call is logged +
        appended to ctx.ctx_txt.
        """
        import json, traceback

        # ── 0) Safety check: plan must be validated ─────────────────────────
        if not getattr(ctx, "plan_validated", False):
            note = "Plan not validated – skipping execution stage."
            log_message(f"[execute_actions] {note}", "WARNING")
            ctx.ctx_txt += f"\n[Execute Actions] {note}"
            setattr(ctx, "action_results", [])
            return  # **early exit**

        plan = getattr(ctx, "plan", [])
        if not plan:
            ctx.ctx_txt += "\n[Execute Actions] No steps to run."
            setattr(ctx, "action_results", [])
            return

        results = []
        for call_str in plan:
            self._tick_activity()           # keeps idle-monitor happy

            code = Tools.parse_tool_call(call_str)
            if not code:
                msg = "Error: unparsable call"
                log_message(f"[execute_actions] {call_str!r} → {msg}", "ERROR")
                results.append((call_str, msg))
                continue

            try:
                out = self.run_tool(code)
                log_message(f"[execute_actions] {code} → OK", "INFO")
            except Exception as e:
                out = f"Execution error: {e}"
                log_message(
                    f"[execute_actions] {code} → {e}\n{traceback.format_exc()}",
                    "ERROR"
                )
            results.append((code, out))

        # ── 4) Persist & annotate context ───────────────────────────────────
        setattr(ctx, "action_results", results)

        pretty = "\n".join(f"{c} → {o}" for c, o in results)
        ctx.ctx_txt += f"\n[Executed Actions]\n{pretty}"

    def _stage_verify_results(self, ctx):
        """
        Check each action’s output for errors and either retry or flag issues.
        """
        troubles = [ (c,o) for c,o in getattr(ctx, "action_results", []) if o.lower().startswith("error") ]
        if not troubles:
            return None

        # simple retry logic: retry once
        retry = []
        for code, _ in troubles:
            try:
                out = self.run_tool(code)
            except Exception as e:
                out = f"Error during retry: {e}"
            retry.append((code, out))
        setattr(ctx, "retry_results", retry)
        lines = "\n".join(f"{c} → {o}" for c,o in retry)
        ctx.ctx_txt += f"\n[Verification Retries]:\n{lines}"

        # if still errors, ask for fallback
        still_bad = [ (c,o) for c,o in retry if o.lower().startswith("error") ]
        if still_bad:
            prompt = (
                "The following actions failed twice:\n"
                + "\n".join(f"  • {c}" for c,_ in still_bad)
                + "\nPlease suggest an alternative approach or correct tool calls."
            )
            advice = chat(
                model=self.config_manager.config["secondary_model"],
                messages=[{"role":"system","content":prompt}],
                stream=False
            )["message"]["content"].strip()
            ctx.ctx_txt += f"\n[Retry Advice]: {advice}"
        return None

    def _stage_adversarial_loop(self, ctx):
        """
        Run an adversarial “critic” over the assistant’s own response.
        If it finds issues, it will re-invoke final_inference with those
        criticisms baked into the prompt.
        """
        # only run once we have a candidate answer
        if not ctx.final_response:
            return None

        # ask the secondary LLM to poke holes in our answer
        system = (
            "You are an adversarial critic.  "
            "Given a user request, the context-analysis, and a candidate response, "
            "identify any factual errors, missing steps, logical flaws or unclear points.  "
            "If everything is correct, reply exactly NO_ISSUES."
        )
        user = (
            f"User message: {ctx.user_message}\n\n"
            f"Context analysis + tool outputs:\n{ctx.ctx_txt}\n\n"
            f"Candidate response:\n{ctx.final_response.strip()}\n\n"
            "List each issue as a bullet (– …)."
        )

        resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[{"role":"system","content":system},
                      {"role":"user",  "content":user}],
            stream=False
        )["message"]["content"].strip()

        if resp.upper() == "NO_ISSUES":
            return None

        # log the critique
        ctx.ctx_txt += f"\n[Adversarial critique]:\n{resp}"

        # re-run final_inference, appending the critique to the prompt
        revised_prompt = ctx.assembled + "\n\n[Adversarial critique]:\n" + resp
        new_answer = self.run_inference(revised_prompt, ctx.skip_tts)

        # overwrite with the improved answer
        ctx.final_response = new_answer.strip()
        self.history_manager.add_entry("assistant", ctx.final_response)
        log_message(f"Adversarial loop produced revision: {ctx.final_response!r}", "INFO")
        return None



    # ----------------------------------------
    # NEW: Intent Clarification
    # ----------------------------------------
    def _clarify_intent(self, user_message):
        """
        If the user message is ambiguous, ask a quick follow-up question.
        Returns either None (no clarification needed) or a prompt to send to the user.
        """
        system = (
            "You are an Intent Clarifier. If the user’s request is unclear or might have multiple interpretations, "
            "ask ONE simple question that will let you disambiguate. "
            "If everything is clear, respond with NO_CLARIFICATION."
        )
        user   = f"User said: “{user_message}” — do you need more info?"
        resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            stream=False
        )["message"]["content"].strip()
        if resp.upper() == "NO_CLARIFICATION":
            return None
        return resp

    def _parse_timeframe(self, period: str) -> (str, str):
        from datetime import datetime, timedelta
        today = datetime.now().date()

        if period == "yesterday":
            start = datetime.combine(today - timedelta(days=1), datetime.min.time())
            end   = datetime.combine(today,                        datetime.min.time())
        elif period == "today":
            start = datetime.combine(today,                        datetime.min.time())
            end   = datetime.combine(today + timedelta(days=1),   datetime.min.time())
        elif period.startswith("last "):
            # e.g. “last 7 days”
            try:
                n = int(period.split()[1])
            except ValueError:
                n = 1
            start = datetime.combine(today - timedelta(days=n),   datetime.min.time())
            end   = datetime.combine(today + timedelta(days=1),   datetime.min.time())
        else:
            # fallback to “yesterday”
            start = datetime.combine(today - timedelta(days=1),   datetime.min.time())
            end   = datetime.combine(today,                        datetime.min.time())

        return start.isoformat(), end.isoformat()

    def _stage_subtask_management(self, ctx):
        """
        Allows the agent to break a parent task into subtasks,
        list or update their status, or assemble results when done.
        """
        prompt = (
            "You may manage SUBTASKS of existing tasks.  Use exactly one Tools call from:\n"
            "  • list_subtasks(parent_id)\n"
            "  • add_subtask(parent_id, text)\n"
            "  • set_task_status(id, 'in_progress'|'done')\n"
            "  • NO_OP (if no subtask action needed)\n"
        )
        decision = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[
                {"role":"system", "content": prompt},
                {"role":"user",   "content": ctx["user_message"]}
            ],
            stream=False
        )["message"]["content"].strip()

        call = Tools.parse_tool_call(decision)
        if call and call.upper() != "NO_OP":
            out = self.run_tool(call)
            # refresh both parent tasks and subtasks
            try:
                ctx["global_tasks"]   = json.loads(Tools.list_tasks())
                # extract parent_id if relevant
                import re
                m = re.match(r"(\w+)\(\s*(\d+)", call)
                if m and m.group(1) == "list_subtasks":
                    pid = int(m.group(2))
                    ctx["subtasks"] = Tools.list_subtasks(pid)
            except:
                pass
            ctx["ctx_txt"] += f"\n[Subtask Op] {call} → {out}"
        return None

    
    def _history_timeframe_query(self, user_message: str) -> str | None:
        """
        Detects time-based history queries like:
          - "what did we talk about yesterday?"
          - "show me our chats from last week"
          - "what was said between 2025-05-01 and 2025-05-10?"
          - "past 3 days"
        If matched, returns a formatted string of the relevant history and
        enqueues it to TTS; otherwise returns None.
        """
        import re
        from datetime import datetime, timedelta, date

        # normalize
        msg = user_message.lower()

        # 1) explicit range "from X to Y"
        m = re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", msg)
        if m:
            start = datetime.fromisoformat(m.group(1)).isoformat()
            # include full Y-day
            end_dt = datetime.fromisoformat(m.group(2)) + timedelta(days=1)
            end = end_dt.isoformat()
        else:
            today = date.today()
            # 2) single keywords
            if "yesterday" in msg:
                start_dt = datetime.combine(today - timedelta(days=1), datetime.min.time())
                end_dt   = datetime.combine(today,             datetime.min.time())
            elif "today" in msg:
                start_dt = datetime.combine(today,             datetime.min.time())
                end_dt   = datetime.combine(today + timedelta(days=1), datetime.min.time())
            elif "last week" in msg:
                # ISO week starts Monday
                monday = today - timedelta(days=today.weekday())
                start_dt = datetime.combine(monday - timedelta(weeks=1), datetime.min.time())
                end_dt   = datetime.combine(monday,               datetime.min.time())
            elif "this week" in msg:
                monday = today - timedelta(days=today.weekday())
                start_dt = datetime.combine(monday, datetime.min.time())
                end_dt   = datetime.combine(monday + timedelta(weeks=1), datetime.min.time())
            elif "last month" in msg:
                first_this = today.replace(day=1)
                # subtract one day to get last month's last day, then to 1st
                prev_last_day = first_this - timedelta(days=1)
                start_dt = datetime.combine(prev_last_day.replace(day=1), datetime.min.time())
                end_dt   = datetime.combine(first_this,            datetime.min.time())
            elif "this month" in msg:
                first = today.replace(day=1)
                # next month
                if first.month == 12:
                    nm = first.replace(year=first.year+1, month=1)
                else:
                    nm = first.replace(month=first.month+1)
                start_dt = datetime.combine(first, datetime.min.time())
                end_dt   = datetime.combine(nm,    datetime.min.time())
            else:
                # 3) "past N days" or "N days ago"
                m2 = re.search(r"(\d+)\s*(?:days?\s*(?:ago|past)?)", msg)
                if m2:
                    n = int(m2.group(1))
                    start_dt = datetime.combine(today - timedelta(days=n), datetime.min.time())
                    end_dt   = datetime.combine(today + timedelta(days=1), datetime.min.time())
                else:
                    return None  # no timeframe found
            start, end = start_dt.isoformat(), end_dt.isoformat()

        # 4) fetch and filter
        history_json = Tools.get_chat_history(1000, start)
        try:
            results = json.loads(history_json).get("results", [])
        except Exception:
            return None

        # only keep those before `end`
        filtered = [
            r for r in results
            if r.get("timestamp", "") < end
        ]
        if not filtered:
            out = "I couldn't find any messages in that time frame."
        else:
            lines = [
                f"{r['timestamp']}  {r['role'].capitalize()}: {r['content']}"
                for r in filtered
            ]
            out = "\n".join(lines)

        # speak & print
        print(out)
        log_message(f"History time‐frame ({start}→{end}): {len(filtered)} messages", "INFO")
        self.tts_manager.enqueue(out)
        return out

    def _stage_tool_self_improvement(self, ctx):
        """
        Stage: tool_self_improvement
        ────────────────────────────
        Trigger – `planning` (or the user) sets **ctx.needs_tool_work = True**.
        Goal     – Let the LLM iteratively CREATE / MODIFY / TEST tools until it
                    declares “NO_TOOL”.

        Loop outline (max `tool_iter_max`, default = 4):
            1. Build a *local* conversation containing:
                • A reminder of the task and current plan
                • Any evaluation rubric the planner stored in
                    ctx.tool_eval_plan  (dict)  – optional
                • A concise history of previous iterations
                • A cheatsheet of helper functions
            2. Stream the secondary-model; expect ONE ```tool_code``` block.
            3. Parse → execute immediately with  run_tool()  (same validator
            you already use elsewhere).  Capture result / exception.
            4. Append a compact   [tool_improvement]   entry to ctx.ctx_txt
            (so downstream stages can see the evolution).
            5. Feed the *result summary* back into the next iteration’s history.
            6. Exit when   NO_TOOL   or when max iterations reached.
        On exit:
            • `Tools.load_external_tools()` to make new functions live.
            • Observer hooks for start/end/tokens just like other stages.
        """
        import json, textwrap, traceback

        # ── 0. quick exit ────────────────────────────────────────────────
        if not getattr(ctx, "needs_tool_work", False):
            return                                                        # skip stage

        run_id = getattr(ctx, "run_id", None)
        if run_id:
            self.observer.log_stage_start(run_id, "tool_self_improvement")

        # ─────────────────────────────────────────────────────────────────
        max_iter = int(self.config_manager.config.get("tool_iter_max", 4))
        history  = []             # [(call_str, truncated_result), …]

        # helper: shorten any result so prompts stay small ↓↓↓
        def _short(r, n=300):
            s = str(r)
            return (s[:n] + " …") if len(s) > n else s

        for iteration in range(1, max_iter + 1):
            self._tick_activity()                                         # idle-reset

            # ── 1. Build conversation for secondary model ───────────────
            cheat_sheet = textwrap.dedent("""
                You are in TOOL-IMPROVEMENT mode.

                ✔ Helper functions you may call *exactly one per message*
                (wrap in a ```tool_code``` block):
                    • create_tool(name:str, code:str, overwrite=True|False)
                    • get_tool_source(name:str)
                    • test_tool(name:str, cases:list[dict])
                    • run_tool_once("foo(123)")
                    • run_python_snippet(\"\"\"print('hi')\"\"\")
                If you are satisfied and wish to leave this mode, output:

                    ```tool_code
                    NO_TOOL
                    ```
            """)

            # –– contextual snapshot (task + evaluation plan) ––
            task_snip = _short(ctx.user_msg if hasattr(ctx, "user_msg") else "")
            plan_snip = _short(getattr(ctx, "plan_summary", ""))
            eval_plan = _short(json.dumps(getattr(ctx, "tool_eval_plan", {}), indent=2))

            hist_lines = [
                f"{i}. {c}  →  {r}" for i, (c, r) in enumerate(history, 1)
            ]
            hist_txt = "\n".join(hist_lines) or "(first pass)"

            system_prompt = (
                cheat_sheet
                + f"\nCURRENT TASK: {task_snip}"
                + (f"\nCURRENT PLAN: {plan_snip}" if plan_snip else "")
                + (f"\nEVALUATION RUBRIC:\n{eval_plan}" if eval_plan else "")
                + f"\n\nEarlier iterations:\n{hist_txt}\n"
                + "\nRespond with ONE tool_code block now."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                # A minimal user kick – keeps roles consistent
                {"role": "user",   "content": "Make your next improvement."}
            ]

            # ── 2. Stream the LLM response ───────────────────────────────
            buf = ""
            print(f"\n🛠  Tool-loop #{iteration}: ", end="", flush=True)
            try:
                for part in chat(
                    model=self.config_manager.config["secondary_model"],
                    messages=messages,
                    stream=True
                ):
                    tok = part["message"]["content"]
                    buf += tok
                    # observer streaming trace
                    if run_id:
                        self.observer.log_stage_token(
                            run_id, "tool_self_improvement", tok
                        )
                    print(tok, end="", flush=True)
                    self._tick_activity()
            except Exception as e_stream:
                log_message(f"[tool_self_improvement] streaming error: {e_stream}", "ERROR")
                break
            print()

            # ── 3. Parse the SINGLE tool_code block or inline call ───────
            call_str = Tools.parse_tool_call(buf) or "NO_TOOL"
            if call_str.upper() == "NO_TOOL":
                log_message("[tool_self_improvement] LLM signalled finish.", "DEBUG")
                break                                                    # exit cleanly

            # ── 4. Execute helper immediately ────────────────────────────
            try:
                result = self.run_tool(call_str)
            except Exception as e_exec:
                result = f"Execution-error: {e_exec}\n{traceback.format_exc(limit=2)}"

            # ⟹ record to ctx + history
            ctx.ctx_txt += f"\n[tool_improvement] {call_str}  →  {_short(result,120)}"
            history.append((call_str, _short(result)))

            # terse console feedback
            print(f"↪ {call_str}  →  {_short(result,160)}")

            # ── 5. decide whether to continue (LLM may decide) ───────────
            # (If the helper just ran tests and all passed, it will likely
            #  output NO_TOOL next round.  We rely on the language model’s
            #  judgement; max_iter is the hard stop.)

        # ── 6. Finalise ──────────────────────────────────────────────────
        try:
            Tools.load_external_tools()          # pick up any new .py files
        except Exception as e_reload:
            log_message(f"[tool_self_improvement] reload failed: {e_reload}", "ERROR")

        if run_id:
            self.observer.log_stage_end(run_id, "tool_self_improvement")


    # NEW: External Knowledge Retrieval (RAG)
    # ----------------------------------------
    def _fetch_external_knowledge(self, user_message):
        """
        If the user query seems to need up-to-date facts, do a quick web search.
        Returns a short summary string (or empty).
        """
        # naive keyword check: you can customize
        if any(kw in user_message.lower() for kw in ("latest", "today", "current")):
            raw = Tools.search_internet(user_message)
            # assume raw is JSON text
            try:
                data = json.loads(raw)
                top = data["web"]["results"][:2]
                lines = [f"{r['title']}: {r.get('description','')}" for r in top]
                return "External facts:\n" + "\n".join(lines)
            except:
                return ""
        return ""

    # ----------------------------------------
    # NEW: Memory Summarization
    # ----------------------------------------
    def _memory_summarize(self):
        """
        Every N turns compresses full history to a short summary stored in memory.
        Returns a summary string (or None if not time yet or on error).
        """
        # only run every 10th turn
        if len(self.history_manager.history) % 10 != 0:
            return None

        full = "\n".join(m["content"] for m in self.history_manager.history)
        try:
            resp = chat(
                model=self.config_manager.config["secondary_model"],
                messages=[
                    {"role": "system", "content":
                        "You are a Memory Agent—summarize this conversation in one paragraph."
                    },
                    {"role": "user", "content": full}
                ],
                stream=False
            )
            # extract and log the summary
            summary = resp["message"]["content"].strip()
            log_message("Memory summary: " + summary, "INFO")
            return summary

        except Exception as e:
            # catch ResponseError, network errors, etc.
            log_message(f"Memory summarization failed: {e}", "ERROR")
            return None

    def _stage_html_filtering(self, ctx):
        """
        Strip tags & boilerplate from raw HTML so downstream agents
        receive only meaningful text.  If no HTML present, do nothing.
        """
        raw = ctx["user_message"]
        # quick heuristic: looks like HTML if it contains both "<" and "</"
        if "<" in raw and "</" in raw:
            try:
                from bs4 import BeautifulSoup
                clean_text = BeautifulSoup(raw, "html.parser").get_text(
                    separator=" ",
                    strip=True
                )
                # replace the user_message with clean text for all later stages
                ctx["user_message"] = clean_text
                ctx["ctx_txt"] += "\n[HTML stripped]"
                log_message("HTML stripped successfully.", "DEBUG")
            except Exception as e:
                log_message(f"HTML filtering failed: {e}", "WARNING")
        return None

    def _stage_chunk_and_summarize(self, ctx):
        """
        When ctx['user_message'] is extremely large, break it into
        manageable chunks, summarize each, and collapse the output
        into ctx['user_message'] so later stages handle only the summary.
        """
        text = ctx["user_message"]
        # threshold: only summarize if > 4 000 characters (~1 000 tokens)
        if len(text) < 4000:
            return None

        chunk_size = 4000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []

        for idx, chunk in enumerate(chunks, 1):
            prompt = (
                f"You are a world-class summarizer.  Summarize chunk {idx}/{len(chunks)} "
                f"in 2–3 sentences, preserving every key fact and name:\n\n{chunk}"
            )
            try:
                summ = chat(
                    model=self.config_manager.config["secondary_model"],
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )["message"]["content"].strip()
                summaries.append(summ)
            except Exception as e:
                log_message(f"Chunk summarization failed: {e}", "ERROR")
                continue

        combined = "\n".join(summaries)
        ctx["user_message"] = combined
        ctx["ctx_txt"]     += "\n[Chunked summary created]"
        log_message("Large text summarized into chunks.", "INFO")
        return None

    # ----------------------------------------
    # NEW: Human-in-the-Loop Review
    # ----------------------------------------
    def _human_review(self, assembled):
        """
        For very long or high-risk responses, escalate to a human.
        Returns True to proceed automatically, False to pause.
        """
        # here we just check length > 200 tokens
        if len(assembled.split()) > 200:
            print("\n⚠️  Response is lengthy—please review before sending:")
            print(assembled[:500] + "…")
            input("Press Enter to continue automatically…")
        return True
    # ── Idle-handling helpers ─────────────────────────────────────────────
    def _touch_activity(self):
        """Mark ‘now’ as the last time *anything* happened."""
        self.last_interaction = datetime.now()

    def _tick_activity(self):                     # alias used in tight loops
        self.last_interaction = datetime.now()   

    def _idle_monitor(self):
        """
        Fires only after <idle_interval> seconds of *real* silence.
        Any token, tool call, or stage transition resets the timer via
        self._touch_activity(), so the idle mull can never interrupt an
        active run.
        """
        import time
        while not self._idle_stop.is_set():
            time.sleep(1)                                     # light-weight poll
            idle_secs = (datetime.now() - self.last_interaction).total_seconds()
            if idle_secs < self.idle_interval:
                continue                                      # still “busy”
            # ---- idle window reached ------------------------------------------------
            ctx  = self.current_ctx
            recent = (ctx.ctx_txt or "").replace("\n", " ")
            snip   = (recent[-200:] + "...") if len(recent) > 200 else recent
            prompt = (
                f"I've been idle for {int(idle_secs)} seconds. "
                f"Recent context: {snip} "
                "What would you like me to work on next?"
            )
            log_message(f"[Idle Monitor] Triggered after {int(idle_secs)} s", "INFO")
            # Reset BEFORE launching the mull so cascaded failures can't recurse
            self._touch_activity()
            # Internal, silent request (no TTS)
            self.new_request(prompt, sender_role="assistant", skip_tts=True)


    def _start_idle_mull(self):
        """
        Background thread: after a random idle interval between idle_min/idle_max,
        if still silent, schedule a new_request(...) that pulls in latest context
        similarly to _idle_monitor, but on a randomized cadence.
        """
        import threading, random, time
        from datetime import datetime

        stop_evt = threading.Event()
        self._idle_stop_event = stop_evt

        def _mull_loop():
            while not stop_evt.is_set():
                interval = random.uniform(self.idle_min, self.idle_max)
                time.sleep(interval)
                idle_time = (datetime.now() - self.last_interaction).total_seconds()
                if idle_time >= interval:
                    log_message("Random idle detected: scheduling background mull", "INFO")
                    # build a dynamic prompt as above
                    ctx = self.current_ctx
                    recent = (ctx.ctx_txt or "").replace("\n", " ")
                    snip = (recent[-200:] + "...") if len(recent) > 200 else recent
                    prompt = (
                        f"It's been {int(idle_time)}s since the last command. "
                        f"Recent context: {snip} "
                        "Anything you’d like me to pick up on?"
                    )
                    # skip TTS so it doesn’t speak to an empty room
                    self._executor.submit(
                        self.new_request,
                        prompt,
                        sender_role="assistant",
                        skip_tts=True
                    )

        thread = threading.Thread(target=_mull_loop, daemon=True)
        thread.start()
        self._idle_thread = thread


    def stop_idle_mull(self):
        """
        Stop the background idle‐mull loop and idle monitor.
        """
        if hasattr(self, "_idle_stop_event"):
            self._idle_stop_event.set()
            log_message("Idle mull loop stopped.", "INFO")
        if hasattr(self, "_idle_stop"):
            self._idle_stop.set()
            log_message("Idle monitor stopped.", "INFO")

    # ----------------------------------------
    # NEW: Chain-of-Thought Agent
    # ----------------------------------------
    def _chain_of_thought(self,
                          user_message: str,
                          context_analysis: str,
                          external_facts: str,
                          memory_summary: str,
                          planning_summary: str,
                          tool_summaries: list[str],
                          final_response: str) -> str:
        """
        Build an incremental chain-of-thought that builds on prior reflections,
        weaving in just the new stages for this turn.
        """

        import os, json
        from datetime import datetime

        # 1) Load previous reflections (if any)
        thoughts_path = os.path.join(session_folder, "thoughts.json")
        prior = []
        if os.path.isfile(thoughts_path):
            try:
                with open(thoughts_path, "r", encoding="utf-8") as f:
                    prior = json.load(f)
            except:
                prior = []

        # 2) Prepare the prompt for the reflection agent
        system = (
            "You are the Reflection Agent, an expert metacognitive overseer embedded in "
            "this conversational system. Your singular mission is to maintain and evolve a "
            "running narrative of the assistant’s reasoning process over time. You will be "
            "provided with: (a) a bullet-list of *all prior reflection entries* you have "
            "written, and (b) a bullet-list of *new stages* from the current turn (context "
            "analysis, external facts, memory summary, planning summary, tool outputs, final "
            "response).  \n\n"
            "Your task is to produce **one** concise reflection—exactly 2–3 sentences—that "
            "**appends** to the end of the existing narrative. Do **not** rehash old content. "
            "Do **not** number your response. Focus purely on weaving the new stages into "
            "the overarching story of how the assistant arrived at its response.  "
            "Be precise, avoid filler, and use direct, active language."
        )


        # Format prior reflections as bullets
        if prior:
            prior_block = "\n".join(
                f"{i+1}. {entry['chain_of_thought']}"
                for i, entry in enumerate(prior)
            )
        else:
            prior_block = "(none so far)"

        # Gather the new stages succinctly
        stages = [
            f"User message: {user_message}",
            f"Context analysis: {context_analysis}"
        ]
        if external_facts:
            stages.append(f"External facts: {external_facts}")
        if memory_summary:
            stages.append(f"Memory summary: {memory_summary}")
        if planning_summary:
            stages.append(f"Planning summary: {planning_summary}")
        if tool_summaries:
            stages.append("Tool outputs:\n" + "\n".join(tool_summaries))
        stages.append(f"Final response: {final_response}")

        new_block = "\n".join(f"- {s}" for s in stages)

        # 3) Call the secondary model
        payload = [
            {"role": "system",  "content": system},
            {"role": "assistant", "content": f"Prior reflections:\n{prior_block}"},
            {"role": "user",    "content": f"New stages:\n{new_block}\n\nPlease append the next reflection to the narrative."}
        ]
        resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=payload,
            stream=False
        )["message"]["content"].strip()

        # 4) Return just the new reflection (no numbering)
        #    The calling code will append it to thoughts.json.
        return resp

    # ----------------------------------------
    # NEW: Notification & Audit
    # ----------------------------------------
    def _emit_event(self, event, payload):
        """
        Stub for webhook or metrics emission.
        You can hook this to requests.post(...) or a Prometheus counter.
        """
        log_message(f"[EVENT] {event}: {payload}", "DEBUG")
        # e.g. requests.post("https://my.monitor/api", json={...})

    # ---------------------------------------------------------------- #
    #                    ORIGINAL (UNCHANGED) METHODS                  #
    # ---------------------------------------------------------------- #
    def build_payload(self, override_messages=None, model_key="primary_model"):
        cfg = self.config_manager.config
        if override_messages is not None:
            messages = override_messages
        elif model_key == "primary_model":
            system_prompt = cfg.get("system","")
            sys_msg = {"role":"system","content":system_prompt}
            if self.history_manager.history:
                last = next((m["content"] for m in reversed(self.history_manager.history)
                             if m["role"]=="user"), "")
                mem = Utils.embed_text(last)
            else:
                mem = ""
            mem_msg = {"role":"system",
                       "content":f"Memory Context:\n{mem}\n\nSummary Narrative:\n\n"}
            messages = [sys_msg, mem_msg] + self.history_manager.history
        else:
            messages = override_messages or []
            
        base_temp = cfg[f"{model_key.split('_')[0]}_temperature"]
        temp      = base_temp + 0.5 * self.observer.recent_failure_ratio() + self._temp_bump
        payload = {
            "model":       cfg[model_key],
            "temperature": min(temp, 1.5),  # cap it
            "messages":    messages,
            "stream":      cfg["stream"]
        }
        if self.format_schema:
            payload["format"] = self.format_schema
        if cfg.get("raw"):
            payload["raw"] = True
        if cfg.get("options"):
            payload["options"] = cfg["options"]
        return payload

    def chat_completion_stream(self, processed_text):
        """
        Stream the primary‐model’s response to `processed_text` as a new user message.
        Ensures the model sees a fresh user turn rather than two assistant turns in a row,
        and aborts if it detects runaway repetition.
        """

        log_message("Primary-model stream starting...", "DEBUG")

        # 1) Build override messages so the LLM sees processed_text as a user turn
        cfg     = self.config_manager.config
        sys_msg = {"role": "system", "content": cfg.get("system", "")}

        # Optional: include a memory/context message if you use that pattern
        mem_content = ""  # or pull from your memory manager
        override = [sys_msg]
        if mem_content:
            override.append({"role": "system", "content": f"Memory Context:\n{mem_content}\n\n"})

        # add the full chat history so far
        override.extend(self.history_manager.history)
        # finally, our new “user” turn
        override.append({"role": "user", "content": processed_text})

        # 2) Build the payload with our override_messages
        payload = self.build_payload(
            override_messages=override,
            model_key="primary_model"
        )

        # 3) Stream and watch for repetition
        print("⟳ Reply: ", end="", flush=True)
        recent_chunks = deque(maxlen=5)

        for part in chat(model=payload["model"], messages=payload["messages"], stream=True):
            # immediate abort if external stop requested
            if self.stop_flag:
                log_message("Primary-model stream aborted by stop_flag.", "WARNING")
                yield "", True
                return

            tok = part["message"]["content"]
            self.observer.log_stage_token(self.current_ctx.run_id, "final_inference", tok)
            print(tok, end="", flush=True)
            yield tok, part.get("done", False)

            # repetition guard
            snippet = tok.strip()
            if snippet:
                recent_chunks.append(snippet)
                if len(recent_chunks) == recent_chunks.maxlen and len(set(recent_chunks)) == 1:
                    log_message(f"Detected runaway repetition of {snippet!r}; aborting stream.", "WARNING")
                    self.stop_flag = True
                    yield "", True
                    return

        print()
        log_message("Primary-model stream finished.", "DEBUG")

    def chat_completion_nonstream(self, processed_text):
        log_message("Primary-model non-stream starting...", "DEBUG")
        payload = self.build_payload(None, model_key="primary_model")
        try:
            resp = chat(model=payload["model"], messages=payload["messages"], stream=False)
            return resp["message"]["content"]
        except Exception as e:
            log_message(f"Primary-model non-stream error: {e}", "ERROR")
            return ""

    def process_text(self, text, skip_tts: bool = False):
        """
        Robust, self-adapting wrapper that keeps retrying until we obtain a
        satisfactory LLM answer.  Each retry escalates through a predefined
        list of tactics (temperature bumps, model switch, history-window trim,
        non-stream fallback, etc.).  It NEVER gives up.
        """

        from contextlib import contextmanager

        log_message("process_text: Starting...", "DEBUG")

        # ---------- helpers -------------------------------------------------
        @contextmanager
        def _temporary_history(window: int | None):
            """
            Temporarily clip self.history_manager.history to its last <window>
            entries.  If window is None we leave history untouched.
            """
            if window is None:
                yield
                return

            original = self.history_manager.history
            try:
                self.history_manager.history = original[-window:]
                yield
            finally:
                self.history_manager.history = original

        def _build_stream():
            """(re)create a fresh monitored stream with current params."""
            stream = self.chat_completion_stream(converted)
            return self.watchdog.monitor_stream(stream, self.current_ctx)

        # ---------- pre-flight ---------------------------------------------
        text = text.replace("*", "")
        converted = Utils.convert_numbers_to_words(text)
        sentence_endings = re.compile(r"[.?!]+")
        in_think = False

        # watchdog (reuse if already exists)
        if not hasattr(self, "watchdog"):
            self.watchdog = WatchdogManager(
                quality_model=self.config_manager.config["secondary_model"],
                halluc_model=self.config_manager.config["secondary_model"],
                quality_threshold=0.8,
                halluc_threshold=0.5,
                max_retries=0,          # we do the retries
            )
        wd = self.watchdog

        # rotation data
        apology_variants = [
            "Sorry, that drifted—let me tighten things up and retry.",
            "Hmm, still not crisp.  Adjusting my approach…",
            "Let me rethink and take another shot.",
            "One more try with a fresh strategy.",
        ]

        # escalating tactics table
        tactics = [
            # (use_stream, use_secondary, history_window, extra_temp, halluc_delta)
            (True,  False, None, 0.0, 0.0),   # 1
            (True,  False, None, 0.2, 0.1),   # 2
            (True,  True,  None, 0.3, 0.2),   # 3
            (True,  True,   25,  0.4, 0.3),   # 4
            (True,  True,   10,  0.5, 0.4),   # 5
            (False, True,   10,  0.5, 0.5),   # 6
            (True,  True,    5,  0.6, 0.5),   # 7
        ]

        attempt = 0
        while True:  # never stop trying
            attempt += 1
            strat = tactics[(attempt - 1) % len(tactics)]
            use_stream, use_secondary, hist_win, temp_bump, hallu_delta = strat

            # 1) apply strategy-specific knobs
            self._temp_bump = temp_bump
            wd.halluc_threshold = min(1.0, wd.halluc_threshold + hallu_delta)

            if use_secondary:
                # temporarily swap primary/secondary models in config for this try
                orig_primary = self.config_manager.config["primary_model"]
                self.config_manager.config["primary_model"] = self.config_manager.config["secondary_model"]
            else:
                orig_primary = None  # marker for no swap

            # ensure stream flag matches tactic
            self.config_manager.config["stream"] = use_stream

            log_message(
                f"[Retry #{attempt}] "
                f"{'stream' if use_stream else 'non-stream'}, "
                f"model={'secondary' if use_secondary else 'primary'}, "
                f"hist={hist_win or 'full'}, "
                f"Δtemp={temp_bump:.2f}, "
                f"halluc_thr={wd.halluc_threshold:.2f}",
                "INFO",
            )

            try:
                with _temporary_history(hist_win):
                    if use_stream:
                        # ------- streaming branch -----------------------------------
                        tokens = ""
                        tts_buffer = ""
                        self.stop_flag = False
                        monitored = _build_stream()

                        for chunk, done in monitored:
                            tokens += chunk
                            with display_state.lock:
                                display_state.current_tokens = tokens

                            # remove <think> tags from spoken text
                            data = chunk
                            idx = 0
                            while idx < len(data):
                                if not in_think:
                                    start = data.find("<think>", idx)
                                    if start == -1:
                                        tts_buffer += data[idx:]
                                        break
                                    tts_buffer += data[idx:start]
                                    idx = start + len("<think>")
                                    in_think = True
                                else:
                                    end = data.find("</think>", idx)
                                    if end == -1:
                                        break
                                    idx = end + len("</think>")
                                    in_think = False

                            # flush complete sentences to TTS
                            while True:
                                m = sentence_endings.search(tts_buffer)
                                if not m:
                                    break
                                end_i = m.end()
                                sentence = tts_buffer[:end_i].strip()
                                tts_buffer = tts_buffer[end_i:].lstrip()
                                cleaned = re.sub(r"[*_`]", "", clean_text(sentence))
                                if cleaned and not skip_tts:
                                    self.tts_manager.enqueue(cleaned)
                                    log_message(f"TTS enqueued: {cleaned}", "DEBUG")

                            if done:
                                # SUCCESS
                                return tokens
                    else:
                        # ------- non-stream branch ----------------------------------
                        res = self.chat_completion_nonstream(converted)
                        with display_state.lock:
                            display_state.current_tokens = res
                        return res

            except StopStreamException:
                # Watchdog aborted → escalate
                pass
            except Exception as e:
                # transport or model error → log & escalate
                log_message(f"[Retry #{attempt}] Transport/model error: {e}", "ERROR")

            # -------- failure handling ----------------------------------------
            apology = apology_variants[(attempt - 1) % len(apology_variants)]
            if not skip_tts:
                self.tts_manager.enqueue(apology)
            log_message(f"Retry #{attempt} failed → apologising: {apology}", "WARNING")

            # restore swapped model, if any, before next loop
            if orig_primary is not None:
                self.config_manager.config["primary_model"] = orig_primary

            # small sleep prevents hammering the local LLM server if it's crashing
            time.sleep(0.5)


    def inference_thread(self, user_message, result_holder, skip_tts):
        try:
            # process_text now handles its own retries and won’t raise on streaming errors
            result = self.process_text(user_message, skip_tts)
        except Exception as e:
            # last-ditch catch – log and return fallback
            log_message(f"inference_thread: fatal error: {e}", "ERROR")
            result = "Sorry, something went seriously wrong. Let me try again."
        result_holder.append(result)

    def run_inference(self, prompt, skip_tts=False):
        log_message("run_inference: Starting...", "DEBUG")
        holder = []
        with self.inference_lock:
            if self.current_thread and self.current_thread.is_alive():
                self.stop_flag = True
                self.current_thread.join()
                self.stop_flag = False
            self.tts_manager.stop()
            self.tts_manager.start()
            self.current_thread = threading.Thread(
                target=self.inference_thread,
                args=(prompt, holder, skip_tts)
            )
            self.current_thread.start()
        self.current_thread.join()
        return holder[0] if holder else ""
    
    def run_tool(self, tool_code: str) -> str:
        """
        Execute a Tools.<func>(...) invocation, supporting:
        - keyword args with literal constants
        - bare positional args (quoted or unquoted → treated as strings)
        - Name nodes (treated as their .id)
        - automatic conversion of kwargs→args for positional-only tools
        """

        # ── mark activity so idle-monitor won’t fire while we work ──────────
        if hasattr(self, "_tick_activity"):
            self._tick_activity()

        # ── OBSERVABILITY: emit tool_start ─────────────────────────────────
        rid       = getattr(self, "current_run_id", None)
        tool_name = tool_code.split("(", 1)[0]
        if rid is not None:
            self.observer.log_tool_start(rid, tool_name)

        # Strip out any type annotations in keyword args (e.g. prompt: str="…" → prompt="…")
        tool_code = re.sub(
            r'(\b[A-Za-z_]\w*)\s*:\s*[A-Za-z_]\w*\s*=',
            r'\1=',
            tool_code
        )

        log_message(f"run_tool: Executing {tool_code!r}", "DEBUG")

        # ------------------------------------------------------------------
        # 0) Quick split-based parser for simple, *unquoted* args
        # ------------------------------------------------------------------
        m_quick = re.fullmatch(
            r'\s*([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*',
            tool_code,
            re.DOTALL
        )
        if m_quick:
            func_name, body = m_quick.group(1), m_quick.group(2)
            func = getattr(Tools, func_name, None)
            if func and not re.search(r'["\']', body):
                args, kwargs = [], {}
                for part in re.split(r'\s*,\s*', body):
                    part = part.strip()
                    if not part:
                        continue
                    if '=' in part:
                        k, v = part.split('=', 1)
                        k, v = k.strip(), v.strip()
                        if v[:1] in "\"'" and v[-1:] in "\"'":
                            v = v[1:-1]
                        kwargs[k] = v
                    else:
                        v = part
                        if v[:1] in "\"'" and v[-1:] in "\"'":
                            v = v[1:-1]
                        args.append(v)

                # default for get_chat_history()
                if func_name == "get_chat_history" and not args and not kwargs:
                    args = [5]

                log_message(f"run_tool: Parsed {func_name} with args={args} kwargs={kwargs}", "DEBUG")
                try:
                    sig = inspect.signature(func)
                    pos_names = [p.name for p in sig.parameters.values()
                                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                    if not args and kwargs and all(n in kwargs for n in pos_names):
                        args = [kwargs.pop(n) for n in pos_names]
                    result = func(*args, **kwargs)
                    output = str(result)
                    log_message(f"run_tool: {func_name} returned {output!r}", "INFO")
                except Exception as e:
                    output = f"Error executing `{func_name}`: {e}"
                    log_message(f"run_tool error: {e}", "ERROR")

                if rid is not None:
                    self.observer.log_tool_end(rid, tool_name, output)

                if hasattr(self, "_tick_activity"):
                    self._tick_activity()
                return output

        # ------------------------------------------------------------------
        # 1) Regex fallback for single unquoted arg: func(foo bar)
        # ------------------------------------------------------------------
        m_simple = re.fullmatch(
            r'\s*([A-Za-z_]\w*)\(\s*([^)]+?)\s*\)\s*',
            tool_code,
            re.DOTALL
        )
        if m_simple:
            func_name, raw_arg = m_simple.group(1), m_simple.group(2)
            if '"' not in raw_arg and "'" not in raw_arg:
                func = getattr(Tools, func_name, None)
                if func:
                    log_message(f"run_tool: Fallback parse → {func_name}('{raw_arg}')", "DEBUG")
                    try:
                        result = func(raw_arg)
                        output = str(result)
                        log_message(f"run_tool: {func_name} returned {output!r}", "INFO")
                    except Exception as e:
                        output = f"Error executing `{func_name}`: {e}"
                        log_message(f"run_tool error: {e}", "ERROR")

                    if rid is not None:
                        self.observer.log_tool_end(rid, tool_name, output)

                    if hasattr(self, "_tick_activity"):
                        self._tick_activity()
                    return output

        # ------------------------------------------------------------------
        # 2) Full AST-based parsing for everything else
        # ------------------------------------------------------------------
        try:
            tree = ast.parse(tool_code.strip(), mode="eval")
            if not isinstance(tree, ast.Expression) or not isinstance(tree.body, ast.Call):
                raise ValueError("Not a function call")
            call = tree.body

            # function lookup
            if not isinstance(call.func, ast.Name):
                raise ValueError("Unsupported function expression")
            func_name = call.func.id
            func = getattr(Tools, func_name, None)
            if not func:
                raise NameError(f"Unknown tool `{func_name}`")

            # positional args
            args = []
            for arg in call.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.Name):
                    args.append(arg.id)
                else:
                    seg = ast.get_source_segment(tool_code, arg) or ""
                    args.append(seg.strip())

            # keyword args
            kwargs = {}
            for kw in call.keywords:
                if not kw.arg:
                    continue
                v = kw.value
                if isinstance(v, ast.Constant):
                    kwargs[kw.arg] = v.value
                elif isinstance(v, ast.Name):
                    kwargs[kw.arg] = v.id
                else:
                    seg = ast.get_source_segment(tool_code, v) or ""
                    kwargs[kw.arg] = seg.strip()

            # default for get_chat_history()
            if func_name == "get_chat_history" and not args and not kwargs:
                args = [5]

            log_message(f"run_tool: Calling {func_name} with args={args} kwargs={kwargs}", "DEBUG")

            sig = inspect.signature(func)
            pos_names = [p.name for p in sig.parameters.values()
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if not args and kwargs and all(n in kwargs for n in pos_names):
                args = [kwargs.pop(n) for n in pos_names]

            result = func(*args, **kwargs)
            output = str(result)
            log_message(f"run_tool: {func_name} returned {output!r}", "INFO")

        except Exception as e:
            output = f"Error executing `{tool_code}`: {e}"
            log_message(f"run_tool error: {e}", "ERROR")

        # ── OBSERVABILITY: emit tool_end ───────────────────────────────────
        if rid is not None:
            self.observer.log_tool_end(rid, tool_name, output)

        if hasattr(self, "_tick_activity"):
            self._tick_activity()
        return output


        
    def _stage_prompt_optimization(self, ctx):
        """
        Replace the primary system prompt for this run with the best one
        discovered so far.
        """
        base = self.config_manager.config.get("system", "")
        best = self.rl_manager.best_prompt(base)
        if best != base:
            ctx.setdefault("ctx_txt", "")
            ctx["ctx_txt"] += "\n[Prompt optimized]"
            self.config_manager.config["system"] = best
        return None
    
    def _stage_rl_experimentation(self, ctx):
        """
        With small probability, try a mutated prompt + stage list and mark the run
        as 'exploratory'.  Record reward at the end.
        """
        if random.random() < 0.2:   # 20 % of turns are exploration
            base_prompt  = self.config_manager.config.get("system", "")
            base_stages  = ctx["ALL_AGENTS"] or []   # rough proxy
            new_prompt, new_stages = self.rl_manager.propose(base_prompt, base_stages)
            ctx["explore_prompt"]  = new_prompt
            ctx["explore_stages"]  = new_stages
            self.config_manager.config["system"] = new_prompt
            # overwrite active stages for this run
            ctx["pipeline_overridden"] = True
            ctx["overridden_stages"]   = new_stages
            ctx["ctx_txt"] += "\n[RL exploration enabled]"
        return None
    
    def _stage_execute_tasks(self, ctx):
        """
        Parallel‐execute each non-done task in its own thread, measure each 
        mini-stage’s performance, and adjust prompts if needed.
        """

        # Helper to let handlers use ctx.attr and ctx["attr"]
        class TaskSubctx:
            def __init__(self, data):
                self.__dict__.update(data)
                self.stage_outputs = {}
                self.stage_counts = {}
            def __getitem__(self, key):
                return getattr(self, key)
            def __setitem__(self, key, value):
                setattr(self, key, value)
            def get(self, key, default=None):
                return getattr(self, key, default)

        tasks = ctx.get("global_tasks", [])
        if not tasks:
            return None

        def _run_one(task):
            # Build a plain dict of the subcontext’s initial state
            base = {
                "user_message":     f"Work on task: {task['text']}",
                "sender_role":      ctx.get("sender_role", "user"),
                "run_id":           ctx["run_id"],
                "skip_tts":         True,
                "ctx_txt":          "",
                "tool_summaries":   [],
                "assembled":        "",
                "final_response":   None,
                "workspace_memory": ctx["workspace_memory"],
                "ALL_TOOLS":        ctx["ALL_TOOLS"],
                "ALL_AGENTS":       ctx["ALL_AGENTS"],
                "global_tasks":     ctx["global_tasks"],
            }
            # Wrap it so handlers see attributes and dict‐style access
            subctx = TaskSubctx(base)

            mini_stages = [
                "context_analysis",
                "planning_summary",
                "tool_chaining",
                "assemble_prompt",
                "final_inference"
            ]

            for stage in mini_stages:
                handler = getattr(self, f"_stage_{stage}", None)
                if not handler:
                    continue

                start = time.time()
                success = True
                try:
                    # Call with or without prev_output arg
                    sig = inspect.signature(handler)
                    if len(sig.parameters) == 2:
                        handler(subctx, None)
                    else:
                        handler(subctx)
                except Exception:
                    success = False
                    # record timing, then re-raise to abort this subtask
                    raise
                finally:
                    duration = time.time() - start
                    self.prompt_evaluator.record(stage, duration, success)
                    new_sys = self.prompt_evaluator.adjust(
                        stage, self.config_manager.config["system"]
                    )
                    if new_sys:
                        self.config_manager.config["system"] = new_sys

            # On success (or after a failure), record and mark done
            result = subctx.get("final_response", "")
            ctx["ctx_txt"] += f"\n[Task {task.get('id')} done]: {result}"
            task["done"] = True

        # Run in parallel as before
        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as ex:
            futures = [ex.submit(_run_one, t) for t in tasks if not t.get("done")]
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    log_message(f"[execute_tasks] subtask failed: {e}", "ERROR")

        return None

    
    def _stage_define_criteria(self, ctx, planning_summary: str) -> list[str]:
        """
        Ask the model to translate the planning summary into a set of
        concrete, measurable success criteria.
        Returns a JSON array of criterion strings.
        """
        
        prompt = (
            "You are a Criteria Agent.  Based on this one-sentence plan:\n\n"
            f"    {planning_summary}\n\n"
            "Output a JSON list of 3–5 clear, discrete, measurable criteria "
            "that will tell us when the task is complete.  No extra text."
        )
        resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[{"role":"system","content":prompt}],
            stream=False
        )["message"]["content"].strip()
        try:
            criteria = json.loads(resp)
        except:
            criteria = []
        ctx.ctx_txt += f"\n[Defined Criteria]: {criteria}"
        ctx.stage_outputs["define_criteria"] = criteria
        return criteria


    # -------------------------------------------------------------------
    # New workspace‐memory helpers
    # -------------------------------------------------------------------
    def _load_workspace_memory(self) -> dict:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        path = os.path.join(WORKSPACE_DIR, "workspace_memory.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_workspace_memory(self, memory: dict) -> None:
        path = os.path.join(WORKSPACE_DIR, "workspace_memory.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
        except Exception as e:
            log_message(f"Failed to save workspace memory: {e}", "ERROR")

    # -------------------------------------------------------------------
    # Alias legacy "planning" stage
    # -------------------------------------------------------------------
    def _stage_planning(self, ctx):
        return self._stage_planning_summary(ctx)

    def _assemble_stage_list(self, user_message: str, base_stages: list[str]) -> list[str]:
        """
        Dynamically adjust the pipeline based on the user’s task.
        Example rules:
        - If the user asks for a summary, ensure 'summary_request' runs first.
        - If they mention tasks or subtasks, include task stages.
        - If they reference files, include file management/validation early.
        """

        stages = base_stages.copy()

        # 1) Summaries → bump summary_request to front
        if re.search(r'\b(summarize|summary|brief)\b', user_message, re.IGNORECASE):
            if 'summary_request' in stages:
                idx = stages.index('summary_request')
                stages.insert(0, stages.pop(idx))

        # 2) Task-related → append task stages if missing
        if re.search(r'\b(task|todo|subtask|assign)\b', user_message, re.IGNORECASE):
            for t in ('task_management', 'subtask_management'):
                if t not in stages:
                    stages.append(t)

        # 3) File-related → move file stages to front
        if re.search(r'\b(file|load|save|read|write)\b', user_message, re.IGNORECASE):
            for t in ('file_management', 'file_validation'):
                if t in stages:
                    idx = stages.index(t)
                    stages.insert(0, stages.pop(idx))

        return stages

    def new_request(self, user_message: str, sender_role: str = "user", skip_tts: bool = False) -> str:
        """
        Entry-point for a new user request.  
        Handles context-switching, per-context state, dynamic stage-list assembly,
        explicit data-flow between stages, and final response delivery.
        """

        # 0) Context-switch detection: "/topic NAME" or "/ctx NAME"
        m = re.match(r"^/(topic|ctx)\s+(.+)$", user_message.strip(), re.IGNORECASE)
        if m:
            topic = m.group(2).strip()
            self._activate_context(topic)
            return f"Switched to context '{topic}'."

        # 1) Grab or create current context
        ctx = getattr(self, "current_ctx", None)
        if ctx is None:
            self._activate_context("global")
            ctx = self.current_ctx

        # 1a) Rebind managers
        self.history_manager   = ctx.history_manager
        self.tts_manager       = ctx.tts_manager
        Tools._history_manager = ctx.history_manager

        # 2) Reset per-turn fields
        ctx.user_message     = user_message
        ctx.sender_role      = sender_role
        ctx.skip_tts         = skip_tts
        ctx.ctx_txt          = ""
        ctx.tool_summaries   = []
        ctx.assembled        = ""
        ctx.final_response   = None
        ctx.stage_counts.clear()
        ctx.stage_outputs    = {}   # ◀ new: will hold each stage’s output

        # 2a) Record user message (no TTS echo)
        ctx.history_manager.add_entry(sender_role, user_message)

        # 3) Telemetry
        run_id = self.observer.start_run(user_message)
        self.current_run_id = run_id
        ctx.run_id          = run_id

        # 4) Build & prune base stage list
        stack  = Tools.load_agent_stack()
        stages = list(stack.get("stages", []))
        if "record_user_message" in stages:
            stages.remove("record_user_message")

        # 4.5) Dynamic stage assembly
        stages = self._assemble_stage_list(user_message, stages)

        # 5) Ensure core stages in order
        def _ensure(name: str, after: str | None):
            if name not in stages and after and after in stages:
                stages.insert(stages.index(after) + 1, name)

        _ensure("context_analysis",             stages[0] if stages else None)
        _ensure("workspace_setup",              "context_analysis")
        _ensure("file_management",              "workspace_setup")
        _ensure("file_validation",              "file_management")
        _ensure("html_filtering",               "file_management")
        _ensure("chunk_and_summarize",          "html_filtering")
        _ensure("intent_clarification",         "context_analysis")
        _ensure("external_knowledge_retrieval", "intent_clarification")
        _ensure("memory_summarization",         "external_knowledge_retrieval")
        _ensure("planning_summary",             "memory_summarization")

        # CRITERIA → PLAN → EXECUTE → VERIFY
        _ensure("define_criteria",              "planning_summary")
        _ensure("task_decomposition",           "define_criteria")
        _ensure("plan_validation",              "task_decomposition")
        _ensure("execute_actions",              "plan_validation")
        _ensure("verify_results",               "execute_actions")

        # GLOBAL TASK LIST
        _ensure("task_management",              "verify_results")
        _ensure("subtask_management",           "task_management")
        _ensure("execute_tasks",                "subtask_management")

        # back to reply
        _ensure("tool_chaining",                "execute_tasks")
        _ensure("assemble_prompt",              "tool_chaining")
        _ensure("final_inference",              "assemble_prompt")
        _ensure("chain_of_thought",             "final_inference")
        _ensure("notification_audit",           "chain_of_thought")
        _ensure("flow_health_check",            "notification_audit")


        # 6) Insert adversarial_loop after final_inference if missing
        if "final_inference" in stages and "adversarial_loop" not in stages:
            idx = stages.index("final_inference") + 1
            stages.insert(idx, "adversarial_loop")

        # 7) Define explicit one-up dependency map
        deps = {
            "context_analysis":             None,
            "intent_clarification":         "context_analysis",
            "workspace_setup":              None,
            "file_management":              "workspace_setup",
            "file_validation":              "file_management",
            "html_filtering":               "file_management",
            "chunk_and_summarize":          "html_filtering",
            "external_knowledge_retrieval": "intent_clarification",
            "memory_summarization":         "external_knowledge_retrieval",
            "planning_summary":             "memory_summarization",
            "define_criteria":              "planning_summary",
            "task_decomposition":           "define_criteria",
            "plan_validation":              "task_decomposition",
            "execute_actions":              "plan_validation",
            "verify_results":               "execute_actions",
            "task_management":              "verify_results",
            "subtask_management":           "task_management",
            "execute_tasks":                "subtask_management",
            "tool_chaining":                "execute_tasks",
            "assemble_prompt":              "tool_chaining",
            "final_inference":              "assemble_prompt",
            "adversarial_loop":             "final_inference",
            "chain_of_thought":             "final_inference",
            "notification_audit":           "chain_of_thought",
            "flow_health_check":            "notification_audit",
        }



        log_message(f"[Context={ctx.name}] Pipeline stages: {stages}", "INFO")

        for stage in stages:
            # runaway guard
            cnt = ctx.stage_counts.get(stage, 0) + 1
            ctx.stage_counts[stage] = cnt
            if cnt > 1:
                log_message(f"[{ctx.name}] Skipping repeated stage '{stage}'", "WARNING")
                continue

            handler = getattr(self, f"_stage_{stage}", None)
            if handler is None:
                continue

            self.observer.log_stage_start(run_id, stage)

            # fetch the single dependency’s output (if any)
            dep = deps.get(stage)
            prev_output = ctx.stage_outputs.get(dep) if dep else None

            try:
                sig = inspect.signature(handler)
                # bound method: parameters are (ctx, prev_output) for new-style handlers
                if len(sig.parameters) == 2:
                    result = handler(ctx, prev_output)
                else:
                    result = handler(ctx)
            except Exception as e:
                self.observer.log_error(run_id, stage, e)
                log_message(f"[{ctx.name}] Stage '{stage}' failed: {e}", "ERROR")
                # retry logic for final_inference
                if stage == "final_inference" and getattr(self, "_temp_bump", 0.0) == 0.0:
                    log_message("Retrying final_inference with bumped temp", "INFO")
                    self._temp_bump = 0.2
                    try:
                        result = handler(ctx, prev_output) if len(sig.parameters)==2 else handler(ctx)

                        # If the stage gave back text, run the “code-fence executor”
                        if isinstance(result, str) and result:
                            result = self._inject_python_results(result)


                        # record non-empty string outputs
                        if isinstance(result, str) and result:
                            ctx.stage_outputs[stage] = result

                    except Exception:
                        result = ""
                    finally:
                        self._temp_bump = 0.0
            else:
                self.observer.log_stage_end(run_id, stage)

            # record non-empty string outputs
            if isinstance(result, str) and result:
                ctx.stage_outputs[stage] = result

            # early exit for summary_request
            if stage == "summary_request" and isinstance(result, str) and result:
                return result
            if stage == "timeframe_history_query" and isinstance(result, str) and result:
                ctx.ctx_txt += "\n[History lookup]:\n" + result
                continue
            # adversarial_loop runs even after final_inference

        # 8) Wrap up
        self._save_workspace_memory(ctx.workspace_memory)
        self.observer.complete_run(run_id)

        return ctx.final_response or "Sorry, I couldn't process that."



    def _stage_summary_request(self, ctx: Context):

        # use the Context field, not dict access
        msg = ctx.user_message.lower()
        if re.search(r'\b(summary|summarize|how was our conversation|feedback)\b', msg):
            # pick a period from the message, default to "today"
            m = re.search(r'\b(today|yesterday|last\s+\d+\s+days?)\b', msg)
            period = m.group(0) if m else "today"
            # fetch and parse history
            hist = Tools.get_chat_history(period)
            try:
                entries = json.loads(hist).get("results", [])
            except Exception:
                entries = []
            lines = "\n".join(f"{e['role'].capitalize()}: {e['content']}" for e in entries)

            prompt = (
                "Here is our conversation:\n\n"
                f"{lines}\n\n"
                "Please provide a brief, high-level summary of what we discussed."
            )
            summary = Tools.secondary_agent_tool(prompt, temperature=0.5).strip()
            # short-circuit: set final_response too
            ctx.final_response = summary
            return summary

        return None

    # -------------------------------------------------------------------
    # Stage: task_management
    # -------------------------------------------------------------------
    def _stage_task_management(self, ctx):
        """
        Let the secondary model decide whether to add/update/remove/list tasks.
        Any successful Tools.* call is run and then recorded in ctx['global_tasks']
        and in ctx['ctx_txt'].
        """
        prompt = (
            "You may manage the GLOBAL TASK LIST.  Use exactly one Tools call from:\n"
            "  • add_task(text)\n"
            "  • update_task(id, text)\n"
            "  • remove_task(id)\n"
            "  • list_tasks()\n"
            "If no task action is needed, respond with NO_OP."
        )
        decision = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[
                {"role": "system",  "content": prompt},
                {"role": "user",    "content": ctx["user_message"]}
            ],
            stream=False
        )["message"]["content"].strip()

        call = Tools.parse_tool_call(decision)
        if call and call.upper() != "NO_OP":
            out = self.run_tool(call)
            # refresh our in-memory view
            try:
                ctx["global_tasks"] = json.loads(Tools.list_tasks())
            except:
                pass
            ctx["ctx_txt"] += f"\n[TASK_OP] {call} → {out}"
        return None

    # ------------------------------
    # Stage 1: timeframe_history_query
    # ------------------------------
    def _stage_timeframe_history_query(self, ctx: Context):
        out = self._history_timeframe_query(ctx.user_message)
        if out:
            ctx.tts_manager.enqueue(out)
            ctx.ctx_txt += "\n[History lookup]:\n" + out
        return None

    # ------------------------------
    # Stage 2: record_user_message
    # ------------------------------
    def _stage_record_user_message(self, ctx):
        um = ctx["user_message"]
        role = ctx.get("sender_role", "user")   # ← respect caller
        self.history_manager.add_entry(role, um)
        _ = Utils.embed_text(um)
        return None

    # ------------------------------
    # Stage: context_analysis
    # ------------------------------
    def _stage_context_analysis(self, ctx: Context, prev_output: str | None = None) -> str:
        """
        Perform a focused context analysis of the user’s latest message.
        Returns the analysis text.
        """
        buf = []
        for tok, done in self._stream_context(ctx.user_message):   # ← unchanged call
            buf.append(tok)
            if done:
                break
        ca = "".join(buf).strip()
        log_message(f"Context analysis: {ca!r}", "DEBUG")

        # record it for both human‐readability and downstream stages
        ctx.ctx_txt += ca + "\n"
        return ca



    # ------------------------------
    # Stage: intent_clarification
    # ------------------------------
    def _stage_intent_clarification(self, ctx: Context, context_analysis: str) -> str:
        """
        If the user’s request is ambiguous, first try to resolve it internally.
        Only if that fails do we ask the user.
        Returns:
          - ""              if no clarification is needed or we auto‐resolved
          - the clarification question to speak if we must ask the user
        """
        # 1) Do we even need to clarify?
        clar = self._clarify_intent(context_analysis)
        if not clar:
            return ""

        # 2) Try to self‐answer
        auto_sys = (
            "You are the AUTOCLARIFIER AGENT.  "
            "Answer the question below using only the existing conversation history.  "
            "If you can answer it, respond with that answer.  "
            "Otherwise respond with UNANSWERABLE."
        )
        auto_resp = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[
                {"role": "system",  "content": auto_sys},
                {"role": "user",    "content": clar}
            ],
            stream=False
        )["message"]["content"].strip()

        # 3) Log what happened
        ctx.ctx_txt += f"[Self-clarification] Q={clar!r} → A={auto_resp!r}\n"

        # 4) If we succeeded, inject the resolution into the user_message
        if auto_resp and auto_resp.upper() != "UNANSWERABLE":
            ctx.user_message = f"{ctx.user_message} {auto_resp}"
            return ""

        # 5) Otherwise we really need to ask the user
        self.history_manager.add_entry("assistant", clar)
        if not ctx.skip_tts:
            self.tts_manager.enqueue(clar)

        return clar


    # ------------------------------
    # Stage 5: external_knowledge_retrieval
    # ------------------------------
    def _stage_external_knowledge_retrieval(self, ctx) -> str:
        facts = self._fetch_external_knowledge(ctx.user_message)
        if facts:
            ctx.ctx_txt += "\n" + facts
            return facts
        return ""


    # ------------------------------
    # Stage 6: memory_summarization
    # ------------------------------
    def _stage_memory_summarization(self, ctx: Context) -> str:
        summary = self._memory_summarize()
        if summary:
            ctx.ctx_txt += "\nMemory summary:\n" + summary
            return summary
        return ""


    # ------------------------------
    # Stage 7: planning_summary
    # ------------------------------
    def _stage_planning_summary(self, ctx) -> str:
        """
        • Streams the *tool-decision* output from `_stream_tool`.
        • Extracts the first valid `tool_code` invocation and stores it on
          the context as `ctx.next_tool_call`.
        • Sets `ctx.needs_tool_work` ⇢ True if the function name is **not**
          yet available on `Tools` (triggers the later self-improvement loop).
        • Emits a one-sentence, user-friendly planning summary (and optional
          TTS).
        """
        import re, inspect

        # 1️⃣  Run the tool-decision stream and capture its raw output
        peek: list[str] = []
        for tok, done in self._stream_tool(ctx):
            self._tick_activity()          # keep idle-monitor happy
            peek.append(tok)
            if done:
                break

        raw_block = re.sub(
            r"^```tool_code\s*|\s*```$",
            "",
            "".join(peek),
            flags=re.DOTALL,
        ).strip().strip("`")

        call: str | None = Tools.parse_tool_call(raw_block)
        ctx.next_tool_call = call                # make it available downstream

        # 2️⃣  Decide whether the tool exists already
        tool_names = [
            name for name, fn in inspect.getmembers(Tools, inspect.isfunction)
            if not name.startswith("_")
        ]
        if call:
            fn_name = call.split("(", 1)[0].strip()
            ctx.needs_tool_work = fn_name not in tool_names
        else:
            ctx.needs_tool_work = False

        # 3️⃣  Produce a user-facing planning sentence
        if call:
            plan_msg = chat(
                model=self.config_manager.config["secondary_model"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Planning Agent. "
                            "Reply with ONE casual sentence that explains "
                            "— at a very high level — what the upcoming tool "
                            "call will do. No inner thoughts, no code fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The user said: “{ctx.user_message}”. "
                            f"We are about to run `{call}`."
                        ),
                    },
                ],
                stream=False,
            )["message"]["content"].strip()

            log_message(f"Planning summary: {plan_msg}", "INFO")

            if plan_msg and not getattr(ctx, "skip_tts", False):
                self.tts_manager.enqueue(plan_msg)

            ctx.ctx_txt += f"\n[Planning summary] {plan_msg}"
            return plan_msg

        # 4️⃣  Nothing was selected
        ctx.ctx_txt += "\n[Planning summary] (no tool selected)"
        log_message("Planning summary: no tool selected", "WARNING")
        return ""




    # ------------------------------
    # Stage 8: tool_chaining
    # ------------------------------
    def _stage_tool_chaining(self, ctx):
        """
        1) Detect and run any “Next Action: Call `func()`” suggestion.
        2) Otherwise do the normal streaming tool-chaining.
        """
        

        run_id = ctx["run_id"]

        # 1) Auto‐invoke any Next Action suggestion
        m = re.search(r'Next Action:\s*Call\s*`([^`]+)`', ctx["ctx_txt"])
        if m:
            code = m.group(1)
            log_message(f"Auto-invoking suggested tool: {code}", "INFO")
            try:
                out = self.run_tool(code)
                ctx["ctx_txt"] += f"\n[Auto-run {code} → {out}]"
            except Exception as e:
                ctx["ctx_txt"] += f"\n[Suggestion {code} failed: {e}]"
                log_message(f"Suggestion {code} failed: {e}", "ERROR")
            # skip further chaining this turn
            return None

        # 2) Existing streaming‐based tool chaining
        summaries = []
        invoked   = set()
        tc        = ctx["ctx_txt"]

        for i in range(5):
            buf = []
            try:
                for tok, done in self._stream_tool(tc):
                    buf.append(tok)
                    if done:
                        break
            except Exception as e:
                log_message(f"Tool‐streaming failed: {e}", "ERROR")
                break

            raw = re.sub(r"^```tool_code\s*|\s*```$", "", "".join(buf), flags=re.DOTALL).strip()
            code = Tools.parse_tool_call(raw)
            if not code or code.upper() == "NO_TOOL":
                break

            fn = code.split("(")[0]
            if fn in invoked:
                break
            invoked.add(fn)

            log_message(f"Invoking via tool_chaining: {code}", "INFO")
            try:
                out = self.run_tool(code)
            except Exception as e:
                log_message(f"run_tool error on {code}: {e}", "ERROR")
                break

            # summarize that output
            summary = None
            try:
                data = json.loads(out)
                if fn in ("search_internet", "brave_search"):
                    lines = [f"- {r.get('title','')} ({r.get('url','')})"
                            for r in data["web"]["results"][:3]]
                    summary = "Top results:\n" + "\n".join(lines)
                elif fn == "get_current_location":
                    summary = f"Location: {data.get('city')}, {data.get('regionName')}"
            except:
                pass
            if summary is None:
                summary = f"{fn} → {out}"
            summaries.append(summary)
            tc += "\n" + summary

        if summaries:
            ctx["tool_summaries"] = summaries
            ctx["ctx_txt"] += "\n" + "\n\n".join(summaries)
        return None
    # ────────────────────────────────────────────────


    # ------------------------------
    # Stage 9: assemble_prompt
    # ------------------------------
    def _stage_assemble_prompt(self, ctx):
        assembled = f"```context_analysis\n{ctx['ctx_txt'].strip()}\n```"
        if ctx["tool_summaries"]:
            assembled += "\n```tool_output\n" + "\n\n".join(ctx["tool_summaries"]) + "\n```"
        assembled += "\n" + ctx["user_message"]
        # record & embed
        self.history_manager.add_entry("user", assembled)
        _ = Utils.embed_text(assembled)
        log_message(f"Final prompt prepared", "DEBUG")
        ctx["assembled"] = assembled
        return None

    # ------------------------------
    # Stage 10: final_inference
    # ------------------------------
    def _stage_final_inference(self, ctx):
        resp = self.run_inference(ctx["assembled"], ctx["skip_tts"])
        # clean markup
        clean = re.sub(r"```.*?```", "", resp, flags=re.DOTALL)
        clean = clean.replace("`","")
        clean = re.sub(r"[*_]","", clean).strip()
        # record
        self.history_manager.add_entry("assistant", clean)
        try:
            session_log.write(json.dumps({
                "role":"assistant",
                "content": clean,
                "timestamp": datetime.now().isoformat()
            })+"\n")
            session_log.flush()
        except:
            pass
        ctx["final_response"] = clean
        return None
    # -------------------------------------------------------------------
    # New workspace-file stages
    # -------------------------------------------------------------------
    def _stage_workspace_setup(self, ctx):
        """Ensure the workspace folder exists."""
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        log_message(f"Workspace ready at {WORKSPACE_DIR}", "DEBUG")
        return None

    def _stage_file_management(self, ctx):
        """
        Ask the secondary model whether to do a file operation.
        If so, run it and record it in workspace_memory.
        """
        prompt = (
            "You may create, append to, delete or list files in the workspace.  "
            "Use exactly one Tools call from:\n"
            "  • create_file(filename, content)\n"
            "  • append_file(filename, content)\n"
            "  • delete_file(filename)\n"
            "  • list_workspace()\n"
            "If no file action is needed, respond with NO_OP."
        )
        decision = chat(
            model=self.config_manager.config["secondary_model"],
            messages=[
                {"role":"system",  "content": prompt},
                {"role":"user",    "content": ctx["user_message"]}
            ],
            stream=False
        )["message"]["content"].strip()

        call = Tools.parse_tool_call(decision)
        if call and call.upper() != "NO_OP":
            result = self.run_tool(call)
            # track it
            mem = ctx["workspace_memory"]
            mem.setdefault("operations", []).append({
                "call":      call,
                "result":    result,
                "timestamp": datetime.now().isoformat()
            })
            ctx["ctx_txt"] += f"\n[File op] {call} → {result}"
        return None

    def _stage_workspace_save(self, ctx):
        """Persist our in-memory record of workspace ops to disk."""
        self._save_workspace_memory(ctx["workspace_memory"])
        return None
    # ------------------------------
    # Stage 11: chain_of_thought
    # ------------------------------
    def _stage_chain_of_thought(self, ctx):
        external_facts   = ctx.stage_outputs.get("external_knowledge_retrieval", "")
        memory_summary   = ctx.stage_outputs.get("memory_summarization", "")
        planning_summary = ctx.stage_outputs.get("planning_summary", "")
        cot = self._chain_of_thought(
            user_message=ctx.user_message,
            context_analysis=ctx.ctx_txt,
            external_facts=external_facts,
            memory_summary=memory_summary,
            planning_summary=planning_summary,
            tool_summaries=ctx.tool_summaries,
            final_response=ctx.final_response
        )
        try:
            path = os.path.join(session_folder, "thoughts.json")
            bag  = json.load(open(path)) if os.path.exists(path) else []
            bag.append({"timestamp": datetime.now().isoformat(), "chain_of_thought": cot})
            with open(path, "w") as f:
                json.dump(bag, f, indent=2)
        except:
            pass
        return None


    # ------------------------------
    # Stage 12: notification_audit
    # ------------------------------
    def _stage_notification_audit(self, ctx):
        self._emit_event("assistant_response", {
            "input":  ctx["user_message"],
            "output": ctx["final_response"]
        })
        # this is our final return value
        return ctx["final_response"]

def voice_to_llm_loop(chat_manager: ChatManager, playback_lock, output_stream):

    log_message("Voice-to-LLM loop started. Waiting for speech...", "INFO")
    last_response    = None
    max_words        = config.get("max_response_words", 1000)
    rms_threshold    = config.get("rms_threshold", 0.01)
    silence_duration = 2.0  # seconds of sustained silence → end of speech

    while True:
        # 1) Capture until silence
        chunk = audio_queue.get(); audio_queue.task_done()
        buffer = [chunk]; silence_start = None
        #log_message("Recording audio until silence detected...", "DEBUG")
        while True:
            chunk = audio_queue.get(); audio_queue.task_done()
            buffer.append(chunk)
            rms = np.sqrt(np.mean(chunk.flatten()**2))
            if rms >= rms_threshold:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    break

        #log_message(f"Silence for {silence_duration}s; processing audio...", "INFO")
        audio_array = np.concatenate(buffer, axis=0).flatten().astype(np.float32)

        # 2) Optional enhancement
        if config.get("enable_noise_reduction", False):
            log_message("Enhancing audio (denoise, EQ, compression)...", "PROCESS")
            audio_array = apply_eq_and_denoise(audio_array, SAMPLE_RATE)
            log_message("Audio enhancement complete.", "SUCCESS")

        # 3) Transcribe via consensus helper
        #log_message("Transcribing via consensus helper...", "PROCESS")
        transcription = consensus_whisper_transcribe_helper(
            audio_array,
            language="en",
            rms_threshold=rms_threshold,
            consensus_threshold=config.get("consensus_threshold", 0.8)
        )
        if not transcription or not validate_transcription(transcription):
            #log_message("Invalid transcription; skipping.", "WARNING")
            continue

        labeled = f"{transcription}"
        log_message(f"Transcribed prompt: {labeled}", "INFO")

        # 4) Log user turn & clear pending TTS
        session_log.write(json.dumps({
            "role":    "user",
            "content": labeled,
            "timestamp": datetime.now().isoformat()
        }) + "\n")
        session_log.flush()
        flush_current_tts()

        # 5) Send through new_request (TTS happens inside)
        response = chat_manager.new_request(labeled, skip_tts=False)
        if response is None:
            log_message("new_request returned None; skipping.", "WARNING")
            continue

        # 6) Clean up any fences/markup
        clean_resp = re.sub(r"```(?:tool_output|tool_code|context_analysis).*?```", "",
                            response, flags=re.DOTALL)
        clean_resp = clean_resp.replace("`", "")
        clean_resp = re.sub(r"[*_]", "", clean_resp).strip()

        # 7) Dedupe & length check
        if not clean_resp or clean_resp == last_response:
            continue
        if len(clean_resp.split()) > max_words:
            log_message(f"Hallucination detected (> {max_words} words); discarding.", "WARNING")
            last_response = None
            continue

        last_response = clean_resp
        log_message(f"LLM response ready: {clean_resp}", "INFO")

        # 8) No extra enqueue: new_request has already enqueued the final response
        log_message("Ready for next voice input...", "INFO")

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
                        # ── SHUT DOWN FLASK SERVER ───────────────────────────
                        srv = globals().get("_flask_server")
                        if srv:
                            log_message("Shutting down existing Flask server...", "INFO")
                            try:
                                srv.shutdown()
                            except Exception as e:
                                log_message(f"Error shutting down Flask server: {e}", "WARNING")
                        # ── RE-EXEC THE PROCESS ───────────────────────────────
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception:
                    continue


    threading.Thread(target=_monitor_files, daemon=True).start()
    log_message("Main function starting.", "INFO")

    # Start the TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
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
    config_manager  = ConfigManager(config)
    history_manager = HistoryManager()
    tts_manager     = TTSManager()
    memory_manager  = MemoryManager()
    mode_manager    = ModeManager()

    # ─── Build ChatManager (make it global for SSE) ────────────────────
    global chat_manager
    chat_manager = ChatManager(
        config_manager,
        history_manager,
        tts_manager,
        tools_data=Tools.load_agent_stack().get("tools", []),
        format_schema=Utils.load_format_schema(config.get("options", "")),
        memory_manager=memory_manager,
        mode_manager=mode_manager
    )

    # Launch voice loop (pass in playback primitives)
    voice_thread = threading.Thread(
        target=voice_to_llm_loop,
        args=(chat_manager, playback_lock, output_stream),
        daemon=True
    )
    voice_thread.start()
    log_message("Main: Voice-to-LLM loop thread started.", "DEBUG")

    # Launch text-override loop
    text_thread = threading.Thread(
        target=text_input_loop,
        args=(chat_manager,),
        daemon=True
    )
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
    # text_thread may remain blocked on input()
    session_log.close()

    # Save chat history
    history_path = os.path.join(session_folder, "chat_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_manager.history, f, indent=4)

    log_message(f"Chat session saved in folder: {session_folder}", "SUCCESS")
    log_message("Main: System terminated.", "INFO")
    print("Chat session saved in folder:", session_folder)
    print("System terminated.")


# ─── START OBSERVABILITY DASHBOARD & SSE SERVER ───────────────────────

app = Flask(__name__)
CORS(app)

# 1) SSE endpoint
@app.route("/stream")
def stream():
    def event_stream():
        q = []
        cm = globals().get("chat_manager")
        if cm:
            cm.observer.subscribe(lambda data: q.append(data))
        while True:
            if q:
                yield f"data: {q.pop(0)}\n\n"
            else:
                time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

# 2) Tweak endpoint (optional)
@app.route("/api/tweak", methods=["POST"])
def tweak():
    p = request.json or {}
    cm = globals().get("chat_manager")
    if cm:
        cm.observer._emit({
            "runId":     p.get("runId"),
            "type":      "tweak_applied",
            "stage":     p.get("stage"),
            "newData":   p.get("newData"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    return {"status": "ok"}

# 3) In-browser dashboard
INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Observable Agent Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Doto:wght@100..900&family=Overpass+Mono:wght@300..700&display=swap');
    :root {
      --bg: #121212;
      --fg: rgba(255,255,255,0.8);
      --border: #333;
      --panel-bg: rgba(255,255,255,0.05);
    }
    * { box-sizing: border-box; }
    body {
      margin:0; padding:0;
      background: var(--bg);
      color: var(--fg);
      font-family: 'Doto', sans-serif;
    }
    #grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      height: 100vh;
    }
    .col {
      overflow-y: auto;
      padding: 8px;
      border-right: 1px solid var(--border);
    }
    .col:last-child { border-right: none; }
    details {
      background: var(--panel-bg);
      margin: 6px 0;
      padding: 6px;
      border-radius: 4px;
    }
    summary {
      cursor: pointer;
      font-family: 'Overpass Mono', monospace;
      font-weight: 500;
      font-size: 0.95em;
    }
    .loading {
      font-style: italic;
      opacity: 0.6;
    }
    /* draggable registry items */
    .registry .stage-item {
      background: var(--panel-bg);
      margin:4px 0;
      padding:4px;
      border:1px solid var(--border);
      border-radius:4px;
      cursor: grab;
      user-select: none;
    }
    .registry .drag-over {
      box-shadow: inset 0 0 8px gold;
    }
  </style>
</head>
<body>
  <div id="grid">
    <section id="col-stages" class="col"></section>
    <section id="col-tools"  class="col"></section>
    <section id="col-registry" class="col registry"></section>
  </div>
  <script>
    // utility: hash a string to an HSL color
    function colorFor(name) {
      let hash=0; for(let c of name) hash=(hash<<5)-hash+c.charCodeAt(0)|0;
      let h = ((hash%360)+360)%360;
      return `hsl(${h}, 60%, 50%)`;
    }

    // 1) Registry: fetch & render stages as draggable list
    const reg = document.getElementById("col-registry");
    fetch("/api/stages")
      .then(r=>r.json())
      .then(({stages})=>{
        stages.forEach((s,i)=>{
          let d=document.createElement("div");
          d.textContent=s;
          d.id="reg-"+i;
          d.draggable=true;
          d.className="stage-item";
          d.style.color=colorFor(s);
          d.dataset.stage=s;
          reg.appendChild(d);
        });
      });

    // drag & drop handlers
    let dragSrc=null;
    reg.addEventListener("dragstart", e=>{
      dragSrc=e.target;
      e.dataTransfer.setData("text/plain","");
      e.target.style.opacity=0.5;
    });
    reg.addEventListener("dragover", e=>{
      e.preventDefault();
      let over=e.target.closest(".stage-item");
      reg.querySelectorAll(".stage-item").forEach(el=>el.classList.remove("drag-over"));
      if(over) over.classList.add("drag-over");
    });
    reg.addEventListener("dragleave", e=>{
      e.target.closest(".stage-item")?.classList.remove("drag-over");
    });
    reg.addEventListener("drop", e=>{
      e.preventDefault();
      let over=e.target.closest(".stage-item");
      if(over && dragSrc && dragSrc!==over) {
        reg.insertBefore(dragSrc, over.nextSibling);
        updateOrder();
      }
      reg.querySelectorAll(".stage-item").forEach(el=>el.classList.remove("drag-over"));
    });
    reg.addEventListener("dragend", e=>{
      e.target.style.opacity=1;
    });
    function updateOrder(){
      let newStages = Array.from(reg.children).map(d=>d.dataset.stage);
      fetch("/api/stages", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({stages:newStages})
      });
    }

    // cache references
    const colStages = document.getElementById("col-stages");
    const colTools  = document.getElementById("col-tools");

    // 2) SSE → handle events
    const es = new EventSource("/stream");
    es.onmessage = e=>{
      let ev = JSON.parse(e.data);
      handleEvent(ev);
      // persist
      let saved = JSON.parse(localStorage.getItem("obs_events")||"[]");
      saved.push(e.data);
      if(saved.length>500) saved.shift();
      localStorage.setItem("obs_events", JSON.stringify(saved));
    };
    // replay on load
    JSON.parse(localStorage.getItem("obs_events")||"[]")
      .forEach(d=>handleEvent(JSON.parse(d)));

    function handleEvent(e){
      // STAGE START
      if(e.type==="stage_start"){
        let details = document.createElement("details");
        details.id = `stage-${e.runId}-${e.stage}`;
        details.style.borderLeft = `4px solid ${colorFor(e.stage)}`;
        let sum = document.createElement("summary");
        sum.textContent = `[${e.runId}] ▶ ${e.stage}`;
        details.appendChild(sum);
        let content = document.createElement("div");
        content.className = "stage-content";
        details.appendChild(content);
        colStages.appendChild(details);
        colStages.scrollTop = colStages.scrollHeight;
      }
      // STAGE STREAM (token)
      if(e.type==="stage_stream"){
        let container = document.querySelector(`#stage-${e.runId}-${e.stage} .stage-content`);
        if(container){
          let span = document.createElement("span");
          span.textContent = e.token;
          container.appendChild(span);
          // ensure loading indicator
          if(!container.querySelector(".loading")){
            let load = document.createElement("div");
            load.className = "loading";
            load.textContent = "⏳";
            container.appendChild(load);
          }
          container.parentElement.open = true;
          container.scrollTop = container.scrollHeight;
        }
      }
      // STAGE END
      if(e.type==="stage_end"){
        let details = document.getElementById(`stage-${e.runId}-${e.stage}`);
        if(details){
          details.querySelectorAll(".loading").forEach(n=>n.remove());
          let done = document.createElement("div");
          done.style.opacity = 0.6;
          done.textContent = "✅ done";
          details.querySelector(".stage-content").appendChild(done);
        }
      }
      // TOOL events
      if(e.type==="tool_start"||e.type==="tool_end"){
        let div = document.createElement("div");
        div.className="tool";
        if(e.type==="tool_start"){
          div.textContent = `[${e.runId}] 🔄 ${e.tool}`;
        } else {
          div.textContent = `[${e.runId}] ✅ ${e.tool} ⇒ ${e.output}`;
        }
        colTools.appendChild(div);
        colTools.scrollTop = colTools.scrollHeight;
      }
    }
  </script>
</body>
</html>"""

# ─── DASHBOARD ENDPOINTS & FLASK LAUNCH ────────────────────────────────

@app.route("/")
def index():
    tools = Tools.load_agent_stack().get("tools", [])
    return render_template_string(INDEX_HTML, tools=tools)

# GET current pipeline stages
@app.route("/api/stages", methods=["GET"])
def get_stages():
    stages = Tools.load_agent_stack().get("stages", [])
    return jsonify({"stages": stages})

# POST a new ordering → update agent_stack.json
@app.route("/api/stages", methods=["POST"])
def update_stages():
    data = request.get_json() or {}
    new_stages = data.get("stages", [])
    try:
        Tools.update_agent_stack(
            {"stages": new_stages},
            justification=f"reordered via UI at {datetime.now(timezone.utc).isoformat()}"
        )
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

def _run_flask():
    """
    Launch Flask via a manually created server.  Mark its socket
    O_CLOEXEC so it’s closed on execv, letting the next process
    re-bind port 5000 cleanly.
    """
    host, port = "0.0.0.0", 5000

    while True:
        try:
            from werkzeug.serving import make_server
            server = make_server(host, port, app, threaded=True)

            # ensure the listening socket is closed on exec:
            server.socket.set_inheritable(False)

            # save a reference so monitor can shut it down:
            globals()["_flask_server"] = server

            log_message(f"Flask SSE server listening on http://{host}:{port}", "INFO")
            server.serve_forever()
            break
        except OSError as e:
            if "Address already in use" in str(e):
                log_message("Port 5000 still in use—retrying in 0.5s...", "WARNING")
                time.sleep(0.5)
                continue
            else:
                raise

# Only start Flask once per process (even across execv reloads)
if not globals().get("_flask_thread_started"):
    globals()["_flask_thread_started"] = True
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()
    print("🚀 Observable dashboard: http://localhost:5000")

if __name__ == "__main__":
    main()

