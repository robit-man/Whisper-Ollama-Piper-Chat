#!/usr/bin/env python
"""
Voice-activated LLM Chat via Whisper & Piper TTS with Config and Session Logging
----------------------------------------------------------------------------------
This script does the following:
  1. Bootstraps a virtual environment ("whisper_env") if not already running inside one,
     installs required dependencies, and re-launches itself in the venv.
  2. Checks for the Piper executable and ONNX files. If missing, it determines the correct release
     (ignoring platform-specific folder names) from
         https://github.com/rhasspy/piper/releases/download/2023.11.14-2/
     and downloads & extracts it into a folder named "piper" (in the same directory as this script).
     It also downloads "glados_piper_medium.onnx.json" and "glados_piper_medium.onnx" if they are missing.
  3. Loads configuration from "config.json" (if it exists) or creates it with default values.
     The configuration includes:
         - ollama_model (default "gemma3:4b")
         - onnx_json (default "glados_piper_medium.onnx.json")
         - onnx_model (default "glados_piper_medium.onnx")
         - whisper_model (default "base")
  4. Creates a new chat session folder under "chat_sessions" named with the startup datetime,
     and all chat messages are logged to "session.txt" inside that folder.
  5. Continuously captures microphone input and buffers the audio. Every few seconds (if the RMS
     indicates meaningful input), the buffered audio is transcribed using Whisper.
  6. The transcription (appended to a per-session chat history) is sent as context to the default
     LLM via the Ollama API (with streaming enabled). If the default model does not exist, it is
     pulled automatically.
  7. The streaming LLM response is accumulated and split into chunks by sentence-ending delimiters.
     Emojis and asterisks are removed from each chunk before being enqueued for TTS.
  8. The TTS worker processes the queue by launching the Piper executable (using the ONNX files)
     via JSON input, piping its raw audio output directly to aplay.
  9. If new voice input is detected while there is an active TTS session, the current TTS queue and
     playback are flushed to allow interruption and seamless transition to the new topic.
     
Press Ctrl+C at any time to exit.
"""

import sys, os, subprocess, platform, re, json, time, threading, queue, datetime

# ----- VENV BOOTSTRAP CODE -----
def in_virtualenv():
    return sys.prefix != sys.base_prefix

def create_and_activate_venv():
    venv_dir = os.path.join(os.getcwd(), "whisper_env")
    if not os.path.isdir(venv_dir):
        print("Virtual environment 'whisper_env' not found. Creating it...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("Virtual environment created.")
    if not in_virtualenv():
        new_python = (os.path.join(venv_dir, "Scripts", "python.exe")
                      if os.name == "nt" else os.path.join(venv_dir, "bin", "python"))
        print("Re-launching script inside the virtual environment...")
        subprocess.check_call([new_python] + sys.argv)
        sys.exit()

if not in_virtualenv():
    create_and_activate_venv()

# ----- Automatic Piper and ONNX Files Setup -----
def setup_piper_and_onnx():
    """
    Check if the Piper executable and ONNX files are available.
    If missing, download the appropriate Piper release from the GitHub releases
    and extract it into a folder named "piper" (in the same directory as this script).
    Also download the ONNX files if they are missing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe = os.path.join(piper_folder, "piper")
    
    # Determine the correct release filename based on OS/architecture.
    os_name = platform.system()
    machine = platform.machine()
    release_filename = ""
    if os_name == "Linux":
        if machine == "x86_64":
            release_filename = "piper_linux_x86_64.tar.gz"
        elif machine == "aarch64":
            release_filename = "piper_linux_aarch64.tar.gz"
        elif machine == "armv7l":
            release_filename = "piper_linux_armv7l.tar.gz"
        else:
            print("Unsupported Linux architecture:", machine)
            sys.exit(1)
    elif os_name == "Darwin":
        if machine in ("arm64", "aarch64"):
            release_filename = "piper_macos_aarch64.tar.gz"
        elif machine in ("x86_64", "AMD64"):
            release_filename = "piper_macos_x64.tar.gz"
        else:
            print("Unsupported macOS architecture:", machine)
            sys.exit(1)
    elif os_name == "Windows":
        release_filename = "piper_windows_amd64.zip"
    else:
        print("Unsupported OS:", os_name)
        sys.exit(1)
    
    if not os.path.isfile(piper_exe):
        print(f"Piper executable not found at {piper_exe}.")
        download_url = f"https://github.com/rhasspy/piper/releases/download/2023.11.14-2/{release_filename}"
        archive_path = os.path.join(script_dir, release_filename)
        print(f"Downloading Piper release from {download_url}...")
        try:
            subprocess.check_call(["wget", "-O", archive_path, download_url])
        except Exception as e:
            print("Error downloading Piper archive:", e)
            sys.exit(1)
        print("Download complete.")
        os.makedirs(piper_folder, exist_ok=True)
        if release_filename.endswith(".tar.gz"):
            subprocess.check_call(["tar", "-xzvf", archive_path, "-C", piper_folder, "--strip-components=1"])
        elif release_filename.endswith(".zip"):
            subprocess.check_call(["unzip", "-o", archive_path, "-d", piper_folder])
        else:
            print("Unsupported archive format.")
            sys.exit(1)
        print("Piper extracted to", piper_folder)
    else:
        print("Piper executable found at", piper_exe)
    
    # Check for ONNX files.
    onnx_json = os.path.join(script_dir, "glados_piper_medium.onnx.json")
    onnx_model = os.path.join(script_dir, "glados_piper_medium.onnx")
    if not os.path.isfile(onnx_json):
        print("ONNX JSON file not found. Downloading...")
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json"
        subprocess.check_call(["wget", "-O", onnx_json, url])
        print("Downloaded ONNX JSON file.")
    else:
        print("ONNX JSON file exists.")
    if not os.path.isfile(onnx_model):
        print("ONNX model file not found. Downloading...")
        url = "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx"
        subprocess.check_call(["wget", "-O", onnx_model, url])
        print("Downloaded ONNX model file.")
    else:
        print("ONNX model file exists.")

setup_piper_and_onnx()

# ----- Automatic Dependency Installation -----
SETUP_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".setup_complete")
if not os.path.exists(SETUP_MARKER):
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "sounddevice", "numpy", "scipy",
        "openai-whisper",   # For Whisper transcription
        "ollama"            # For Ollama Python API
    ])
    with open(SETUP_MARKER, "w") as f:
        f.write("Setup complete")
    print("Dependencies installed. Restarting script...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- Configuration Loading -----
def load_config():
    """
    Load configuration from config.json (in the script directory). If not present, create it with default values.
    Returns a dictionary of configuration values.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    defaults = {
        "ollama_model": "gemma3:4b",
        "onnx_json": "glados_piper_medium.onnx.json",
        "onnx_model": "glados_piper_medium.onnx",
        "whisper_model": "base"
    }
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump(defaults, f, indent=4)
        print("Created default config.json")
        return defaults
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        # Ensure any missing keys are set to defaults.
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
        return config

config = load_config()
DEFAULT_MODEL = config["ollama_model"]

# ----- Chat Session Logging Setup -----
def setup_chat_session():
    """
    Create a new folder for the chat session inside "chat_sessions", named based on the current datetime.
    Returns the session folder path and opens a log file "session.txt" for appending.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sessions_folder = os.path.join(script_dir, "chat_sessions")
    os.makedirs(sessions_folder, exist_ok=True)
    session_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(sessions_folder, session_name)
    os.makedirs(session_folder, exist_ok=True)
    session_log_path = os.path.join(session_folder, "session.txt")
    session_log = open(session_log_path, "a", encoding="utf-8")
    print(f"Chat session started: {session_name}")
    return session_folder, session_log

import datetime
session_folder, session_log = setup_chat_session()
chat_history = []  # Global chat history list

# ----- Now import custom packages -----
from sounddevice import InputStream
import numpy as np
from scipy.io.wavfile import write
import whisper
from ollama import chat
import threading, queue, re, time, json

print("Loading Whisper model ({} model)...".format(config["whisper_model"]))
whisper_model = whisper.load_model(config["whisper_model"])

# ----- Global Settings and Queues -----
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024

tts_queue = queue.Queue()    # TTS request queue
audio_queue = queue.Queue()  # Microphone audio queue

# Global variable for current TTS process (for interruption).
current_tts_process = None
tts_lock = threading.Lock()

def flush_current_tts():
    """Clear the TTS queue and kill any current TTS playback."""
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
            except Exception as e:
                print("Error killing Piper process:", e)
            try:
                aproc.kill()
            except Exception as e:
                print("Error killing aplay process:", e)
            current_tts_process = None

# ----- TTS Processing: Using Piper via subprocess -----
def process_tts_request(text):
    """
    Build a JSON payload from text and call the local Piper executable (from "piper/piper")
    using the local ONNX files. Pipe raw audio output to aplay for immediate playback.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_exe = os.path.join(script_dir, "piper", "piper")
    onnx_json = os.path.join(script_dir, config["onnx_json"])
    onnx_model = os.path.join(script_dir, config["onnx_model"])
    for path, desc in [(piper_exe, "Piper executable"),
                       (onnx_json, "ONNX JSON file"),
                       (onnx_model, "ONNX model file")]:
        if not os.path.isfile(path):
            print(f"Error: {desc} not found at {path}")
            return
    payload = {"text": text, "config": onnx_json, "model": onnx_model}
    payload_str = json.dumps(payload)
    cmd_piper = [piper_exe, "-m", onnx_model, "--debug", "--json-input", "--output_raw"]
    cmd_aplay = ["aplay", "--buffer-size=777", "-r", "22050", "-f", "S16_LE"]
    print(f"\n[TTS] Synthesizing: '{text}'")
    try:
        with tts_lock:
            proc = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc_aplay = subprocess.Popen(cmd_aplay, stdin=proc.stdout)
            global current_tts_process
            current_tts_process = (proc, proc_aplay)
        proc.stdin.write(payload_str.encode("utf-8"))
        proc.stdin.close()
        proc_aplay.wait()
        proc.wait()
        with tts_lock:
            current_tts_process = None
        stderr_output = proc.stderr.read().decode("utf-8")
        if stderr_output:
            print("[Piper STDERR]:")
            print(stderr_output)
    except Exception as e:
        print("Error during TTS processing:")
        print(e)

def tts_worker(q):
    while True:
        text = q.get()
        if text is None:
            break
        process_tts_request(text)
        q.task_done()

# ----- Microphone Audio Capture -----
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)
    audio_queue.put(indata.copy())

def start_audio_capture():
    print("Starting microphone capture... (Press Ctrl+C to stop)")
    stream = InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
    stream.start()
    return stream

# ----- Whisper Transcription & LLM Prompt Trigger with Chat History and Interruption -----
def voice_to_llm_loop():
    print("Voice-to-LLM loop started. Listening for voice input...")
    while True:
        time.sleep(5)  # Check interval
        if audio_queue.empty():
            continue
        chunks = []
        while not audio_queue.empty():
            chunks.append(audio_queue.get())
            audio_queue.task_done()
        if not chunks:
            continue
        audio_data = np.concatenate(chunks, axis=0)
        audio_array = audio_data.flatten().astype(np.float32)
        rms = np.sqrt(np.mean(np.square(audio_array)))
        if rms < 0.01:
            print("Audio RMS too low. Skipping transcription.")
            continue
        print("Transcribing voice input...")
        try:
            result = whisper_model.transcribe(audio_array, language="en")
        except Exception as e:
            print("Error during transcription:", e)
            continue
        transcription = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        if not transcription:
            print("No transcription result.")
            continue
        print(f"Transcribed prompt: {transcription}")
        session_log.write(f"User: {transcription}\n")
        session_log.flush()
        chat_history.append({"role": "user", "content": transcription})
        flush_current_tts()  # Interrupt current TTS playback
        pull_model_if_needed(DEFAULT_MODEL)
        # Pass the entire chat history for context
        try:
            stream = chat(model=DEFAULT_MODEL,
                          messages=chat_history,
                          stream=True)
        except Exception as e:
            print("Error calling Ollama API:", e)
            continue
        print("LLM response (streaming): ", end="", flush=True)
        accumulated = ""
        full_response = ""
        pattern = re.compile(r"(.*?[,.!?])(\s|$)")
        for chunk in stream:
            chunk_text = chunk['message']['content']
            print(chunk_text, end="", flush=True)
            accumulated += chunk_text
            full_response += chunk_text
            while True:
                match = pattern.match(accumulated)
                if not match:
                    break
                sentence = match.group(1).strip()
                sentence = clean_text(sentence)
                if sentence:
                    tts_queue.put(sentence)
                accumulated = accumulated[len(match.group(0)):]
        if accumulated.strip():
            sentence = clean_text(accumulated.strip())
            if sentence:
                tts_queue.put(sentence)
        print()
        if full_response.strip():
            chat_history.append({"role": "assistant", "content": full_response.strip()})
            session_log.write(f"Assistant: {full_response.strip()}\n")
            session_log.flush()
        print("--- Awaiting further voice input ---")

def pull_model_if_needed(model):
    try:
        subprocess.run(["ollama", "show", model], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print(f"Model '{model}' not found. Pulling model via Ollama...")
        try:
            subprocess.check_call(["ollama", "pull", model])
            print(f"Model '{model}' pulled successfully.")
        except Exception as e:
            print(f"Failed to pull model '{model}':", e)

def clean_text(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text.replace("*", "")

# ----- Main Function -----
def main():
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,))
    tts_thread.daemon = True
    tts_thread.start()
    
    try:
        mic_stream = start_audio_capture()
    except Exception as e:
        print("Error starting microphone capture:", e)
        sys.exit(1)
    
    voice_thread = threading.Thread(target=voice_to_llm_loop)
    voice_thread.daemon = True
    voice_thread.start()
    
    print("\nVoice-activated LLM mode (default model: {}) is running.".format(DEFAULT_MODEL))
    print("Speak into the microphone; your transcribed prompt (with context) will be sent to the LLM.")
    print("LLM responses are streamed and spoken via Piper. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    mic_stream.stop()
    tts_queue.put(None)
    tts_queue.join()
    tts_thread.join()
    voice_thread.join()
    session_log.close()
    
    # Save chat history to a JSON file.
    history_path = os.path.join(session_folder, "chat_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)
    
    print("Chat session saved in folder:", session_folder)
    print("System terminated.")

if __name__ == "__main__":
    main()
