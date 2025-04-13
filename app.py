#!/usr/bin/env python
"""
Voice-activated LLM Chat via Whisper & Piper TTS with Context & Interruption
----------------------------------------------------------------------------
This script does the following:
  1. Auto bootstraps a virtual environment ("whisper_env") if not already running inside one,
     installs required dependencies, and re-launches itself in the venv.
  2. Checks for the Piper executable and ONNX files. If missing, it detects the OS/architecture and
     downloads the appropriate Piper release from:
         https://github.com/rhasspy/piper/releases/download/2023.11.14-2/
     It then extracts Piper into a folder named "piper" (beside this script). It also downloads
     "glados_piper_medium.onnx.json" and "glados_piper_medium.onnx" if needed.
  3. Ensures the Ollama CLI is installed. If not, on Linux it installs Ollama via:
         curl -fsSL https://ollama.com/install.sh | sh
  4. Continuously listens to your microphone. Every few seconds, if the captured audio’s RMS is
     above a threshold (i.e. meaningful speech), it is transcribed using Whisper (base model).
  5. The transcription is added to a per‑session chat history (both user and model messages) which
     is sent as context to the LLM via the Ollama Python API (default model "gemma3:4b"). If the model
     is missing, it is pulled via "ollama pull".
  6. The streaming LLM response is accumulated and split into chunks when a sentence-ending delimiter
     (comma, period, exclamation, or question mark) is encountered; emojis and asterisks are filtered out.
  7. Each complete chunk is enqueued to a TTS queue. A TTS worker thread processes queued text by
     building a JSON payload and launching Piper (via its JSON input mode), piping its raw audio output
     to aplay for immediate playback.
  8. When new voice input is detected (i.e. a new transcription occurs while an older TTS session is in
     progress), the current TTS queue is cleared and any active playback is cancelled for a seamless
     interruption and context switch.
     
Press Ctrl+C to exit.
"""

import sys, os, subprocess, platform, re, json, time, threading, queue

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
    Checks if the Piper executable and ONNX files are present.
    If Piper is missing, it determines the appropriate release (ignoring platform-specific folder names)
    and downloads/extracts it into a folder named "piper" (in the same directory as this script).
    Also downloads the ONNX files if they are not present.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_folder = os.path.join(script_dir, "piper")
    piper_exe = os.path.join(piper_folder, "piper")
    
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
    
    # Download ONNX files if missing.
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
        "openai-whisper",   # Whisper for transcription
        "ollama"            # Ollama Python API
    ])
    with open(SETUP_MARKER, "w") as f:
        f.write("Setup complete")
    print("Dependencies installed. Restarting script...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- System Check for Ollama CLI and Auto-Pull Model -----
def ensure_ollama_installed():
    """Ensure that the Ollama CLI is installed. If not, and if on Linux, install it."""
    try:
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Ollama CLI is installed.")
    except Exception:
        print("Ollama CLI not found. Attempting to install Ollama on Linux...")
        if platform.system() == "Linux":
            try:
                subprocess.check_call("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
                print("Ollama installed successfully.")
            except Exception as e:
                print("Failed to install Ollama:", e)
        else:
            print("Automatic Ollama installation is only supported on Linux.")
ensure_ollama_installed()

def pull_model_if_needed(model):
    """If the model does not exist in Ollama, pull it."""
    try:
        # Try to run an ollama command to show the model.
        subprocess.run(["ollama", "show", model], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print(f"Model '{model}' not found. Pulling model via Ollama...")
        try:
            subprocess.check_call(["ollama", "pull", model])
            print(f"Model '{model}' pulled successfully.")
        except Exception as e:
            print(f"Failed to pull model '{model}':", e)

# ----- Now import custom packages -----
from sounddevice import InputStream
import numpy as np
from scipy.io.wavfile import write
import whisper
from ollama import chat
import threading, queue, re, time, json

# ----- Global Settings and Queues -----
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024
DEFAULT_MODEL = "gemma3:4b"

tts_queue = queue.Queue()    # TTS request queue
audio_queue = queue.Queue()  # Microphone audio queue

print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")

# Global variable to hold current TTS subprocesses for interruption.
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
    Build a JSON payload from text and call the Piper executable (from "piper/piper")
    using the local ONNX files. Pipe the raw audio output to aplay for immediate playback.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_exe = os.path.join(script_dir, "piper", "piper")
    onnx_json = os.path.join(script_dir, "glados_piper_medium.onnx.json")
    onnx_model = os.path.join(script_dir, "glados_piper_medium.onnx")
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
chat_history = []  # Global chat history list.

def voice_to_llm_loop():
    print("Voice-to-LLM loop started. Listening for voice input...")
    while True:
        time.sleep(5)  # Check interval.
        if audio_queue.empty():
            continue
        # Retrieve audio chunks.
        chunks = []
        while not audio_queue.empty():
            chunks.append(audio_queue.get())
            audio_queue.task_done()
        if not chunks:
            continue
        audio_data = np.concatenate(chunks, axis=0)
        audio_array = audio_data.flatten().astype(np.float32)
        # Compute RMS to ensure there is enough signal.
        rms = np.sqrt(np.mean(np.square(audio_array)))
        if rms < 0.01:  # Adjust threshold as needed.
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
        # Update chat history with user message.
        chat_history.append({"role": "user", "content": transcription})
        # Before sending new prompt, flush any current TTS playback for interruption.
        flush_current_tts()
        # Ensure the default model exists; if not, pull it.
        pull_model_if_needed(DEFAULT_MODEL)
        # Build the full context by sending chat_history.
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
        # Update chat history with assistant's full response.
        if full_response.strip():
            chat_history.append({"role": "assistant", "content": full_response.strip()})
        print("--- Awaiting further voice input ---")

def pull_model_if_needed(model):
    """Ensure that the LLM model is available; if not, pull it using Ollama."""
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
    """Remove emojis and asterisks from the text."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
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
    
    print("\nVoice-activated LLM mode (default model: gemma3:4b) is running.")
    print("Speak into the microphone; your transcribed prompt (with context) will be sent to the LLM.")
    print("LLM responses will stream and be spoken via Piper. Press Ctrl+C to exit.")
    
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
    print("System terminated.")

if __name__ == "__main__":
    main()
