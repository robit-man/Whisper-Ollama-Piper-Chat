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
"""

import sys, os, subprocess, platform, re, json, time, threading, queue, datetime, inspect, difflib
from datetime import datetime

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
    Ensure Piper executable and ONNX files are available.
    Downloads and extracts files if missing.
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
        "ollama",           # For Ollama Python API
        "python-dotenv",    # For environment variables
        "beautifulsoup4",   # For BS4 scraping
        "html5lib",         # Parser for BS4
        "pywifi",           # For WiFi scanning
        "psutil",           # For system utilization
        "num2words",        # For converting numbers to words
        "noisereduce",       # For noise cancellation (fallback, not used here)
        "denoiser"          # For real-time speech enhancement via denoiser
    ])
    with open(SETUP_MARKER, "w") as f:
        f.write("Setup complete")
    print("Dependencies installed. Restarting script...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ----- Configuration Loading -----
def load_config():
    """
    Load configuration from config.json (in the script directory). If not present, create it with default values.
    New keys include settings for primary/secondary models, temperatures, RMS threshold, debug audio playback,
    noise reduction, and consensus threshold.
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
        "whisper_model_base": "base",      # Use "base" model for primary transcription
        "whisper_model_medium": "medium",  # Use "medium" model for validation
        "stream": True,
        "raw": False,
        "images": None,
        "options": {},
        "system": "You are a helpful assistant.",
        "conversation_id": "default_convo",
        "rms_threshold": 0.01,
        "debug_audio_playback": False,
        "enable_noise_reduction": True,
        "consensus_threshold": 0.8       # Similarity ratio required for consensus between models
    }
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump(defaults, f, indent=4)
        print("Created default config.json")
        return defaults
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
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
    print(f"Chat session started: {session_name}")
    return session_folder, session_log

session_folder, session_log = setup_chat_session()

# ----- Import Additional Packages -----
from sounddevice import InputStream
from scipy.io.wavfile import write
import whisper
from ollama import chat
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pywifi
from pywifi import const
import psutil
from num2words import num2words
import sounddevice as sd  # For debug audio playback
import numpy as np
from scipy.signal import butter, lfilter  # For EQ enhancement
import torch
from denoiser import pretrained  # For speech enhancement via denoiser
load_dotenv()

# ----- Load Whisper Models -----
print("Loading Whisper models...")
whisper_model_base = whisper.load_model(config["whisper_model_base"])
whisper_model_medium = whisper.load_model(config["whisper_model_medium"])
print("Both base and medium models loaded.")

# ----- Load Denoiser Model -----
print("Loading denoiser model (DNS64)...")
denoiser_model = pretrained.dns64()   # Loads a pretrained DNS64 model from denoiser
print("Denoiser model loaded.")

# ----- Global Settings and Queues -----
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024
tts_queue = queue.Queue()
audio_queue = queue.Queue()
current_tts_process = None
tts_lock = threading.Lock()

def flush_current_tts():
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

# ----- TTS Processing -----
def process_tts_request(text):
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

# ----- EQ-based Audio Enhancement Helper -----
def apply_eq_boost(audio, sample_rate, lowcut=300, highcut=3000, gain=2.0):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype="band")
    band = lfilter(b, a, audio)
    audio_boosted = audio + (gain - 1.0) * band
    max_val = np.max(np.abs(audio_boosted))
    if max_val > 1:
        audio_boosted = audio_boosted / max_val
    return audio_boosted.astype(np.float32)

def clean_text(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text.replace("*", "")

# ----- Consensus Transcription Helper -----
def consensus_whisper_transcribe_helper(audio_array, language="en", rms_threshold=0.01, consensus_threshold=0.8):
    # First check RMS level to reject low-volume chunks
    rms = np.sqrt(np.mean(np.square(audio_array)))
    if rms < rms_threshold:
        print("Audio RMS too low (RMS: {:.5f}). Skipping transcription.".format(rms))
        return ""
    
    transcription_base = ""
    transcription_medium = ""
    
    # Worker functions for threading
    def transcribe_with_base():
        nonlocal transcription_base
        try:
            result = whisper_model_base.transcribe(audio_array, language=language)
            transcription_base = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        except Exception as e:
            print("Error during base transcription:", e)
    
    def transcribe_with_medium():
        nonlocal transcription_medium
        try:
            result = whisper_model_medium.transcribe(audio_array, language=language)
            transcription_medium = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        except Exception as e:
            print("Error during medium transcription:", e)
    
    # Run both transcriptions concurrently in separate threads.
    thread_base = threading.Thread(target=transcribe_with_base)
    thread_medium = threading.Thread(target=transcribe_with_medium)
    thread_base.start()
    thread_medium.start()
    thread_base.join()
    thread_medium.join()
    
    # If either transcription is empty, discard.
    if not transcription_base or not transcription_medium:
        print("One of the models returned no transcription.")
        return ""
    
    # Compare the two transcriptions
    similarity = difflib.SequenceMatcher(None, transcription_base, transcription_medium).ratio()
    print(f"Transcription similarity: {similarity:.2f}")
    if similarity >= consensus_threshold:
        # Accept the transcription (returning the base model's output)
        return transcription_base
    else:
        print("No consensus between base and medium models; ignoring transcription.")
        return ""

# ----- Transcription Validation Helper -----
def validate_transcription(text):
    if not any(ch.isalpha() for ch in text):
        return False
    words = text.split()
    if len(words) < 2:
        return False
    return True

# ----- Manager & Helper Classes for Tool-Calling Chat System -----
class ConfigManager:
    def __init__(self, config):
        config["model"] = config["primary_model"]
        self.config = config

class HistoryManager:
    def __init__(self):
        self.history = []
    def add_entry(self, role, content):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)

class TTSManager:
    def enqueue(self, text):
        tts_queue.put(text)
    def stop(self):
        flush_current_tts()
    def start(self):
        pass

class MemoryManager:
    def store_message(self, conversation_id, role, message, embedding):
        pass
    def retrieve_similar(self, conversation_id, embedding, top_n=3, mode="conversational"):
        return []
    def retrieve_latest_summary(self, conversation_id):
        return None

class ModeManager:
    def detect_mode(self, history):
        return "conversational"

class DisplayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_tokens = ""
        self.current_request = ""
        self.current_tool_calls = ""

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
        return emoji_pattern.sub(r'', text)
    @staticmethod
    def convert_numbers_to_words(text):
        def replace_num(match):
            number_str = match.group(0)
            try:
                return num2words(int(number_str))
            except ValueError:
                return number_str
        return re.sub(r'\b\d+\b', replace_num, text)
    @staticmethod
    def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    @staticmethod
    def cosine_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
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
                return json.load(f)
        except Exception:
            return default
    @staticmethod
    def load_format_schema(fmt):
        if not fmt:
            return None
        if fmt.lower() == "json":
            return "json"
        if os.path.exists(fmt):
            try:
                with open(fmt, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    @staticmethod
    def monitor_script(interval=5):
        script_path = os.path.abspath(__file__)
        last_mtime = os.path.getmtime(script_path)
        while True:
            time.sleep(interval)
            try:
                new_mtime = os.path.getmtime(script_path)
                if new_mtime != last_mtime:
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception:
                pass
    @staticmethod
    def embed_text(text):
        try:
            response = chat(model="nomic-embed-text", messages=[{"role": "user", "content": text}], stream=False)
            embedding = json.loads(response["message"]["content"])
            vec = np.array(embedding, dtype=float)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            return vec / norm
        except Exception as e:
            return np.zeros(768)

class Tools:
    @staticmethod
    def parse_tool_call(text):
        pattern = r"```tool_(?:code|call)\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    @staticmethod
    def see_whats_around():
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        url = "http://127.0.0.1:8080/camera/default_0"
        try:
            import requests
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{timestamp}.jpg"
                file_path = os.path.join(images_dir, filename)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return file_path
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    @staticmethod
    def get_battery_voltage():
        try:
            home_dir = os.path.expanduser("~")
            file_path = os.path.join(home_dir, "voltage.txt")
            with open(file_path, "r") as f:
                return float(f.readline().strip())
        except Exception as e:
            raise RuntimeError(f"Error reading battery voltage: {e}")
    @staticmethod
    def brave_search(topic):
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
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
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            return response.text if response.status_code == 200 else f"Error {response.status_code}: {response.text}"
        except Exception as e:
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
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            return soup.prettify()
        except Exception as e:
            return f"Error during scraping: {e}"
    @staticmethod
    def find_file(filename, search_path="."):
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                return root
        return None
    @staticmethod
    def get_current_location():
        try:
            import requests
            response = requests.get("http://ip-api.com/json", timeout=5)
            return response.json() if response.status_code == 200 else {"error": f"HTTP error {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    @staticmethod
    def get_system_utilization():
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
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
            response = chat(model=secondary_model, messages=payload["messages"], stream=False)
            return response["message"]["content"]
        except Exception as e:
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

    def build_payload(self):
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
        if self.history_manager.history:
            last_user_msg = next((msg["content"] for msg in reversed(self.history_manager.history) if msg["role"] == "user"), "")
            _ = Utils.embed_text(last_user_msg)
            mem_context = ""
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
        return payload

    def chat_completion_stream(self, processed_text):
        payload = self.build_payload()
        tokens = ""
        try:
            stream = chat(model=self.config_manager.config["primary_model"],
                          messages=payload["messages"],
                          stream=self.config_manager.config["stream"])
            for part in stream:
                if self.stop_flag:
                    yield "", True
                    return
                content = part["message"]["content"]
                tokens += content
                with display_state.lock:
                    display_state.current_tokens = tokens
                yield content, part.get("done", False)
                if part.get("done", False):
                    break
        except Exception:
            yield "", True

    def chat_completion_nonstream(self, processed_text):
        payload = self.build_payload()
        try:
            response = chat(model=self.config_manager.config["primary_model"],
                            messages=payload["messages"],
                            stream=False)
            return response["message"]["content"]
        except Exception:
            return ""

    def process_text(self, text, skip_tts=False):
        processed_text = Utils.convert_numbers_to_words(text)
        sentence_endings = re.compile(r'[.?!]+')
        tokens = ""
        if self.config_manager.config["stream"]:
            buffer = ""
            for content, done in self.chat_completion_stream(processed_text):
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
                    if sentence and not skip_tts:
                        threading.Thread(target=self.tts_manager.enqueue, args=(sentenceCleaned,), daemon=True).start()
                if done:
                    break
            if buffer.strip():
                tokens += buffer.strip()
                with display_state.lock:
                    display_state.current_tokens = tokens
            return tokens
        else:
            result = self.chat_completion_nonstream(processed_text)
            tokens = result
            with display_state.lock:
                display_state.current_tokens = tokens
            return tokens

    def inference_thread(self, user_message, result_holder, skip_tts):
        result = self.process_text(user_message, skip_tts)
        result_holder.append(result)

    def run_inference(self, prompt, skip_tts=False):
        result_holder = []
        with self.inference_lock:
            if self.current_thread and self.current_thread.is_alive():
                self.stop_flag = True
                self.current_thread.join()
                self.stop_flag = False
            self.tts_manager.stop()
            self.tts_manager.start()
            self.current_thread = threading.Thread(
                target=self.inference_thread,
                args=(prompt, result_holder, skip_tts)
            )
            self.current_thread.start()
        self.current_thread.join()
        return result_holder[0] if result_holder else ""

    def run_tool(self, tool_code):
        allowed_tools = {}
        for attr in dir(Tools):
            if not attr.startswith("_"):
                method = getattr(Tools, attr)
                if callable(method):
                    allowed_tools[attr] = method
        try:
            result = eval(tool_code, {"__builtins__": {}}, allowed_tools)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {e}"

    def new_request(self, user_message, skip_tts=False):
        self.history_manager.add_entry("user", user_message)
        _ = Utils.embed_text(user_message)
        with display_state.lock:
            display_state.current_request = user_message
            display_state.current_tool_calls = ""
        result = self.run_inference(user_message, skip_tts)
        tool_code = Tools.parse_tool_call(result)
        if tool_code:
            tool_output = self.run_tool(tool_code)
            tool_output_cleaned = clean_text(tool_output)
            print(tool_output_cleaned)
            formatted_output = f"```tool_output\n{tool_output}\n```"
            combined_prompt = f"{user_message}\n{formatted_output}"
            self.history_manager.add_entry("user", combined_prompt)
            _ = Utils.embed_text(combined_prompt)
            final_result = self.new_request(combined_prompt, skip_tts=False)
            return final_result
        else:
            self.history_manager.add_entry("assistant", result)
            _ = Utils.embed_text(result)
            return result

# ----- Voice-to-LLM Loop (for microphone input) -----
def voice_to_llm_loop(chat_manager: ChatManager):
    print("Voice-to-LLM loop started. Listening for voice input...")
    while True:
        time.sleep(5)
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
        
        # Apply noise reduction using the denoiser if enabled.
        if config.get("enable_noise_reduction", True):
            print("Applying denoiser to audio chunk...")
            # Convert audio to torch tensor and add batch and channel dimensions
            audio_tensor = torch.tensor(audio_array).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                enhanced_tensor = denoiser_model(audio_tensor)
            # Squeeze back to 1D numpy array
            audio_array = enhanced_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        # Optionally apply EQ enhancement and playback for debugging.
        audio_to_transcribe = audio_array
        if config.get("debug_audio_playback", False):
            audio_to_transcribe = apply_eq_boost(audio_array, SAMPLE_RATE)
            print("Playing back the enhanced audio for debugging...")
            sd.play(audio_to_transcribe, samplerate=SAMPLE_RATE)
            sd.wait()
        
        # Run both Whisper models concurrently and get consensus transcription.
        transcription = consensus_whisper_transcribe_helper(
            audio_to_transcribe,
            language="en",
            rms_threshold=config["rms_threshold"],
            consensus_threshold=config["consensus_threshold"]
        )
        if not transcription:
            continue
        
        # Validate transcription.
        if not validate_transcription(transcription):
            print("Transcription validation failed (likely hallucinated). Skipping.")
            continue
        
        print(f"Transcribed prompt: {transcription}")
        session_log.write(json.dumps({"role": "user", "content": transcription, "timestamp": datetime.now().isoformat()}) + "\n")
        session_log.flush()
        flush_current_tts()
        
        # Process transcription via ChatManager.
        response = chat_manager.new_request(transcription)
        print("LLM response:", response)
        session_log.write(json.dumps({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}) + "\n")
        session_log.flush()
        print("--- Awaiting further voice input ---")

# ----- New: Text Input Override Loop -----
def text_input_loop(chat_manager: ChatManager):
    """
    This loop runs in parallel to the voice transcription. It continuously reads text input from the keyboard.
    When the user types a message and hits enter, it is processed as if it were a spoken prompt.
    """
    print("\nText override mode is active. Type your message and press Enter to send it to the LLM.")
    while True:
        try:
            user_text = input()  # Blocking call in its own thread.
            if not user_text.strip():
                continue
            print(f"You typed: {user_text}")
            session_log.write(json.dumps({"role": "user", "content": user_text, "timestamp": datetime.now().isoformat()}) + "\n")
            session_log.flush()
            flush_current_tts()
            response = chat_manager.new_request(user_text)
            print("LLM response:", response)
            session_log.write(json.dumps({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}) + "\n")
            session_log.flush()
        except Exception as e:
            print("Error in text input loop:", e)

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
    
    config_manager = ConfigManager(config)
    history_manager = HistoryManager()
    tts_manager = TTSManager()
    memory_manager = MemoryManager()
    mode_manager = ModeManager()
    
    chat_manager = ChatManager(config_manager, history_manager, tts_manager, tools_data=True,
                               format_schema=None, memory_manager=memory_manager, mode_manager=mode_manager)
    
    # Start the voice-to-LLM loop thread (for microphone input)
    voice_thread = threading.Thread(target=voice_to_llm_loop, args=(chat_manager,))
    voice_thread.daemon = True
    voice_thread.start()
    
    # Start the text input override thread
    text_thread = threading.Thread(target=text_input_loop, args=(chat_manager,))
    text_thread.daemon = True
    text_thread.start()
    
    print("\nVoice-activated LLM mode (primary model: {}) is running.".format(config["primary_model"]))
    print("Speak into the microphone or type your prompt. LLM responses are streamed and spoken via Piper. Press Ctrl+C to exit.")
    
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
    # Note: text_thread may be blocked on input() so it might not join cleanly.
    session_log.close()
    
    # Package entire chat history into a JSON file.
    history_path = os.path.join(session_folder, "chat_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_manager.history, f, indent=4)
    
    print("Chat session saved in folder:", session_folder)
    print("System terminated.")

if __name__ == "__main__":
    main()
