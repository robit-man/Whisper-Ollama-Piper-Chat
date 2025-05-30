# A standalone natural communication stack with local LLM's

### Just Run This Line to install and deploy!
```bash
cd "$HOME" && \
{ [ -f Whisper-Ollama-Piper-Chat/app.py ] || git clone https://github.com/robit-man/Whisper-Ollama-Piper-Chat.git; } && \
cd Whisper-Ollama-Piper-Chat && \
{ [ -f .env ] || echo "BRAVE_API_KEY=" > .env; } && \
if command -v apt-get >/dev/null; then dpkg -s portaudio19-dev &>/dev/null || { sudo apt-get update && sudo apt-get install -y portaudio19-dev libsndfile1; }; fi && \
if command -v brew >/dev/null;   then brew ls --versions portaudio &>/dev/null || brew install portaudio;       fi && \
python3 app.py

````

Please note that you must populate the created .env with a free [brave search api key](https://brave.com/search/api/)

## Input: [Whisper](https://pypi.org/project/openai-whisper/) + [Denoiser](https://github.com/facebookresearch/denoiser)
## Cognition: [Ollama](https://ollama.com/)
## Output: [Piper](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2)


## OLD NOTES -----------------------------------------

Just run ```python3 app.py``` in a folder on your system, it will automatically handle all installation steps, from ollama install if missing, all the way to pulling weights for each stage of transduction of data

it has only been tested on non-arm linux (Ubuntu 24.04)

and you must obviously have CUDA capable hardware to run the models for a comfortable realtime factor

and one nuance I ran into was the following requirement I had to install the following libraries manually at a system level (for agents)

```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
```

if you want to enable the brave search functionality through their [free rate-limited api](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started), you must create a .env file parallel to the app.py, and add the api key [you can retreive through signing up with brave search](https://api-dashboard.search.brave.com/register)

```.env
BRAVE_API_KEY=<brave-search-api-key>
```
where ```<brave-search-api-key>``` is replaced with the retreived api key after creating an account (for agents)
