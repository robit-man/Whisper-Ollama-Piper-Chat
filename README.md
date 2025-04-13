# A standalone natural communication stack with local LLM's

## Input: Whisper
## Cognition: Ollama
## Output: Piper

Just run ```python3 app.py``` in a folder on your system, 

it has only been tested on non-arm linux (Ubuntu 24.04)

and you must obviously have CUDA capable hardware to run the models for a comfortable realtime factor

and one nuance I ran into was the following requirement I had to install manually

```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
```
