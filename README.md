# XTTS Server
An alternative to [xtts-api-server](https://github.com/daswer123/xtts-api-server) and [alltalk_tts](https://github.com/erew123/alltalk_tts), compatible with the [SillyTavern](https://github.com/SillyTavern/SillyTavern) xtts extension. Probably not as good as either, but works with the [coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) fork of [TTS](https://github.com/coqui-ai/TTS) and has as few requirements as possible.

## Installation
Create a new environment with conda/mamba:
```
conda create -n xtts git python pytorch pytorch-cuda torchaudio -c conda-forge -c nvidia -c pytorch
conda activate xtts
```

Clone the repository and install requirements, [Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) are required on Windows:
```
git clone https://github.com/zuellni/xtts-server xtts
pip install -r xtts/requirements.txt
```

## Usage
Download the model, get some speakers and start the server:
```
git lfs install
git clone https://huggingface.co/coqui/xtts-v2 -b main --depth 1 xtts/model
python xtts/server.py -m xtts/model -s xtts/speakers
```
