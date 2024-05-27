# XTTS Server
An alternative to [xtts-api-server](https://github.com/daswer123/xtts-api-server) and [alltalk_tts](https://github.com/erew123/alltalk_tts), compatible with the [SillyTavern](https://github.com/SillyTavern/SillyTavern) xtts extension. Probably not as good as either, but works with the [coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) fork of [TTS](https://github.com/coqui-ai/TTS) and has as few requirements as possible.

## Installation
Create a new environment with conda/miniconda/mamba/micromamba, `python>=3.12` should be fine:
```
conda create -n xtts git python fastapi pytorch pytorch-cuda torchaudio -c conda-forge -c nvidia -c pytorch
conda activate xtts
```

Install dependencies, `coqui-tts` needs [Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) to compile on Windows, `transformers>=4.41.0` fails on streaming:
```
pip install coqui-tts "transformers<=4.40.2"
```

Optional speedup, unlikely to compile on Windows:
```
pip install deepspeed
```

Clone the repository, download the model, get some speakers and start the server:
```
git clone https://github.com/zuellni/xtts-server -b main --depth 1 xtts
git clone https://huggingface.co/coqui/xtts-v2 -b main --depth 1 xtts/model
python xtts/server.py -m xtts/model -s xtts/speakers
```
