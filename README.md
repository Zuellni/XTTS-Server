# XTTS Server
An alternative to [xtts-api-server](https://github.com/daswer123/xtts-api-server) and [alltalk_tts](https://github.com/erew123/alltalk_tts), compatible with the [SillyTavern](https://github.com/SillyTavern/SillyTavern) xtts extension. Probably not as good as either, but works with the [coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) fork of [TTS](https://github.com/coqui-ai/TTS) and has as few requirements as possible.
## Installation
```
conda create -n xtts git python fastapi pytorch pytorch-cuda torchaudio -c conda-forge -c nvidia -c pytorch
conda activate xtts
pip install coqui-tts transformers<=4.40.2
git clone https://github.com/zuellni/xtts-server xtts
python xtts/server.py -m <model_dir> -s <speakers_dir>
```
