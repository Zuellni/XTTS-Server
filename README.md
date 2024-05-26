# XTTS-Server
## Installation
```
conda create -n xtts git python fastapi pytorch pytorch-cuda torchaudio -c conda-forge -c nvidia -c pytorch
conda activate xtts
pip install coqui-tts transformers<=4.40.2
git clone https://github.com/zuellni/xtts-server xtts
python xtts/server.py -m <model_dir> -s <speakers_dir>
```
