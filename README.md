# XTTS Server
An XTTS server with minimal requirements compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern).

## Installation
Create a new environment with mamba:
```
mamba create -n xtts git python pytorch pytorch-cuda torchaudio -c conda-forge -c nvidia -c pytorch
mamba activate xtts
```

Clone the repository and install requirements ([Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) required on Windows):
```
git clone https://github.com/zuellni/xtts-server --branch main --depth 1
cd xtts-server
pip install -r requirements.txt
```

Optionally build [DeepSpeed](https://github.com/microsoft/DeepSpeed) on Windows ([CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) required):
```
git clone https://github.com/microsoft/deepspeed --branch main --depth 1
cd deepspeed
build_win.bat
cd dist
pip install deepspeed-X.X.X-cpXXX-cpXXX-win_amd64.whl
```

## Usage
Download [XTTS-v2](https://huggingface.co/coqui/XTTS-v2), get some speaker files and start the server:
```
cd xtts-server
git lfs install
git clone https://huggingface.co/coqui/xtts-v2 --branch main --depth 1
python server.py -m xtts-v2 -s <speakers_dir>
```
