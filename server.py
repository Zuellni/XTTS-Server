from argparse import ArgumentParser
from enum import StrEnum
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import DirectoryPath
from rich.progress import Progress

from model import Model
from schema import Input, Settings, Speaker

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


@app.post("/set_tts_settings")
def set(settings: Settings):
    model.settings = settings


@app.get("/speakers")
def get():
    return [
        {"name": s.capitalize(), "voice_id": s, "preview_url": ""}
        for s in model.speakers
    ]


@app.get("/tts_stream")
async def stream(request: Request, input: Input = Depends()):
    async def generator():
        async for output in model.stream(input):
            if await request.is_disconnected():
                break

            yield output

    return StreamingResponse(generator(), media_type="audio/ogg")


@app.post("/tts_to_audio")
async def generate(request: Request, input: Input):
    async def generator():
        async for output in model.generate(input):
            if await request.is_disconnected():
                break

            yield output

    return StreamingResponse(generator(), media_type="audio/ogg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("-d", "--deepspeed", action="store_true")
    parser.add_argument("-m", "--model", type=DirectoryPath, required=True)
    parser.add_argument("-s", "--speakers", type=DirectoryPath, required=True)
    args = parser.parse_args()

    with Progress(transient=True) as progress:
        loading = progress.add_task("Loading model", total=None)
        cache = Path(__file__).parent / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        model = Model(args.model, cache, args.deepspeed)

    with Progress(transient=True) as progress:
        suffixes = (".flac", ".mp3", ".ogg", ".wav")
        speakers = [s for s in args.speakers.glob("*.*") if s.suffix in suffixes]
        caching = progress.add_task("Caching speakers", total=len(speakers))

        for speaker in speakers:
            model.add(speaker)
            progress.advance(caching)

        Speakers = StrEnum("Speakers", ((s, s) for s in model.speakers))
        Speaker._member_map_ = Speakers._member_map_
        Speaker._member_names_ = Speakers._member_names_
        Speaker._value2member_map_ = Speakers._value2member_map_

    uvicorn.run(app, host=args.host, port=args.port)
