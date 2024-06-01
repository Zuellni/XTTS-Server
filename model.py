from io import BytesIO

import torch
import torchaudio
from pydantic import DirectoryPath
from safetensors.torch import load_file, save_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import split_sentence
from TTS.tts.models.xtts import Xtts

from schema import Input, Settings


class Model:
    def __init__(self, model: DirectoryPath, cache: DirectoryPath, deepspeed: bool):
        config = XttsConfig()
        config.load_json(model / "config.json")

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, model, use_deepspeed=deepspeed)

        self.settings = Settings()
        self.cache = cache
        self.speakers = {}

    def add(self, path: DirectoryPath):
        name = path.stem.lower()
        file = self.cache / f"{name}.st"
        self.model.cpu()

        if not file.exists():
            latent, embed = self.model.get_conditioning_latents(
                audio_path=path,
                librosa_trim_db=60,
                sound_norm_refs=True,
            )

            state_dict = {"latent": latent.contiguous(), "embed": embed.contiguous()}
            save_file(state_dict, file)

        state_dict = load_file(file)
        self.speakers[name] = (state_dict["latent"], state_dict["embed"])

    def process(self, input: Input):
        inputs = (
            split_sentence(input.text, input.language, self.settings.stream_chunk_size)
            if self.settings.enable_text_splitting
            else [input.text]
        )

        self.model.cuda()
        latent, embed = self.speakers[input.speaker_wav]
        return inputs, input.language, latent, embed

    async def generate(self, input: Input):
        inputs, lang, latent, embed = self.process(input)

        for input in inputs:
            output = self.model.inference(
                text=input,
                language=lang,
                gpt_cond_latent=latent,
                speaker_embedding=embed,
                speed=self.settings.speed,
                temperature=self.settings.temperature,
                length_penalty=self.settings.length_penalty,
                repetition_penalty=self.settings.repetition_penalty,
                top_k=self.settings.top_k,
                top_p=self.settings.top_p,
                enable_text_splitting=False,
            )["wav"]

            output = torch.tensor(output)
            yield self.encode(output)

    async def stream(self, input: Input):
        inputs, lang, latent, embed = self.process(input)

        for input in inputs:
            for output in self.model.inference_stream(
                text=input,
                language=lang,
                gpt_cond_latent=latent,
                speaker_embedding=embed,
                speed=self.settings.speed,
                temperature=self.settings.temperature,
                length_penalty=self.settings.length_penalty,
                repetition_penalty=self.settings.repetition_penalty,
                top_k=self.settings.top_k,
                top_p=self.settings.top_p,
                stream_chunk_size=self.settings.stream_chunk_size,
                enable_text_splitting=False,
            ):
                yield self.encode(output)

    def encode(self, input: torch.Tensor):
        output = BytesIO()
        input = input.unsqueeze(0).cpu()
        torchaudio.save(output, input, 24000, format="ogg")
        return output.getvalue()
