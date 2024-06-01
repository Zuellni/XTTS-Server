from io import BytesIO

import torch
import torch.nn.functional as F
import torchaudio
from pydantic import DirectoryPath, FilePath
from safetensors.torch import load_file, save_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import split_sentence
from TTS.tts.models.xtts import Xtts

from schema import Input, Settings


class Model:
    def __init__(
        self, model: DirectoryPath, device: str, offload: bool, deepspeed: bool
    ):
        config = XttsConfig()
        config.load_json(model / "config.json")

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, model, use_deepspeed=deepspeed)

        self.settings = Settings()
        self.device = device
        self.offload = offload
        self.speakers = {}

    def add(self, path: FilePath):
        name = path.stem.lower()
        file = path.parent / f"{name}.st"
        self.model.cpu()

        if not file.exists():
            cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=path,
                librosa_trim_db=60,
                sound_norm_refs=True,
            )

            state_dict = {
                "cond_latent": cond_latent.contiguous(),
                "speaker_embedding": speaker_embedding.contiguous(),
            }

            save_file(state_dict, file)

        state_dict = load_file(file, device=self.device)

        self.speakers[name] = (
            state_dict["cond_latent"],
            state_dict["speaker_embedding"],
        )

    def prepare(self, input: Input):
        limit = min(
            self.model.tokenizer.char_limits[input.language],
            self.settings.stream_chunk_size,
        )

        inputs = (
            split_sentence(input.text, input.language, limit)
            if self.settings.enable_text_splitting
            else [input.text]
        )

        cond_latent, speaker_embedding = self.speakers[input.speaker_wav]
        return inputs, cond_latent, speaker_embedding

    async def generate(self, input: Input):
        inputs, cond_latent, speaker_embedding = self.prepare(input)
        self.model.to(self.device)

        for text in inputs:
            output = self.model.inference(
                text=text,
                language=input.language,
                gpt_cond_latent=cond_latent,
                speaker_embedding=speaker_embedding,
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

        if self.offload:
            self.model.cpu()

    async def stream(self, input: Input):
        inputs, cond_latent, speaker_embedding = self.prepare(input)
        self.model.to(self.device)

        for text in inputs:
            for output in self.model.inference_stream(
                text=text,
                language=input.language,
                gpt_cond_latent=cond_latent,
                speaker_embedding=speaker_embedding,
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

        if self.offload:
            self.model.cpu()

    def encode(self, input: torch.Tensor):
        output = BytesIO()
        input = input.unsqueeze(0).cpu()
        torchaudio.save(output, input, 24000, format="ogg")
        return output.getvalue()
