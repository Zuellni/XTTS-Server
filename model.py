import warnings
from io import BytesIO

import torch
import torchaudio
from pydantic import DirectoryPath
from safetensors.torch import load_file, save_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from schema import Input, Settings


class Model:
    def __init__(self, model_dir: DirectoryPath, cache_dir: DirectoryPath):
        config = XttsConfig()
        config.load_json(model_dir / "config.json")

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, model_dir)

        self.settings = Settings()
        self.cache_dir = cache_dir
        self.speakers = {}

    def add(self, path: DirectoryPath):
        name = path.stem.lower()
        file = self.cache_dir / f"{name}.st"
        self.model.cpu()

        if not file.exists():
            latent, embed = self.model.get_conditioning_latents(
                audio_path=path,
                librosa_trim_db=60,
                sound_norm_refs=True,
            )

            state_dict = {"latent": latent.contiguous(), "embed": embed.contiguous()}
            save_file(state_dict, file)

        state_dict = load_file(file, device="cuda")
        self.speakers[name] = (state_dict["latent"], state_dict["embed"])

    def generate(self, input: Input):
        latent, embed = self.speakers[input.speaker_wav]
        self.model.cuda()

        output = self.model.inference(
            text=input.text,
            language=input.language,
            gpt_cond_latent=latent,
            speaker_embedding=embed,
            speed=self.settings.speed,
            temperature=self.settings.temperature,
            length_penalty=self.settings.length_penalty,
            repetition_penalty=self.settings.repetition_penalty,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
            enable_text_splitting=self.settings.enable_text_splitting,
        )["wav"]

        output = torch.tensor(output)
        return self.encode(output, "wav")

    async def stream(self, input: Input):
        latent, embed = self.speakers[input.speaker_wav]
        self.model.cuda()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            for chunk in self.model.inference_stream(
                text=input.text,
                language=input.language,
                gpt_cond_latent=latent,
                speaker_embedding=embed,
                **self.settings.dict(),
            ):
                yield self.encode(chunk, "ogg")

    def encode(self, tensor: torch.Tensor, format: str):
        buffer = BytesIO()
        tensor = tensor.unsqueeze(0).cpu()
        torchaudio.save(buffer, tensor, 24000, format=format)
        buffer.seek(0)
        return buffer.read()
