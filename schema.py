from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator


class Speaker(StrEnum):
    pass


class Input(BaseModel):
    text: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    speaker_wav: Annotated[
        str, StringConstraints(min_length=1, strip_whitespace=True, to_lower=True)
    ]
    language: Annotated[
        Literal[
            "ar", "cs", "de", "en", "es", "fr", "hi", "hu",
            "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr",
            "zh-cn",
        ],
        StringConstraints(strip_whitespace=True, to_lower=True),
    ] = "en"

    @field_validator("speaker_wav", mode="after")
    def validate_speaker(cls, value):
        if value not in Speaker:
            raise ValueError("invalid speaker")

        return value

    @field_validator("language", mode="after")
    def validate_language(cls, value):
        return value.split("-")[0]


class Settings(BaseModel):
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.1
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    length_penalty: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0
    repetition_penalty: Annotated[float, Field(ge=1.0, le=10.0)] = 7.0
    top_k: Annotated[int, Field(ge=0, le=100)] = 30
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    stream_chunk_size: Annotated[int, Field(ge=100, le=400)] = 100
    enable_text_splitting: bool = True
