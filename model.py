from dataclasses import dataclass
from typing import Optional
@dataclass
class VerseMetadata:
    AyahNo: Optional[int] = None
    SurahNo: Optional[int] = None
@dataclass
class AudioQueryResponse:
    filtered_item: Optional[VerseMetadata] = None
    transcription: Optional[str] = None
    message: Optional[str] = None

@dataclass
class ErrorResponse:
    error: str