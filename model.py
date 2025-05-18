from dataclasses import dataclass
from typing import Optional


@dataclass
class VerseMetadata:
    """Model for verse metadata"""
    AyahNo: Optional[int] = None
    SurahNo: Optional[int] = None


@dataclass
class AudioQueryResponse:
    """Model for audio query response"""
    filtered_item: Optional[VerseMetadata] = None
    transcription: Optional[str] = None
    message: Optional[str] = None


@dataclass
class ErrorResponse:
    """Model for error responses"""
    error: str