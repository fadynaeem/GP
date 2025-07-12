import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModel, WhisperProcessor, WhisperForConditionalGeneration
)
import tempfile
import librosa
import os
from config import get_config

config = get_config()

token = config.HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(
    config.ARABIC_MODEL_NAME,
    use_auth_token=token
)
model = AutoModel.from_pretrained(
    config.ARABIC_MODEL_NAME, output_hidden_states=True
)
whisper_processor = WhisperProcessor.from_pretrained(
    "tarteel-ai/whisper-base-ar-quran"
)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "tarteel-ai/whisper-base-ar-quran"
)


def get_sentence_embedding(sentence):
    if not sentence or not isinstance(sentence, str):
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).numpy().flatten()


def transcribe_audio_file(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()
        temp_path = temp_file.name
    try:
        audio_waveform, sr = librosa.load(temp_path, sr=16000)
        input_features = whisper_processor(
            audio_waveform,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        return whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
    finally:
        os.remove(temp_path) 