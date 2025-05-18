import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from Config import Config


class AudioService:
    def __init__(self):
        self.model_name = Config.WHISPER_MODEL_NAME
        self.processor = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name
        )

    def transcribe_audio(self, file_path):
        audio_waveform, sr = librosa.load(file_path, sr=16000)
        input_features = self.processor(
            audio_waveform,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription