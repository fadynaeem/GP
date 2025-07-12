import numpy as np
import librosa
import openl3
import noisereduce as nr
from scipy.signal import butter, lfilter, medfilt
import os
import tempfile
from config import get_config

config = get_config()


class AudioPreprocessor:
    def __init__(self, target_sr=None, duration=None):
        self.target_sr = target_sr or config.AUDIO_TARGET_SR
        self.duration = duration or config.AUDIO_DURATION

    @staticmethod
    def bandpass_filter(y, sr, lowcut=85.0, highcut=8000.0, order=5):
        nyq = 0.5 * sr
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return lfilter(b, a, y)

    def _load_and_clean_audio(self, audio_bytes):
        temp_filename = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_bytes)
            y, sr = librosa.load(
                temp_filename, sr=self.target_sr, duration=self.duration
            )
            if len(y) == 0:
                raise ValueError("Loaded audio is empty")
            processing_steps = [
                (lambda x: librosa.effects.trim(x)[0]),
                (self._apply_noise_reduction),
                (lambda x: self.bandpass_filter(x, sr)),
                (librosa.util.normalize),
                (lambda x: medfilt(x, kernel_size=3))
            ]
            processed = y
            for step_fn in processing_steps:
                try:
                    processed = step_fn(processed)
                except Exception:
                    continue
            return processed, sr
        except Exception:
            return None, None
        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

    def _apply_noise_reduction(self, y):
        noise_sample_length = min(int(0.5 * self.target_sr), len(y) // 4)
        noise_sample = y[:noise_sample_length]
        return nr.reduce_noise(
            y=y,
            sr=self.target_sr,
            y_noise=noise_sample,
            prop_decrease=0.8,
            stationary=False
        )

    def _extract_openl3(self, y, sr):
        try:
            min_length = sr
            if len(y) < min_length:
                y = np.pad(y, (0, max(0, min_length - len(y))), mode='constant')
            emb, _ = openl3.get_audio_embedding(
                y, sr,
                input_repr="mel256",
                content_type="music",
                embedding_size=512
            )
            if emb is None or len(emb) == 0:
                raise ValueError("Empty embeddings returned")
            return np.mean(emb, axis=0)
        except Exception as exc:
            raise ValueError(
                "OpenL3 feature extraction failed: {}".format(
                    str(exc)
                )
            )

    def preprocess_audio(self, audio_bytes):
        if not audio_bytes:
            raise ValueError("Empty audio bytes provided")
        y, sr = self._load_and_clean_audio(audio_bytes)
        if y is None or sr is None:
            raise ValueError("Audio loading/cleaning failed")
        return self._extract_openl3(y, sr) 