## Quran Recognition System — QuranRecognition

QuranRecognition is a multi-model deep learning system built with PyTorch and HuggingFace Transformers that owns the full Quranic audio intelligence pipeline end-to-end.
It enforces single-responsibility layers, centralizes adaptive preprocessing, and uses transformer-based speaker and verse embeddings to guarantee per-reciter identification and Surah–Ayah localization while scaling horizontally through a low-latency Flask API.

## Goals

**Reciter Identification**: identify which of the known reciters is speaking from raw audio input, handling phonetic variation, recording quality differences, and overlapping tajweed styles — achieving **94.64% top-1 accuracy**.

**Surah–Ayah Localization**: pinpoint the exact chapter and verse being recited using transformer-based verse embeddings and similarity analysis — achieving **96% accuracy** across all 114 Surahs.

**Handle Arabic phonetic complexity**: design a preprocessing pipeline specifically for Quranic audio — and stylistic variation across reciters and recording formats.

**Scalable real-time inference**: deploy batched, low-latency Flask API endpoints that serve both identification and localization simultaneously, built for production-volume audio streams.

## Models

**Reciter Identification Model — OpenL3 Embeddings + Vector Similarity (94.64% accuracy)**
Accepts any Quranic audio clip, extracts a deep audio embedding via OpenL3, and identifies the reciter through nearest-neighbor similarity search against a Pinecone Vector DB index of known reciter voice profiles.

**The problem (why reciter ID is hard)**
Quranic audio from different reciters shares the exact same source text, making traditional audio fingerprinting ineffective:

- Multiple reciters recite the exact same verses

- Recording conditions (studio, mosque, live) vary widely
- Emotional delivery and pace differ significantly per reciter

**Solution — OpenL3 + Pinecone Vector DB**:
- **OpenL3** extracts rich, fixed-size audio embeddings that capture timbre, prosody, and vocal texture — without requiring labeled reciter data at inference time
- **Pinecone Vector DB** stores pre-indexed embeddings for all known reciters; identification becomes a fast approximate nearest-neighbor (ANN) search — `similarity(query_embedding, index) → reciter`
- Double-validated via **t-SNE cluster visualization** — embeddings form tight, well-separated per-reciter clusters confirming genuine voice separation
- No retraining needed to add a new reciter — just upsert their embeddings into Pinecone

**Surah–Ayah Localization Model — Verse Embedding Matcher**
Implements a transformer-based sequence matcher with dual Surah and Ayah classification heads.
Validated through **cosine similarity analysis** across verse embeddings — confirming strong intra-verse coherence and high inter-verse discriminability.

## Audio Processing Pipeline

```
              POST /api/recognize
                      │
                      ▼
        Raw Audio (MP3 / WAV )
                      │
                      ▼  ADAPTIVE PREPROCESSING
     ┌────────────────────────────────────────┐
     │  Format Normalization (SR, bit depth)  │
     │  Voice Activity Detection (VAD)        │
     │  Band-pass Filter (recitation freqs)   │
     │  Spectral Noise Reduction              │
     │  RMS Normalization                     │
     │  Feature Extraction (OpenL3 Embeddings) │
     └──────────────────┬─────────────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
   ┌──────────────────┐   ┌──────────────────────┐
   │ Reciter ID Model │   │ Surah–Ayah Localizer  │
   │  (Transformer)   │   │   (Transformer)       │
   │   94.64% acc.    │   │    96.00% acc.        │
   └────────┬─────────┘   └──────────┬────────────┘
            │                        │
            └──────────┬─────────────┘
                       ▼
              Flask REST API
           (Batched Inference)
                       │
          ┌────────────┴──────────────┐
          ▼                           ▼
  /identify response          /localize response
  { reciter, confidence }    { surah, ayah, confidence }
```

### Design Patterns

| Pattern | Implementation |
|---|---|
| **Adapter** | `AudioFormatAdapter` wraps multi-format audio I/O (MP3, WAV) behind a unified interface — the preprocessing pipeline never handles format-specific logic directly. |
| **Template Method** | `BaseRecognitionModel` defines the invariant inference algorithm (`preprocess → embed → classify → postprocess`); concrete models such as `ReciterModel` and `LocalizationModel` override only their classification heads — the pipeline skeleton never changes. |


### Tech Stack

Python 3.11 · PyTorch 2.x · HuggingFace Transformers · librosa · scipy · Flask · scikit-learn (t-SNE) · Docker


## Modules

| Module | Responsibility |
|---|---|
| `preprocessing` | Adaptive audio pipeline — format normalization, VAD, voice-optimized filtering, noise reduction, and feature extraction; zero model logic, pure signal processing |
| `models/reciter` | Reciter identification — transformer encoder, speaker embedding head, transfer learning fine-tuning, and t-SNE validation utilities |
| `models/localization` | Surah–Ayah localization — transformer sequence matcher, dual classification heads (Surah + Ayah), cosine similarity analysis |
| `evaluation` | Validation suite — t-SNE cluster visualizations, similarity matrices, accuracy benchmarks, and embedding quality analysis |
| `data` | Dataset utilities — Quranic audio corpus loading, augmentation, and train/val/test splitting |


## Run Locally

### Steps
**1. Clone**
```bash
git clone https://github.com/fadynaeem/QuranRecognition.git
cd QuranRecognition
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model weights**
```bash
python scripts/download_weights.py
```

**4. Start the Flask API**
```bash
python api/app.py
```
