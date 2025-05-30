# Quran Recognition

An AI-powered toolkit for Quranic verse recognition, including audio processing, text analysis, and reader identification.

## Features

- **Verse Recognition**: Identify Quranic verses from text or audio input
- **Reader Identification**: Detect and verify Quran reciters
- **Audio Processing**: Convert Quranic recitations to text
- **Embedding Generation**: Create semantic vector representations of verses
- **Vector Database**: Efficient storage and retrieval of verse data


## Technologies Used
- **Core Language**: Python
- **Audio Transcription**: OpenAI Whisper / Hugging Face Transformers
- **Audio Embeddings**: OpenL3 (for audio feature extraction)
- **Text Embeddings**: Sentence Transformers (e.g., AraBERT, Arabic BERT)
- **Vector Databases**: Pinecone / FAISS (for efficient similarity search)
- **ML Frameworks**: PyTorch / TensorFlow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fadynaeem/GP.git
   cd GP/quran_verse_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```
