import os
from dotenv import load_dotenv
import sys
load_dotenv()


class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable is not set.")
        sys.exit(1)
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable is not set.")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "verse-index")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    
    # Model settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    if not EMBEDDING_MODEL_NAME:
        EMBEDDING_MODEL_NAME = os.getenv(
            "EMBEDDING_MODEL_NAME_DEFAULT", 
            "pourmand1376/arabic-quran-nahj-sahife"
        )
        
    WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME")
    if not WHISPER_MODEL_NAME:
        WHISPER_MODEL_NAME = os.getenv(
            "WHISPER_MODEL_NAME_DEFAULT",
            "tarteel-ai/whisper-base-ar-quran"
        )
    
    KMP_DUPLICATE_LIB_OK = "TRUE"
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))