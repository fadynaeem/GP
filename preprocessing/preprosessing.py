import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Config import Config


class EmbeddingService:
    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.token = Config.HF_TOKEN
        self.tokenizer = None
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.token
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            output_hidden_states=True
        )
        
    def get_embedding(self, text):
        if not text or not isinstance(text, str):
            return np.zeros(self.model.config.hidden_size)
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)
        
        return embeddings.numpy().flatten()