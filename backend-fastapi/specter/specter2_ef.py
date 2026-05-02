import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from chromadb import EmbeddingFunction, Documents, Embeddings


class Specter2EmbeddingFunction(EmbeddingFunction):
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SPECTER2 on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        self.model.to(self.device)
        self.model.eval()
        print("SPECTER2 ready.")

    def __call__(self, input: Documents) -> Embeddings:
        inputs = self.tokenizer(
            input, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embeddings.cpu().numpy().tolist()
