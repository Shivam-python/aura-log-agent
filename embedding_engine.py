from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)