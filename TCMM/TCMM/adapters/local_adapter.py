from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
import numpy as np
import os
import logging

# Kill "Batches:" progress bars from SentenceTransformer.encode()
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class LocalEmbeddingAdapter:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Define local cache path
        # TCCM/models/onnx/{model_name_sanitized}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models", "onnx")
        os.makedirs(models_dir, exist_ok=True)
        
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        model_path = os.path.join(models_dir, safe_name)
        
        # Use SentenceTransformer on GPU (fast) instead of ONNX on CPU (slow)
        import torch as _torch
        _gpu_ok = False
        if _torch.cuda.is_available():
            try:
                _test = _torch.zeros(1, device="cuda")
                del _test
                _gpu_ok = True
            except (RuntimeError, _torch.cuda.OutOfMemoryError):
                print(f"[LocalEmbeddingAdapter] GPU busy/OOM — falling back to CPU")
        if _gpu_ok:
            try:
                from sentence_transformers import SentenceTransformer as _ST
                print(f"[LocalEmbeddingAdapter] Loading SentenceTransformer on GPU ({_torch.cuda.get_device_name(0)})")
                self._st_model = _ST(model_name, device="cuda")
                self._use_st = True
                self.tokenizer = None
                self.model = None
            except (RuntimeError, _torch.cuda.OutOfMemoryError):
                print(f"[LocalEmbeddingAdapter] GPU failed during load — falling back to CPU")
                _gpu_ok = False
        if not _gpu_ok:
            self._use_st = False
            if os.path.exists(model_path):
                print(f"[LocalEmbeddingAdapter] Loading ONNX model from: {model_path} (CPU)")
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    model_path,
                    export=False,
                    provider="CPUExecutionProvider"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                print(f"[LocalEmbeddingAdapter] Exporting ONNX model to: {model_path} (CPU)")
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    model_name,
                    export=True,
                    provider="CPUExecutionProvider"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"[LocalEmbeddingAdapter] Saving model to cache...")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

        if self._use_st:
            self.dimension = self._st_model.get_sentence_embedding_dimension()
        else:
            self.dimension = self.model.config.hidden_size

    def _encode(self, texts, batch_size=128):
        # Determine batch size - respect provided arg or allow chunking if needed
        # For now, simplistic approach without explicit loop chunking as TCMM manages batch sizes
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Mean Pooling - Take attention mask into account for correct averaging
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def validate_embedding_model(self):
        """
        Simple health check.
        """
        # Test with prefix
        test = self.embed("validation test")
        assert len(test) == self.dimension
        assert isinstance(test, list)
        assert isinstance(test[0], float)

    def embed(self, text):
        """
        Embed a single string (passage). Returns list of floats.
        Adds 'passage: ' prefix for BGE.
        """
        if not text:
            return []
        if self._use_st:
            vec = self._st_model.encode(f"passage: {text}", normalize_embeddings=True, show_progress_bar=False)
            return vec.tolist()
        vec = self._encode([f"passage: {text}"])[0]
        return vec.tolist()

    def embed_batch(self, texts):
        """
        Embed a list of strings (passages). Returns list of list of floats.
        Adds 'passage: ' prefix for BGE.
        """
        if not texts:
            return []
        prefixed = [f"passage: {t}" for t in texts]
        if self._use_st:
            vecs = self._st_model.encode(prefixed, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            return vecs.tolist()
        vecs = self._encode(prefixed)
        return vecs.tolist()

    def embed_query(self, text):
        """
        Embed a query string.
        Adds 'query: ' prefix for BGE.
        """
        if not text:
            return []
        if self._use_st:
            vec = self._st_model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
            return vec.tolist()
        vec = self._encode([f"query: {text}"])[0]
        return vec.tolist()

    def embed_query_batch(self, texts):
        """
        Embed a batch of queries.
        Adds 'query: ' prefix for BGE.
        """
        if not texts:
            return []
            
        prefixed = [f"query: {t}" for t in texts]
        vecs = self._encode(prefixed)
        return vecs.tolist()
