"""Lightweight local embeddings using ONNX Runtime without PyTorch."""

from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_runtime():
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="onnx/model.onnx",
    )
    tokenizer_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="tokenizer.json",
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return session, tokenizer


def encode_texts(texts, batch_size=8):
    """Return normalized float32 embeddings with shape (n_texts, 384)."""
    texts = list(texts)
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    session, tokenizer = _load_runtime()
    input_names = {item.name for item in session.get_inputs()}
    batches = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer.encode_batch(
            [str(text) for text in batch],
            add_special_tokens=True,
        )
        max_length = max(len(item.ids) for item in encoded)

        input_ids = np.zeros((len(batch), max_length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)

        for row, item in enumerate(encoded):
            length = len(item.ids)
            input_ids[row, :length] = item.ids
            attention_mask[row, :length] = item.attention_mask

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in input_names:
            token_type_ids = np.zeros_like(input_ids)
            for row, item in enumerate(encoded):
                token_type_ids[row, :len(item.type_ids)] = item.type_ids
            inputs["token_type_ids"] = token_type_ids

        token_embeddings = session.run(None, inputs)[0]
        mask = attention_mask[..., None].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1) / np.maximum(
            mask.sum(axis=1), 1e-9
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)
        batches.append(pooled.astype(np.float32))

    return np.vstack(batches)
