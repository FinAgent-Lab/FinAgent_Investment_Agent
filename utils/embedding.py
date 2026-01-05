import torch
from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDING_MODEL = None

def _get_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:

        # Pydantic v2 호환: model_kwargs 사용
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'}
        )

    return _EMBEDDING_MODEL

async def get_embedding(text: str) -> list[float]:
    model = _get_model()

    embedding = model.embed_query(text)

    return embedding
