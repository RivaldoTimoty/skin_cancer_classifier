from langchain_community.embeddings import HuggingFaceEmbeddings


_embedding_model = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model
