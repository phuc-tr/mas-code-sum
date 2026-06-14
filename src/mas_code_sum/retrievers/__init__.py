from .bm25 import BM25Retriever
from .random import RandomRetriever

RETRIEVER_REGISTRY = {
    "random": RandomRetriever,
    "bm25": BM25Retriever,
}
