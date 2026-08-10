from .bm25 import BM25CrossProjectRetriever, BM25Retriever
from .random import RandomRetriever

RETRIEVER_REGISTRY = {
    "random": RandomRetriever,
    "bm25": BM25Retriever,
    "bm25_cross_project": BM25CrossProjectRetriever,
}
