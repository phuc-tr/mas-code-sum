from .bm25 import BM25Retriever
from .directory import DirectoryRetriever
from .hyde import HyDERetriever
from .random import RandomRetriever
from .random_same_project import RandomSameProjectRetriever

RETRIEVER_REGISTRY = {
    "random": RandomRetriever,
    "random_same_project": RandomSameProjectRetriever,
    "bm25": BM25Retriever,
    "directory": DirectoryRetriever,
    "hyde": HyDERetriever,
}
