from .random import RandomRetriever
from .random_same_project import RandomSameProjectRetriever

RETRIEVER_REGISTRY = {
    "random": RandomRetriever,
    "random_same_project": RandomSameProjectRetriever,
}
