import spacy
from datasets import load_dataset
from collections import Counter, defaultdict
from spacy.tokens import Doc
from typing import List, NewType, Dict
from itertools import chain


dataset = load_dataset("squad")
texts = dataset["train"]["question"][:1000]

nlp = spacy.blank("en")

tokenized_docs = list(nlp.tokenizer.pipe(texts))

CorpusCounter = NewType("CorpusCounter", Counter)


def count_tokens(docs: List[Doc]) -> CorpusCounter:
    return Counter([token.text for doc in docs for token in doc])


def tokens_lower(counter: CorpusCounter) -> CorpusCounter:
    lowercounter: Dict[str, int] = defaultdict(int)
    for token, count in counter.items():
        lowercounter[token.lower()] += count
    return Counter(lowercounter)


def hapax_legomena(counter: CorpusCounter) -> List[str]:
    return [token for token, count in counter.items() if count == 1]


def dis_legomena(counter: CorpusCounter) -> List[str]:
    return [token for token, count in counter.items() if count == 2]


def mid_range_frequencies(counter: CorpusCounter) -> List[str]:
    return [token for token, count in counter.items() if count >= 3]


def type_token_ratio(counter: CorpusCounter) -> float:
    return len(counter) / sum(counter.values())
