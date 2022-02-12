from functools import cached_property, lru_cache
from typing import Tuple, List, Dict
from spacy.language import Language
from spacy.tokens import Doc
import spacy
from collections import Counter, defaultdict


@Language.factory("corpus_statistics")
def create_acronym_component(nlp: Language, name: str):
    return CorpusStatistics(nlp)


class CorpusStatistics:
    def __init__(self, nlp: Language):
        self.corpus_counter: Counter = Counter()
        self.doc_lengths: List[int] = []
        self.nlp = nlp
        self.stopwords: set = nlp.Defaults.stop_words

    def __call__(self, doc: Doc) -> Doc:
        self.corpus_counter.update((token.text for token in doc))
        self.doc_lengths = len(doc)
        return doc

    def __getitem__(self, key: str) -> int:
        # Counter returns 0 rather than KeyError
        if self.corpus_counter[key] == 0:
            raise KeyError(f"'{key}' not in corpus (appears 0 times)")
        return self.corpus_counter[key]

    def _lowercase_counter(self, counter: Counter) -> Counter:
        lowercounter: Dict[str, int] = defaultdict(int)
        for token, count in counter.items():
            lowercounter[token.lower()] += count
        return Counter(lowercounter)

    def _exclude_stopwords(self, counter: Counter) -> Counter:
        # We could make reference to self.corpus_counter here
        # but we want to chain this so an arbitrary counter makes sense
        new_counter = {
            token: count
            for token, count in counter.items()
            if token not in self.stopwords
        }
        return Counter(new_counter)

    def _exclude_punctuation(self, counter: Counter) -> Counter:
        new_counter = {
            token: count
            for token, count in counter.items()
            if not self.nlp.vocab[token].is_punct
        }
        return Counter(new_counter)

    def get_vocab(
        self,
        lowercase: bool = False,
        exclude_stopwords: bool = False,
        exclude_punctuation: bool = False,
    ) -> Counter:
        counter = Counter(self.corpus_counter)
        if not any((lowercase, exclude_punctuation, exclude_stopwords)):
            # if no modifications, we can 'short-circut' this return
            return counter
        if lowercase:
            counter = self._lowercase_counter(counter)
        if exclude_stopwords:
            counter = self._exclude_stopwords(counter)
        if exclude_punctuation:
            counter = self._exclude_punctuation(counter)
        return counter

    @cached_property
    def vocab_size(self) -> int:
        return len(self.corpus_counter)

    @cached_property
    def token_count(self) -> int:
        return sum(self.corpus_counter.values())

    @cached_property
    def type_token_ratio(self) -> float:
        return self.vocab_size / self.token_count

    @cached_property
    def hapax_legomena(self) -> List[str]:
        return [token for token, count in self.corpus_counter.items() if count == 1]

    @cached_property
    def dis_legomena(self) -> List[str]:
        return [token for token, count in self.corpus_counter.items() if count == 2]

    @cached_property
    def mid_range_tokens(self) -> List[str]:
        return [token for token, count in self.corpus_counter.items() if count >= 3]

    @lru_cache
    def frequency_distribution(self, m: int) -> float:
        """Returns the proportion of the vocab that appears `m` times.

        Args:
            m (int): Count of token appearances

        Returns:
            float: Proportion of vocabulary that appears `m` times.
        """
        return (
            len([token for token, count in self.corpus_counter.items() if count == m])
            / self.vocab_size
        )


nlp = spacy.blank("en")
nlp.add_pipe("corpus_statistics")

import spacy
from datasets import load_dataset

from timeit import default_timer


dataset = load_dataset("imdb")
texts = dataset["train"]["text"]

start = default_timer()
for doc in nlp.pipe(texts):
    pass
end = default_timer()
print(f"{end - start:.3f}s")

corpus_statistics = nlp.get_pipe("corpus_statistics")
corpus_statistics

corpus_statistics["she"]
