import array
import functools
from collections import Counter, defaultdict
from functools import cached_property, lru_cache
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .vocabelement import VocabElement, create_all_vocab, create_vocab


def require_frozen(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.frozen:
            raise ValueError("Must call freeze() before use")
        return func(self, *args, **kwargs)

    return wrapper


@Language.factory("corpus_statistics", default_config={"lowercase": False})
def create_corpus_statistics_component(nlp: Language, name: str, lowercase: bool):
    return CorpusStatistics(nlp, lowercase=lowercase)


class CorpusStatistics:
    def __init__(self, nlp: Language, lowercase: bool):
        self.corpus_counter: Counter = Counter()
        self.lowercase = lowercase
        self.nlp = nlp
        self.stopwords: set = nlp.Defaults.stop_words

        self.vocabulary: Dict[str, int] = defaultdict()
        self.vocabulary.default_factory = self.vocabulary.__len__

        self._j_indices: List[int] = []
        self._indptr: List[int] = [0]
        self._values = array.array("i")

        self.frozen = False

    def __call__(self, doc: Doc) -> Doc:
        # TODO: Raise Error if Frozen
        feature_counter: Dict[int, int] = {}
        for token in doc:
            token_text = token.text if not self.lowercase else token.text.lower()
            feature_idx = self.vocabulary[token_text]
            if feature_idx not in feature_counter:
                feature_counter[feature_idx] = 1
            else:
                feature_counter[feature_idx] += 1
        self._j_indices.extend(feature_counter.keys())
        self._values.extend(feature_counter.values())
        self._indptr.append(len(self._j_indices))
        return doc

    def freeze(self) -> None:
        j_indices = np.asarray(self._j_indices, dtype=np.int64)
        indptr = np.asarray(self._indptr, dtype=np.int64)
        values = np.frombuffer(self._values, dtype=np.intc)
        self.X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.vocabulary)),
            dtype=np.int64,
        )
        self.X.sort_indices()
        self.X = self.X.tocsc()

        self.vocab_counts = self.X.sum(axis=0)

        # freeze vocabulary: convert from defaultdict
        self.vocabulary = dict(self.vocabulary)

        self.frozen = True

    @require_frozen
    def __getitem__(self, key: str) -> int:
        return create_vocab(key, self.X, self.vocabulary)

    @cached_property  # type: ignore
    @require_frozen
    def all_vocabulary(self):
        return create_all_vocab(self.X, self.vocabulary)

    @require_frozen
    def __contains__(self, key: str) -> int:
        return key in self.vocabulary

    def __len__(self):
        raise ValueError(
            "Length is ambiguous. Do you mean vocab_size, token_count, or corpus_length?"
        )

    @cached_property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @cached_property  # type: ignore
    @require_frozen
    def token_count(self) -> int:
        return self.X.sum()

    @cached_property  # type: ignore
    @require_frozen
    def corpus_length(self) -> int:
        return self.X.shape[0]

    @cached_property  # type: ignore
    @require_frozen
    def type_token_ratio(self) -> float:
        return self.vocab_size / self.token_count

    # @cached_property  # type: ignore
    @require_frozen
    def _vocab_index_by_count(self, count: int) -> Set[int]:  # type: ignore
        return set((self.vocab_counts == count).nonzero()[1])

    @cached_property  # type: ignore
    @require_frozen
    def hapax_legomena(self) -> List[str]:
        vocab_idx = self._vocab_index_by_count(count=1)
        return [vocab for vocab, idx in self.vocabulary.items() if idx in vocab_idx]

    @cached_property  # type: ignore
    @require_frozen
    def dis_legomena(self) -> List[str]:
        vocab_idx = self._vocab_index_by_count(2)
        return [vocab for vocab, idx in self.vocabulary.items() if idx in vocab_idx]

    @require_frozen  # type: ignore
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
