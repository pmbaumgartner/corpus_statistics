from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import FrozenSet, List

import srsly
from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("simple_corpus_stats")
def create_simple_corpus_stats_component(nlp: Language, name: str):
    return SimpleCorpusStatistics(nlp)


class SimpleCorpusStatistics:
    def __init__(self, nlp: Language):
        self.vocabulary: Counter = Counter()
        self.doc_lengths: List[int] = []

    def __call__(self, doc: Doc) -> Doc:
        # TODO: Raise Error if Frozen
        tokens = [token.text for token in doc]
        self.vocabulary.update(tokens)
        self.doc_lengths.append(len(tokens))
        return doc

    def __getitem__(self, key: str) -> int:
        return self.vocabulary[key]

    def __contains__(self, key: str) -> int:
        return key in self.vocabulary

    def __len__(self) -> int:
        return len(self.vocabulary)

    @cached_property
    def vocab_size(self) -> int:
        return len(self)

    @cached_property
    def token_count(self) -> int:
        return sum(self.doc_lengths)

    @cached_property
    def corpus_length(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def type_token_ratio(self) -> float:
        return self.vocab_size / self.token_count

    @cached_property
    def hapax_legomena(self) -> FrozenSet:
        return frozenset(
            [token for token, count in self.vocabulary.items() if count == 1]
        )

    @cached_property
    def dis_legomena(self) -> FrozenSet:
        return frozenset(
            [token for token, count in self.vocabulary.items() if count == 2]
        )

    def to_disk(self, path, exclude=tuple()):
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            path.mkdir()
        srsly.write_msgpack(path / "vocabulary.msgpack", dict(self.vocabulary))

    def from_disk(self, path, exclude=tuple()):
        self.vocabulary = Counter(srsly.read_msgpack(path / "vocabulary.msgpack"))
        return self
