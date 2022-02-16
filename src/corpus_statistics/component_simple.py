from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import FrozenSet, List, Optional

import srsly
from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("simple_corpus_stats", default_config={"n_train": None})
def create_simple_corpus_stats_component(
    nlp: Language, name: str, n_train: Optional[int]
):
    return SimpleCorpusStatistics(nlp, n_train=n_train)


class SimpleCorpusStatistics:
    def __init__(self, nlp: Language, n_train: Optional[int]):
        self.vocabulary: Counter = Counter()
        self.doc_lengths: List[int] = []
        self.n_train = n_train

        self._known_length = n_train is not None
        self._call_count: int = 0

    def __call__(self, doc: Doc) -> Doc:
        if self._known_length and self._call_count >= self.n_train:
            return doc

        tokens = [token.text for token in doc]
        self.vocabulary.update(tokens)
        self.doc_lengths.append(len(tokens))
        self._call_count += 1
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
        srsly.write_msgpack(path / "doc_lengths.msgpack", self.doc_lengths)
        srsly.write_json(
            path / "vocabulary-meta.json",
            dict(
                n_train=self.n_train,
                _call_count=self._call_count,
            ),
        )

    def from_disk(self, path, exclude=tuple()):
        self.vocabulary = Counter(srsly.read_msgpack(path / "vocabulary.msgpack"))
        self.doc_lengths = srsly.read_msgpack(path / "doc_lengths.msgpack")
        meta = srsly.read_json(path / "vocabulary-meta.json")
        self.n_train = meta["n_train"]
        self._call_count = meta["_call_count"]
        return self
