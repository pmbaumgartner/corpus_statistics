import hashlib
from collections import Counter
from functools import cached_property, lru_cache
from pathlib import Path
from typing import FrozenSet, List

import srsly
from spacy.language import Language
from spacy.tokens import Doc


@lru_cache
def _hash_cache(text: str) -> str:
    return hashlib.md5(text.encode("utf8")).hexdigest()


@Language.factory("simple_corpus_stats")
def create_simple_corpus_stats_component(nlp: Language, name: str):
    return SimpleCorpusStatistics(nlp)


class SimpleCorpusStatistics:
    def __init__(self, nlp: Language):
        self.vocabulary: Counter = Counter()
        self.doc_lengths: List[int] = []
        self.is_rectified: bool = False

        self._call_count: int = 0
        self._doc_hash_counter: Counter = Counter()
        self._seen_full_dup: bool = False
        self._inferred_corpus_length: int = 0

    def __call__(self, doc: Doc) -> Doc:
        self._call_count += 1
        tokens = [token.text for token in doc]
        self.vocabulary.update(tokens)
        self.doc_lengths.append(len(tokens))

        if not self._seen_full_dup:
            doc_hash = _hash_cache(doc.text)
            self._doc_hash_counter.update([doc_hash])
        if not self._seen_full_dup and min(self._doc_hash_counter.values()) == 2:
            # When we've seen every document twice, it means we've iterated through
            # the corpus 2x, so we can infer the length at calls / 2
            self._seen_full_dup = True
            self._inferred_corpus_length = self._call_count // 2
            self._doc_hash_counter = Counter()
        return doc

    def __getitem__(self, key: str) -> int:
        return self.vocabulary[key]

    def __contains__(self, key: str) -> int:
        return key in self.vocabulary

    def __len__(self) -> int:
        return len(self.vocabulary)

    def rectify(self):
        if not self._seen_full_dup:
            raise ValueError(
                "Rectification not necessary."
                " The component hasn't been called on a corpus repeatedly."
            )
        if self.is_rectified:
            raise ValueError("Vocab already rectified.")

        n_repeats: int = self._call_count / self._inferred_corpus_length
        self.vocabulary = Counter(
            {k: v // n_repeats for k, v in self.vocabulary.items()}
        )
        self.doc_lengths = self.doc_lengths[: self._inferred_corpus_length]
        self.is_rectified = True

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
                is_inferred=self.is_rectified,
                _call_count=self._call_count,
                _full_duplicates=self._seen_full_dup,
                _inferred_corpus_length=self._inferred_corpus_length,
            ),
        )

    def from_disk(self, path, exclude=tuple()):
        self.vocabulary = Counter(srsly.read_msgpack(path / "vocabulary.msgpack"))
        self.doc_lengths = srsly.read_msgpack(path / "doc_lengths.msgpack")
        meta = srsly.read_json(path / "vocabulary-meta.json")
        self.is_rectified = meta["is_inferred"]
        self._call_count = meta["_call_count"]
        self._seen_full_dup = meta["_full_duplicates"]
        self._inferred_corpus_length = meta["_inferred_corpus_length"]
        return self
