from typing import List

import numpy as np
import pytest
import spacy

from src.corpus_statistics import __version__, create_simple_corpus_stats_component
from src.corpus_statistics.component_simple import SimpleCorpusStatistics


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def test_docs() -> List[str]:
    """from https://stgries.info/research/ToApp_STG_Dispersion_PHCL.pdf p. 3

    Th. Gries S. (2020) Analyzing Dispersion.
    In: Paquot M., Gries S.T. (eds) A Practical Handbook of Corpus Linguistics.
    Springer, Cham.
    https://doi.org/10.1007/978-3-030-46216-1_5
    """
    docs_str = """
    b a m n i b e u p
    b a s a t b e w q n
    b c a g a b e s t a
    b a g h a b e a a t
    b a h a a b e a x a t
    """
    docs = [doc.strip() for doc in docs_str.strip().split("\n")]
    return docs


@pytest.fixture
def pipeline_with_component():
    nlp = spacy.blank("en")
    nlp.add_pipe("simple_corpus_stats")
    return nlp


@pytest.fixture
def pipeline_with_test_docs(test_docs, pipeline_with_component):
    for doc in pipeline_with_component.pipe(test_docs):
        pass
    return pipeline_with_component


@pytest.fixture
def component_with_test_docs(test_docs, pipeline_with_component):
    for doc in pipeline_with_component.pipe(test_docs):
        pass
    corpus_statistics = pipeline_with_component.get_pipe("simple_corpus_stats")
    return corpus_statistics


def test_docs_basics(component_with_test_docs: SimpleCorpusStatistics):
    assert component_with_test_docs.token_count == 50
    assert component_with_test_docs.corpus_length == 5
    assert component_with_test_docs.vocab_size == 16


def test_contains(component_with_test_docs: SimpleCorpusStatistics):
    assert "a" in component_with_test_docs
    assert "b" in component_with_test_docs


def test_doc_length(component_with_test_docs: SimpleCorpusStatistics):
    assert component_with_test_docs.doc_lengths == [9, 10, 10, 10, 11]


def test_counts(component_with_test_docs: SimpleCorpusStatistics):
    assert component_with_test_docs["a"] == 15
    assert component_with_test_docs["b"] == 10


def test_ttr(component_with_test_docs: SimpleCorpusStatistics):
    assert np.isclose(component_with_test_docs.type_token_ratio, 0.32)


def test_legomenon(component_with_test_docs: SimpleCorpusStatistics):
    for hapax in component_with_test_docs.hapax_legomena:
        assert component_with_test_docs[hapax] == 1
    for dis in component_with_test_docs.dis_legomena:
        assert component_with_test_docs[dis] == 2


def test_persist(pipeline_with_component, tmp_path):
    component_before_persist = pipeline_with_component.get_pipe("simple_corpus_stats")
    before_a_count = component_before_persist["a"]
    pipeline_with_component.to_disk(tmp_path)

    nlp2 = spacy.load(tmp_path)
    component_after_persist = nlp2.get_pipe("simple_corpus_stats")

    after_a_count = component_after_persist["a"]
    assert before_a_count == after_a_count
