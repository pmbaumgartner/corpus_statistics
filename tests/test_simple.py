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
def pipeline():
    nlp = spacy.blank("en")
    nlp.add_pipe("simple_corpus_stats")
    return nlp


@pytest.fixture
def pipeline_with_component(test_docs, pipeline):
    # Need to pass both as tuple, for some reason the component
    # doesn't persist when it's returned by itself from a fixture
    for _ in pipeline.pipe(test_docs):
        pass
    corpus_statistics = pipeline.get_pipe("simple_corpus_stats")
    return pipeline, corpus_statistics


def test_docs_basics(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    assert corpus_statistics.token_count == 50
    assert corpus_statistics.corpus_length == 5
    assert corpus_statistics.vocab_size == 16


def test_contains(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    assert "a" in corpus_statistics
    assert "b" in corpus_statistics
    assert "z" not in corpus_statistics


def test_doc_length(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    assert corpus_statistics.doc_lengths == [9, 10, 10, 10, 11]


def test_counts(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    assert corpus_statistics["a"] == 15
    assert corpus_statistics["b"] == 10


def test_ttr(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    assert np.isclose(corpus_statistics.type_token_ratio, 0.32)


def test_legomenon(pipeline_with_component: SimpleCorpusStatistics):
    pipeline, corpus_statistics = pipeline_with_component
    for hapax in corpus_statistics.hapax_legomena:
        assert corpus_statistics[hapax] == 1
    for dis in corpus_statistics.dis_legomena:
        assert corpus_statistics[dis] == 2


def test_persist(pipeline_with_component, tmp_path):
    pipeline, corpus_statistics = pipeline_with_component
    component_before_persist = corpus_statistics
    before_a_count = component_before_persist["a"]
    pipeline.to_disk(tmp_path)

    pipeline_loaded = spacy.load(tmp_path)
    component_after_persist = pipeline_loaded.get_pipe("simple_corpus_stats")

    after_a_count = component_after_persist["a"]
    # these next assertions are more of a fixture test
    assert before_a_count != 0
    assert after_a_count != 0
    assert len(component_before_persist.doc_lengths) != 0
    # actual persistance assertions
    assert before_a_count == after_a_count
    assert component_before_persist.doc_lengths == component_after_persist.doc_lengths


@pytest.fixture
def pipeline_with_test_docs_iter_2x(test_docs, pipeline):
    for doc in pipeline.pipe(test_docs):
        pass
    for doc in pipeline.pipe(test_docs):
        pass
    corpus_statistics = pipeline.get_pipe("simple_corpus_stats")
    return pipeline, corpus_statistics


def test_rectify(test_docs, pipeline_with_component, pipeline_with_test_docs_iter_2x):
    pipeline, corpus_statistics = pipeline_with_component
    pipeline_2x, corpus_statistics_2x = pipeline_with_test_docs_iter_2x
    assert corpus_statistics_2x._inferred_corpus_length == len(test_docs)
    # this is asserting the "wrong" vocab count
    assert min(corpus_statistics_2x.vocabulary.values()) > 1
    corpus_statistics_2x.rectify()
    assert corpus_statistics_2x.vocabulary == corpus_statistics.vocabulary
    assert min(corpus_statistics_2x.vocabulary.values()) == 1


def test_infer_error_not_duplicated(pipeline_with_component):
    pipeline, corpus_statistics = pipeline_with_component
    with pytest.raises(ValueError):
        corpus_statistics.rectify()


def test_infer_error_already_rectified(pipeline_with_test_docs_iter_2x):
    pipeline_2x, corpus_statistics_2x = pipeline_with_test_docs_iter_2x
    corpus_statistics_2x.rectify()
    with pytest.raises(ValueError):
        corpus_statistics_2x.rectify()
