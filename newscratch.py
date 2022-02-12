from collections import defaultdict
import array
import numpy as np
import scipy.sparse as sp

"""Ripped heavily from scikit-learn CountVectorizer. 

https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/feature_extraction/text.py#L1184
"""


def _count_vocab(raw_documents):
    # Add a new value when a new vocabulary item is seen
    vocabulary = defaultdict()
    vocabulary.default_factory = vocabulary.__len__

    j_indices = []
    indptr = []
    values = array.array("i")

    indptr.append(0)
    for doc in raw_documents:
        feature_counter = {}
        for feature in doc:
            feature_idx = vocabulary[feature]
            if feature_idx not in feature_counter:
                feature_counter[feature_idx] = 1
            else:
                feature_counter[feature_idx] += 1

        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))

    j_indices = np.asarray(j_indices, dtype=np.int64)
    indptr = np.asarray(indptr, dtype=np.int64)
    values = np.frombuffer(values, dtype=np.intc)
    X = sp.csr_matrix(
        (values, j_indices, indptr),
        shape=(len(indptr) - 1, len(vocabulary)),
        dtype=np.int64,
    )
    X.sort_indices()
    return vocabulary, X
