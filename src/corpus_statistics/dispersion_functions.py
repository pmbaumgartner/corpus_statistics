import math
from numbers import Number
from typing import Union

import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import rel_entr
from scipy.stats import entropy
from tqdm import tqdm

NATSPERBIT = np.log2(math.e)


def natstobits(nats: float) -> float:
    return nats * NATSPERBIT


def retreive_scalar(value: Union[np.ndarray, float]) -> Number:
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0]
    elif isinstance(value, Number):
        return value
    else:
        raise ValueError(
            f"{value} not Number or single element array - it's a {type(value)}"
        )


def all_stats(tdm: csc_matrix):
    doc_lengths = tdm.sum(axis=1)
    vocab_counts = tdm.sum(axis=0)
    corpus_len = tdm.shape[0]
    token_counts = tdm.sum()
    s = doc_lengths / token_counts
    # tdm_d = tdm.toarray()
    range_array = tdm.getnnz(axis=0)
    stats = []
    pbar_columns = tqdm(
        range(tdm.shape[1]), desc="Calculating Staistics for Each Token"
    )
    for column in pbar_columns:
        col_d = tdm.getcol(column).toarray()
        freq = col_d.sum()
        prop = freq / token_counts
        range_ = range_array[column]
        stdev = np.std(col_d, axis=0)
        vc = stdev / col_d.mean()
        # p = proportion of doc comprised of this token
        # s = percent of corpus comprised of tokens in this doc
        # v = percent of tokens in doc that are this token
        v = col_d / freq
        p = col_d / doc_lengths
        p_std = np.std(p)
        p_mean = np.mean(p)
        juilland_d = 1 - (p_std / p_mean) * (1 / np.sqrt(corpus_len - 1))
        carroll_d2 = entropy(p, base=2) / np.log2(corpus_len)
        rosengren_sadj = ((np.sqrt(np.multiply(s, col_d))).sum() ** 2) / freq
        dp = (np.abs(np.subtract(v, s))).sum() * 0.5
        dp_norm = dp / (1 - np.min(s))
        kl = natstobits(rel_entr(v, s).sum())
        stats.append(
            dict(
                freq=retreive_scalar(freq),
                prop=retreive_scalar(prop),
                range=retreive_scalar(range_),
                stdev=retreive_scalar(stdev),
                vc=retreive_scalar(vc),
                juilland_d=retreive_scalar(juilland_d),
                carroll_d2=retreive_scalar(carroll_d2),
                rosengren_sadj=retreive_scalar(rosengren_sadj),
                dp=retreive_scalar(dp),
                dp_norm=retreive_scalar(dp_norm),
                kl=retreive_scalar(kl),
            )
        )
    return stats


# https://stackoverflow.com/questions/7769525/optimal-broadcasted-matrix-division-in-numpy-avoiding-temporary-arrays-or-not
