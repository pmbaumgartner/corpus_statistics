# Corpus Statistics: - A Very Basic (for now) spaCy Pipeline Component

If you want to know what tokens your pipeline has seen, this is the component for you.

**Example**

```python
from spacy.lang.en import English

# Use some example data
from datasets import load_dataset
dataset = load_dataset("imdb")
texts = dataset["train"]["text"]

# ✨ start the magic 
nlp = English()  # or spacy.load('a_model')
nlp.add_pipe("simple_corpus_stats")

for doc in nlp.pipe(texts):
    # ➡️ do your pipeline stuff! ➡️
    pass

corpus_stats = nlp.get_pipe("simple_corpus_stats")

# check if a token has been processed through this pipeline
token = "apple"
if token in corpus_stats:
    token_count = corpus_stats[token]
    print(f"'{token}' mentioned {token_count} times")

# 'apple' mentioned 24 times
```

It's got all your favorite [legomena](https://en.wikipedia.org/wiki/Hapax_legomenon) like `hapax` and `dis`.

```python
only_seen_once = len(corpus_stats.hapax_legomena)
percent_of_vocab = only_seen_once / corpus_stats.vocab_size
print(f"{percent_of_vocab*100:.1f}% tokens only occurred once.")
# 47.6% tokens only occurred once.

only_seen_twice = len(corpus_stats.dis_legomena)
percent_of_vocab_2x = only_seen_twice / corpus_stats.vocab_size
print(f"{percent_of_vocab_2x*100:.1f}% tokens occurred twice.")
# 12.3% tokens occurred twice.
```

We counted some things too:

```python
# corpus_stats.vocabulary is a collections.Counter 🔢
print(*corpus_stats.vocabulary.most_common(5), sep="\n")
# ('the', 289838)
# (',', 275296)
# ('.', 236702)
# ('and', 156484)
# ('a', 156282)

mean_doc_length = sum(corpus_stats.doc_lengths) / corpus_stats.corpus_length
print(f"Mean doc length: {mean_doc_length:.1f}")
```

**Use it when you train your models to keep track of what's was in your training data**

Example config:

```conf
...
[nlp]
lang = "en"
pipeline = ["simple_corpus_stats", ...]
...

[components]

[components.simple_corpus_stats]
factory = "simple_corpus_stats"
```

If you use it in a training config, your pipeline will likely see the same docs multiple times and in different order, due to the number of training epochs and batch size, so the vocab counter will be off. To correct for this, you can use the `rectify()`, e.g.

```python
import spacy

nlp = spacy.load("your_trained_model")
corpus_stats = nlp.get_pipe("simple_corpus_stats")

# There should be at least one word seen only once, but the
# min value is > 1 because we saw each doc several times
if min(corpus_stats.vocabulary.values()) == 1:
    corpus_stats.rectify()

assert min(corpus_stats.vocabulary.values()) == 1



