---
icon: lucide/layers
description: Generate per-chunk simprints, choose offset and feature formats, reconstruct text, and match passages across documents.
---

# Granular features

This guide shows how to work with granular features: the per-chunk fingerprints that let you match
individual passages, locate them in the source text, and reconstruct content.

A document code summarizes a whole text in one code. Granular features instead describe each chunk
the text was split into, so you can compare documents passage by passage.

## Generate granular features

Pass `granular=True` to `create()`. Each chunk becomes a `Feature` with a simprint (its
similarity-preserving fingerprint), a character `offset`, a `size`, and the chunk `content`:

```python
import iscc_sct as sct

text = "This is some sample text. It can be a longer document or even an entire book."
meta = sct.create(text, bits=256, granular=True)

feature = meta.features[0].simprints[0]
print(feature.simprint, feature.offset, feature.size)
# XZjeSfdyVi0 0 77
print(feature.content)
# This is some sample text. It can be a longer document or even an entire book.
```

Short texts produce a single chunk. Longer texts produce many — see
[matching passages](#match-passages-across-documents) below.

## Choose a feature format

Granular features come in two interchangeable shapes:

| Format        | Shape                                                        | Produced by                |
| ------------- | ------------------------------------------------------------ | -------------------------- |
| Object-Format | a list of self-contained `Feature` objects                   | `create(granular=True)`    |
| Index-Format  | parallel arrays: `simprints`, `offsets`, `sizes`, `contents` | `gen_text_code_semantic()` |

`create()` returns Object-Format, which is easy to read and iterate. Convert between the two with
`to_index_format()` and `to_object_format()`:

```python
import iscc_sct as sct

meta = sct.create(text, bits=256, granular=True)  # Object-Format

index = meta.to_index_format()
print(index.features[0].simprints)  # ['XZjeSfdyVi0']
print(index.features[0].offsets)  # [0]
```

Index-Format stores related attributes in parallel arrays, which is compact for storage and bulk
indexing. The low-level `gen_text_code_semantic()` returns a plain dict already in Index-Format.

## Control the simprint length

Granular simprints have their own bit-length, set by `bits_granular` (default **64**). It is
independent of the document `bits`, so `bits=256, granular=True` still yields 64-bit simprints:

```python
import iscc_sct as sct

meta = sct.create(text, bits=256, granular=True, bits_granular=256)
digest = sct.decode_base64(meta.features[0].simprints[0].simprint)
print(len(digest) * 8)
# 256
```

Shorter simprints save space; longer simprints discriminate finer between near-matching passages.

## Reconstruct the original text

When features include both `offset` and `content`, `get_content()` stitches the chunks back into the
original text, removing the overlap between adjacent chunks:

```python
import iscc_sct as sct

en = (
    "The International Standard Content Code identifies digital content of any media type. "
    "It is generated algorithmically from the content itself, much like a cryptographic hash. "
    "Unlike a cryptographic hash, the ISCC preserves similarity between related items. "
    "Two near-duplicate files therefore receive two codes that are close to each other. "
    "This property supports deduplication, similarity clustering, and content provenance. "
    "The Semantic Text-Code extends these ideas to the meaning of text across languages. "
    "A translation of a document keeps a code that stays close to the original code. "
    "That makes cross-lingual search and matching practical without machine translation."
)
meta = sct.create(en, granular=True)

print(len(meta.features[0].simprints))  # number of chunks
# 2
print(meta.get_content() == en)
# True
```

`get_overlaps()` returns the overlapping text between consecutive chunks if you need to inspect the
chunk boundaries directly.

## Use byte offsets

Offsets and sizes are character positions by default. Enable `byte_offsets` to report UTF-8 byte
positions instead — useful for fetching a chunk by random access from a remote file:

```python
import iscc_sct as sct

meta = sct.create(en, granular=True, byte_offsets=True)
print(meta.features[0].byte_offsets)  # True
print(meta.features[0].simprints[1].offset)  # byte offset of the second chunk
```

## Match passages across documents

`granular_similarity()` compares the simprints of two `Metadata` objects and returns the matching
passages above a similarity threshold. Each result is a `(Feature, score, Feature)` tuple, where the
score is the `cosine_similarity` between the two simprints:

```python
import iscc_sct as sct

de = (
    "Der International Standard Content Code identifiziert digitale Inhalte jedes Medientyps. "
    "Er wird algorithmisch aus dem Inhalt selbst erzeugt, ähnlich einem kryptografischen Hash. "
    "Anders als ein kryptografischer Hash bewahrt der ISCC die Ähnlichkeit verwandter Objekte. "
    "Zwei nahezu identische Dateien erhalten daher zwei Codes, die nahe beieinander liegen. "
    "Diese Eigenschaft unterstützt Deduplizierung, Ähnlichkeits-Clustering und Herkunftsnachweis. "
    "Der Semantic Text-Code überträgt diese Ideen auf die Bedeutung von Text über Sprachen hinweg. "
    "Eine Übersetzung eines Dokuments behält einen Code, der nahe am Originalcode bleibt. "
    "Das macht sprachübergreifende Suche und Zuordnung ohne maschinelle Übersetzung praktikabel."
)

a = sct.create(en, granular=True)
b = sct.create(de, granular=True)

for feat_a, score, feat_b in sct.granular_similarity(a, b, threshold=80):
    print(score, feat_a.offset, feat_b.offset)
# 81 0 0
# 90 340 356
```

Each English passage is matched to its closest German counterpart, even though the offsets differ
because the languages produce text of different lengths. Raise `threshold` to keep only the
strongest matches; lower it to surface weaker ones. Only the single best match per passage in the
first document is returned.

## Related pages

- **[Compare texts](compare-texts.md)** — Whole-document similarity.
- **[Configuration](configuration.md)** — Defaults for `bits_granular`, chunking, and offsets.
- **[How it works](../explanation/how-it-works.md)** — How chunks and overlaps are produced.
