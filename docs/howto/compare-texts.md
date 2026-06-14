---
icon: lucide/git-compare
description: Measure semantic and cross-lingual similarity between two texts using ISCC Semantic Text-Codes.
---

# Compare texts

This guide shows how to measure how similar two texts are in meaning — including across languages —
by comparing their Semantic Text-Codes.

Similarity is computed from the codes, not the original text. You can store or share the codes and
still compare content without keeping the source documents.

## Compare two whole documents

Generate a code for each text, then measure the Hamming distance between them with
`iscc_distance()`. It returns the number of differing bits — lower means more similar:

```python
import iscc_sct as sct

english = (
    "An ISCC applies to a specific digital asset and is a data-descriptor deterministically "
    "constructed from multiple hash digests using the algorithms and rules in this document. "
    "This document does not provide information on registration of ISCCs."
)
german = (
    "Ein ISCC bezieht sich auf ein bestimmtes digitales Gut und ist ein Daten-Deskriptor, der "
    "deterministisch aus mehreren Hash-Digests unter Verwendung der Algorithmen und Regeln in "
    "diesem Dokument erstellt wird. Dieses Dokument enthält keine Informationen über die "
    "Registrierung von ISCCs."
)

a = sct.create(english)
b = sct.create(german)

print(sct.iscc_distance(a.iscc, b.iscc))
# 3
```

`iscc_distance()` strips the `ISCC:` prefix and the 2-byte header, then compares the code bodies.
The two codes must be the same bit-length, or it raises `ValueError`.

## Read the distance

Distance is measured in bits, so the scale depends on the code length. Compare a translation against
an unrelated text at the same length to see the spread:

```python
import iscc_sct as sct

unrelated = (
    "The recipe calls for two cups of flour, a pinch of salt, and three ripe bananas mashed "
    "until smooth before folding the mixture into the buttered baking tin."
)
u = sct.create(unrelated)

print(sct.iscc_distance(a.iscc, b.iscc))  # translation
# 3
print(sct.iscc_distance(a.iscc, u.iscc))  # unrelated
# 33
```

A translation sits a few bits apart; unrelated content sits near half the bit-length away (random
codes differ in about 50% of their bits).

## Choose a bit-length

Longer codes spread similar and dissimilar texts further apart, which makes a threshold easier to
pick. The same three texts at 256 bits:

```python
import iscc_sct as sct

a256 = sct.create(english, bits=256)
b256 = sct.create(german, bits=256)
u256 = sct.create(unrelated, bits=256)

print(sct.iscc_distance(a256.iscc, b256.iscc))  # translation
# 23
print(sct.iscc_distance(a256.iscc, u256.iscc))  # unrelated
# 123
```

| Goal                                       | Suggested length |
| ------------------------------------------ | ---------------- |
| Compact storage, coarse matching           | 64 bits          |
| Balanced precision                         | 128 bits         |
| Finest discrimination between near-matches | 256 bits         |

Both codes in a comparison must use the same `bits` value. See [configuration](configuration.md) for
how to set the default.

## Score similarity from raw digests

`iscc_distance()` works on code strings. To compare raw digests directly — for example the output of
`soft_hash_text_semantic()` — use `cosine_similarity()`, which scales the distance to a `-100` to
`+100` score:

```python
import iscc_sct as sct

a = sct.soft_hash_text_semantic("An ISCC applies to a specific digital asset.")
b = sct.soft_hash_text_semantic("Ein ISCC bezieht sich auf ein bestimmtes digitales Gut.")

print(sct.cosine_similarity(a, b))
# 71
```

`hamming_distance()` returns the raw bit distance between two equal-length digests if you want the
unscaled value. Both functions raise `ValueError` when the inputs differ in length.

## Match passages instead of whole documents

To find which parts of two documents are similar — rather than scoring them as a whole — use
granular features and `granular_similarity()`. See [granular features](granular-features.md).

## Related pages

- **[Granular features](granular-features.md)** — Chunk-level matching with simprints.
- **[How it works](../explanation/how-it-works.md)** — Why distance reflects meaning.
- **[API reference](../reference/api.md)** — Signatures for every similarity function.
