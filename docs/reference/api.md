---
icon: lucide/code
description: Auto-generated reference for the public iscc-sct API - code generation, data model, options, and similarity functions.
---

# API reference

This page documents the public API re-exported from the `iscc_sct` package. Import it with
`import iscc_sct as sct`. The reference below is generated from the source docstrings.

## High-level API

The primary entry point. Returns a `Metadata` object with the code and, optionally, granular
features.

::: iscc_sct.create
    options:
      heading_level: 3

## Core functions

Lower-level building blocks of the processing pipeline.

::: iscc_sct.gen_text_code_semantic
    options:
      heading_level: 3

::: iscc_sct.code_text_semantic
    options:
      heading_level: 3

::: iscc_sct.soft_hash_text_semantic
    options:
      heading_level: 3

::: iscc_sct.embed_chunks
    options:
      heading_level: 3

## Data model

The result schema. `Metadata` holds the code and feature sets; `FeatureSet` and `Feature` carry the
granular data in either Index-Format or Object-Format.

::: iscc_sct.Metadata
    options:
      heading_level: 3

::: iscc_sct.FeatureSet
    options:
      heading_level: 3

::: iscc_sct.Feature
    options:
      heading_level: 3

## Options

The settings model. Configure it per call, with `ISCC_SCT_*` environment variables, or via
`override()`.

::: iscc_sct.SctOptions
    options:
      heading_level: 3

## Similarity and distance

Functions for comparing codes, digests, and granular features.

::: iscc_sct.iscc_distance
    options:
      heading_level: 3

::: iscc_sct.hamming_distance
    options:
      heading_level: 3

::: iscc_sct.cosine_similarity
    options:
      heading_level: 3

::: iscc_sct.granular_similarity
    options:
      heading_level: 3

## Codecs and helpers

Encoders, decoders, and utilities.

::: iscc_sct.encode_base32
    options:
      heading_level: 3

::: iscc_sct.decode_base32
    options:
      heading_level: 3

::: iscc_sct.encode_base64
    options:
      heading_level: 3

::: iscc_sct.decode_base64
    options:
      heading_level: 3

::: iscc_sct.char_to_byte_offsets
    options:
      heading_level: 3

::: iscc_sct.get_model
    options:
      heading_level: 3
