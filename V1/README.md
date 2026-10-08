# V1 — Baseline Resume–Job Matcher

V1 is the original interpretable baseline.

## Core approach

- TF-IDF with unigram and bigram features
- Cosine similarity
- Per-job Min-Max normalization of cosine scores
- Explicit required-skill matching
- Weighted fusion:

Final Score = 0.25 × Normalized Cosine + 0.75 × Skill Score

## Purpose

V1 established the original similarity-based resume–job ranking pipeline that was later extended by V2 and V3.
