# V3-RS — Final Resume–Job Matching Model

V3 is the final version of the resume–job matching model.

## Evolution

V1 → TF-IDF + cosine similarity + explicit skill matching

V2 → supervised CatBoost ranker with richer matching features

V3-RS → semantic similarity + semantic role-title matching +
weighted skill coverage with a supervised ExtraTrees ranker

## Final model

- Model: ExtraTreesRegressor
- Training pairs: 568,210
- Trees: 200
- min_samples_leaf: 2
- Features: 9

### Feature order

1. Cosine
2. Skill_Score
3. Normalized_Cosine
4. Required_Skill_Count
5. Matched_Skill_Count
6. Role_Similarity
7. Semantic_Similarity
8. Role_Title_Semantic_Similarity
9. Weighted_Skill_Score

## V3 improvements

### Semantic similarity
Uses MiniLM-based embeddings to capture semantic similarity between resume and job text.

### Role-title semantic similarity
Adds semantic matching between experience-title evidence in the resume and the job title.

### Weighted skill score
Uses job-level skill frequency to give relatively greater weight to rarer required skills.

## Final common-benchmark results

| Model | Top-1 | Hit@3 | Hit@5 | NDCG@5 | Pairwise |
|---|---:|---:|---:|---:|---:|
| V1 | 63.43% | 87.80% | 94.88% | 78.16% | 77.05% |
| V2 | 65.91% | 89.53% | 97.09% | 80.95% | 80.68% |
| V3-RS | 90.91% | 98.43% | 99.33% | 94.97% | 95.10% |

Benchmark:
- 25,400 sampled rows
- 10 trials
- 308 held-out resumes

## Full inference

The final V3-RS model scored all 668,360 possible resume–job pairs and generated Top-20 recommendations for all 1,540 resumes.

The trained model file is intentionally distributed separately because it is approximately 3.3 GB.
