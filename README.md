# Resume–Job Matching System

Resume–job matching project developed as a minor project.

## Project progression

- V1: TF-IDF + cosine similarity + explicit skill matching
- V2: supervised CatBoost ranking using richer matching features
- V3-RS: semantic and weighted skill enhancements with an ExtraTrees ranker

## Final V3

The final V3-RS model uses 9 features:

1. Cosine
2. Skill_Score
3. Normalized_Cosine
4. Required_Skill_Count
5. Matched_Skill_Count
6. Role_Similarity
7. Semantic_Similarity
8. Role_Title_Semantic_Similarity
9. Weighted_Skill_Score

The final model and private/supporting artifacts are distributed separately because the trained model and resume-derived data are too large or inappropriate for a public GitHub repository.

## Final evaluation

On the common benchmark:

| Model | Top-1 | Hit@3 | Hit@5 | NDCG@5 | Pairwise |
|---|---:|---:|---:|---:|---:|
| V1 | 63.43% | 87.80% | 94.88% | 78.16% | 77.05% |
| V2 | 65.91% | 89.53% | 97.09% | 80.95% | 80.68% |
| V3-RS | 90.91% | 98.43% | 99.33% | 94.97% | 95.10% |

Final V3 inference scored 668,360 resume–job pairs and generated Top-20 recommendations for 1,540 resumes.

## Final model

`v3_final_extratrees_rs_568210.joblib` is shared separately from this public repository.
