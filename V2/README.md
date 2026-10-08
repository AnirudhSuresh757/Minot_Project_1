# V2 — Supervised Resume–Job Ranker

V2 introduced supervised learning over richer resume–job matching features.

## Model

CatBoost Regressor.

## Rich 6 features

- Cosine
- Skill_Score
- Normalized_Cosine
- Required_Skill_Count
- Matched_Skill_Count
- Role_Similarity

## Training

100,000 labelled resume–job pairs were used during the V2 development.

V2 formed the supervised-learning foundation for the later V3 model.
