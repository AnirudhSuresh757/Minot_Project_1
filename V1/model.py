"""Resume–Job Matching Model V1.

Baseline similarity-based NLP model for resume/job matching and job recommendation.

Core pipeline:
    text normalization
    -> TF-IDF (unigrams + bigrams)
    -> cosine similarity
    + explicit skill matching
    -> score fusion
    -> ranking / recommendation

V1 score for resume-to-job matching:
    final = 0.25 * normalized_cosine + 0.75 * skill_score

Important:
- The TF-IDF vectorizer is FITTED ONCE on the bundled reference corpus.
- New resumes are transformed with transform(), never fit_transform().
- This is a relevance/ranking baseline, not a calibrated probability model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PACKAGE_DIR = Path(__file__).resolve().parent

COSINE_WEIGHT = 0.25
SKILL_WEIGHT = 0.75

# -----------------------------
# Load bundled artifacts
# -----------------------------

match_vectorizer = joblib.load(PACKAGE_DIR / "tfidf_vectorizer.pkl")

candidate_df = pd.read_pickle(PACKAGE_DIR / "candidate_df.pkl").reset_index(drop=True)
entry_job_df = pd.read_pickle(PACKAGE_DIR / "entry_job_df.pkl").reset_index(drop=True)

required_candidate_cols = {"resume_text"}
required_job_cols = {"Title", "Skills", "job_text"}

missing_candidate_cols = required_candidate_cols - set(candidate_df.columns)
missing_job_cols = required_job_cols - set(entry_job_df.columns)

if missing_candidate_cols:
    raise ValueError(f"candidate_df.pkl is missing columns: {sorted(missing_candidate_cols)}")
if missing_job_cols:
    raise ValueError(f"entry_job_df.pkl is missing columns: {sorted(missing_job_cols)}")

# Precompute the fixed reference-space vectors used by V1.
candidate_tfidf = match_vectorizer.transform(candidate_df["resume_text"].fillna(""))
job_tfidf = match_vectorizer.transform(entry_job_df["job_text"].fillna(""))


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_technical_terms(text: Any) -> str:
    """Normalize common technical expressions to stable matching forms."""
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    replacements = {
        "asp.net": "aspnet",
        ".net core": "dotnetcore",
        ".net framework": "dotnetframework",
        ".net": "dotnet",
        "c#": "csharp",
        "c++": "cpp",
        "node.js": "nodejs",
        "node js": "nodejs",
        "typescript": "typescript",
        "javascript": "javascript",
    }

    # Keep the same replacement behavior as the notebook baseline.
    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def normalize_skill(skill: Any) -> str:
    """Normalize a required skill before exact boundary-aware matching."""
    if skill is None or pd.isna(skill):
        return ""

    skill = str(skill).lower().strip()
    skill = re.sub(r"\b(basics?|fundamentals?)\b", "", skill)
    skill = normalize_technical_terms(skill)
    return re.sub(r"\s+", " ", skill).strip()


# =========================================================
# SKILL MATCHING
# =========================================================

def skill_exists(skill: str, resume_text: str) -> bool:
    """Check whether a normalized skill occurs as a complete token/phrase."""
    if not skill:
        return False
    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return re.search(pattern, resume_text) is not None


def get_required_skills(job_skills: Any) -> list[str]:
    """Split a semicolon-separated job skill list and normalize each skill."""
    if job_skills is None or pd.isna(job_skills):
        return []

    skills: list[str] = []
    for skill in str(job_skills).split(";"):
        normalized = normalize_skill(skill)
        if normalized:
            skills.append(normalized)
    return skills


def get_skill_match(resume_text: str, required_skills: list[str]):
    """Return matched skills, missing skills and matched/required skill ratio."""
    matched: list[str] = []
    missing: list[str] = []

    for skill in required_skills:
        if skill_exists(skill, resume_text):
            matched.append(skill)
        else:
            missing.append(skill)

    score = len(matched) / len(required_skills) if required_skills else 0.0
    return matched, missing, score


# =========================================================
# ROLE FAMILY NORMALIZATION FOR RECOMMENDATIONS
# =========================================================

def get_role_family(title: Any) -> str:
    """Map obvious title variants to a compact role-family label.

    This is presentation/deduplication logic, not a learned classifier.
    Only explicit, conservative rules are used in V1.
    """
    title = "" if title is None or pd.isna(title) else str(title)
    title = title.lower().strip()

    # Remove junior/entry-level wording so variants do not duplicate the list.
    title = re.sub(
        r"\b(fresher|entry[\s-]?level|junior|intern(?:ship)?|trainee|graduate)\b",
        "",
        title,
    )
    title = re.sub(r"[-–—]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    # Conservative role-family rules.
    if "data scientist" in title or "data science" in title:
        return "Data Science"
    if "machine learning" in title:
        return "Machine Learning Engineer"
    if "data engineer" in title:
        return "Data Engineer"
    if "data analyst" in title:
        return "Data Analyst"
    if "software engineer" in title:
        return "Software Engineer"
    if "software developer" in title:
        return "Software Developer"
    if "python developer" in title:
        return "Python Developer"
    if "java developer" in title:
        return "Java Developer"
    if ".net developer" in title or "dotnet developer" in title:
        return ".NET Developer"
    if "android developer" in title:
        return "Android Developer"
    if "cybersecurity analyst" in title or "cyber security analyst" in title:
        return "Cybersecurity Analyst"
    if "ai engineer" in title:
        return "AI Engineer"

    return title


# =========================================================
# INTERNAL VALIDATION / SCALING HELPERS
# =========================================================

def _validate_job_index(job_index: int) -> int:
    job_index = int(job_index)
    if not 0 <= job_index < len(entry_job_df):
        raise IndexError(
            f"job_index must be between 0 and {len(entry_job_df) - 1}"
        )
    return job_index


def _validate_top_n(top_n: int) -> int:
    return max(1, int(top_n))


def _minmax(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value > min_value:
        return (values - min_value) / (max_value - min_value)
    return np.zeros_like(values, dtype=float)


# =========================================================
# EXISTING CANDIDATE RANKING
# =========================================================

def rank_job(job_index: int, top_n: int = 10) -> pd.DataFrame:
    """Rank the bundled early-career candidate pool for one job listing."""
    job_index = _validate_job_index(job_index)
    top_n = _validate_top_n(top_n)

    cosine_scores = cosine_similarity(
        job_tfidf[job_index], candidate_tfidf
    )[0]
    normalized_cosine = _minmax(cosine_scores)

    required_skills = get_required_skills(entry_job_df.iloc[job_index]["Skills"])

    skill_scores: list[float] = []
    matched_list: list[list[str]] = []
    missing_list: list[list[str]] = []

    for resume in candidate_df["resume_text"].fillna(""):
        matched, missing, score = get_skill_match(
            normalize_technical_terms(resume), required_skills
        )
        skill_scores.append(score)
        matched_list.append(matched)
        missing_list.append(missing)

    skill_scores_np = np.asarray(skill_scores, dtype=float)
    final_scores = (
        COSINE_WEIGHT * normalized_cosine
        + SKILL_WEIGHT * skill_scores_np
    )

    results = pd.DataFrame({
        "Resume_Index": np.arange(len(candidate_df)),
        "Cosine_Similarity": cosine_scores,
        "Normalized_Cosine": normalized_cosine,
        "Skill_Score": skill_scores_np,
        "Final_Score": final_scores,
        "Matched_Skills": matched_list,
        "Missing_Skills": missing_list,
    })

    return (
        results.sort_values("Final_Score", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )


# =========================================================
# NEW RESUME -> ONE SELECTED JOB
# =========================================================

def match_resume_to_job(resume_text: str, job_index: int) -> dict[str, Any]:
    """Match one new resume against one selected job listing."""
    if resume_text is None or not str(resume_text).strip():
        raise ValueError("resume_text must contain non-empty resume text.")

    job_index = _validate_job_index(job_index)
    normalized_resume = normalize_technical_terms(resume_text)
    job = entry_job_df.iloc[job_index]

    resume_vector = match_vectorizer.transform([normalized_resume])
    job_vector = job_tfidf[job_index]

    cosine_score = float(cosine_similarity(job_vector, resume_vector)[0][0])

    # Preserve the selected-job baseline convention:
    # new resume score is normalized relative to the bundled candidate pool.
    pool_cosines = cosine_similarity(job_vector, candidate_tfidf)[0]
    normalized_cosine = float(np.clip(
        _minmax(np.append(pool_cosines, cosine_score))[-1], 0.0, 1.0
    )) if pool_cosines.size else 0.0

    required_skills = get_required_skills(job["Skills"])
    matched, missing, skill_score = get_skill_match(
        normalized_resume, required_skills
    )

    final_score = (
        COSINE_WEIGHT * normalized_cosine
        + SKILL_WEIGHT * skill_score
    )

    return {
        "job_index": job_index,
        "job_title": job["Title"],
        "cosine_similarity": cosine_score,
        "normalized_cosine": normalized_cosine,
        "skill_score": float(skill_score),
        "final_score": float(final_score),
        "matched_skills": matched,
        "missing_skills": missing,
    }


# =========================================================
# NEW RESUME -> ALL JOBS -> TOP UNIQUE ROLE RECOMMENDATIONS
# =========================================================

def recommend_jobs(resume_text: str, top_n: int = 10) -> pd.DataFrame:
    """Recommend top unique role families from all bundled job listings.

    For recommendation mode, the resume is compared with all 434 job listings.
    Cosine scores are Min-Max normalized across those job listings for this
    particular resume, because the task is to rank jobs relative to one another.
    Skill matching is then combined using the V1 0.25/0.75 weights.
    """
    if resume_text is None or not str(resume_text).strip():
        raise ValueError("resume_text must contain non-empty resume text.")

    top_n = _validate_top_n(top_n)
    normalized_resume = normalize_technical_terms(resume_text)

    resume_vector = match_vectorizer.transform([normalized_resume])

    # One resume against every job in the fixed 434-job corpus.
    cosine_scores = cosine_similarity(job_tfidf, resume_vector).ravel()
    normalized_cosine = _minmax(cosine_scores)

    results: list[dict[str, Any]] = []

    for job_index, job in entry_job_df.iterrows():
        required_skills = get_required_skills(job["Skills"])
        matched, missing, skill_score = get_skill_match(
            normalized_resume, required_skills
        )

        final_score = (
            COSINE_WEIGHT * float(normalized_cosine[job_index])
            + SKILL_WEIGHT * float(skill_score)
        )

        results.append({
            "Job_Index": int(job_index),
            "JobID": job.get("JobID", None),
            "Job_Title": job["Title"],
            "Role": get_role_family(job["Title"]),
            "Cosine_Similarity": float(cosine_scores[job_index]),
            "Normalized_Cosine": float(normalized_cosine[job_index]),
            "Skill_Score": float(skill_score),
            "Final_Score": float(final_score),
            "Matched_Skills": matched,
            "Missing_Skills": missing,
        })

    results_df = pd.DataFrame(results)

    # Keep the highest-scoring listing for each canonical role family.
    results_df = results_df.sort_values(
        "Final_Score", ascending=False
    )
    results_df = results_df.drop_duplicates(
        subset="Role", keep="first"
    )

    return results_df.reset_index(drop=True).head(top_n)


# Backward-compatible alias for code that used the old variable name.
vectorizer = match_vectorizer
