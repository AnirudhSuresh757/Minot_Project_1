"""Quick local test for Resume_Model_V1."""

from model import match_resume_to_job, recommend_jobs


resume = """
Python developer with experience in Python, SQL, Pandas,
TensorFlow, machine learning, Git and data analysis.
"""

print("\n=== ONE RESUME -> ONE JOB ===")
print(match_resume_to_job(resume, job_index=0))

print("\n=== ONE RESUME -> TOP 10 UNIQUE ROLE RECOMMENDATIONS ===")
results = recommend_jobs(resume, top_n=10)
print(results.to_string(index=False))
