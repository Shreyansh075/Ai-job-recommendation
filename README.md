# AI Job Recommendation Engine

A Streamlit portfolio demo that ranks jobs against a candidate profile using cached
`all-MiniLM-L6-v2` sentence embeddings, skills coverage, and experience fit. TF-IDF
remains an explicit fallback for offline/no-transformer environments.

## Run locally

```powershell
pip install -r requirements.txt
python Train_model.py
python -m streamlit run app.py
```

Cover-letter generation uses the OpenAI API. Set `OPENAI_API_KEY` in the environment
before starting Streamlit; optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`).
The key is never stored in the project. Resume text is passed to OpenAI only when a
cover letter is explicitly requested.

## Features

- Skills match, experience match, overall match, and semantic similarity scores.
- Sample candidate profiles, resume upload and skills extraction.
- Location, remote/hybrid/on-site, salary, similarity, and high-match filters.
- Required skills are highlighted as matched or missing; score components are shown
  with their weights.
- Favorite jobs, recent search history, and per-job usefulness feedback, stored under
  `data/` for this local demo.
- Prior usefulness feedback gives a smoothed ranking adjustment of at most five points.
- Export recommendation results as a downloadable PDF.
- Optional job refresh from the public Remotive API. Refreshed listings are merged
  into `data/jobs.csv` and the local recommendation model is rebuilt.
- Apply links use the listing URL when available and otherwise open a job search.
- Cover letters use OpenAI with the candidate resume, skills, and job description.
- Training prints Precision@3 and Recall@3 using documented title-keyword relevance
  labels in `Train_model.py`; treat these as demo ranking checks, not benchmark scores.

The job CSV schema includes `title`, `company`, `location`, `experience`, `salary`,
`skills`, `description`, `job_type`, `remote`, `work_mode`, `posted_date`, and
`apply_url`. On first training, the sentence-transformers package downloads the
embedding model; retrain after refreshing job listings to update the embeddings.
