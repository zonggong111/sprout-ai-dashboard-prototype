# Sprout AI synthetic dashboard

This repository contains a deployment-only Streamlit prototype using synthetic
demonstration data. It contains no production data and requires no credentials,
API keys, database, or external service.

The visible dashboard warning must remain in place:

> SYNTHETIC / PROTOTYPE DATA — DEMONSTRATION ONLY

## Run locally

Python 3.13 is the validated runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Runtime structure

- `dashboard.py` is the Streamlit entry point.
- `src/feedback_analysis.py` provides the dashboard's feedback transformations.
- `outputs/tables/` contains only the synthetic CSV tables read by the app.
- `requirements.txt` contains only deployed Python dependencies.
- `DEPLOYMENT.md` contains Streamlit Community Cloud instructions.

The app resolves data paths from the directory containing `dashboard.py`, so it
does not depend on the current working directory or files outside this folder.
