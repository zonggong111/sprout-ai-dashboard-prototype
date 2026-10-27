# Deployment

## Streamlit Community Cloud settings

- Repository: the GitHub repository containing this folder's contents
- Branch: the branch to deploy
- Main file path: `dashboard.py`
- Python version: 3.13
- Secrets: none

## Deploy

1. Create a GitHub repository and upload the complete contents of this folder,
   preserving the directory structure.
2. In Streamlit Community Cloud, create an app from that repository and branch.
3. Set the entry point to `dashboard.py` and choose Python 3.13 in advanced
   settings.
4. Deploy and verify the three dashboard tabs and the synthetic-data warning.

The local equivalent is:

```bash
streamlit run dashboard.py
```

## Future updates

Push changes to the connected branch. Streamlit Community Cloud detects code and
dependency changes automatically. If demo data changes, update all related CSVs
under `outputs/tables/` together so their metrics remain consistent.

Do not commit secrets, local environments, source documents, screenshots, review
HTML, or non-runtime exports.
