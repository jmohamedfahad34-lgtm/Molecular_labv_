# MoleculeLab — Chemoinformatics Virtual Laboratory

An educational Streamlit app for exploring drug-molecule structures and computed molecular descriptors.

## Features
- 29 curated example molecule records
- Search and category filtering
- SMILES parsing and 2D molecular depictions with RDKit
- Molecular descriptors: formula, molecular weight, exact mass, cLogP, TPSA, H-bond donors/acceptors, rotatable bonds, heavy atoms, aromatic rings and fraction sp3
- Custom SMILES validation
- Experimental 3D conformer generation and XYZ download
- CSV property reports
- Structure–property scatter plots and pairwise comparisons
- Short theory modules, quiz and project-report builder

## Run locally
Use Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

If RDKit installation fails in your environment, follow the official RDKit installation guide. A conda environment is also a supported option.

## Deploy
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open Streamlit Community Cloud and sign in.
3. Choose **Create app**, select your repository, branch, and `app.py`.
4. Deploy and test the live app.

## Important notes
- The molecule list is a curated educational starter dataset. Verify all structures/SMILES against authoritative references before formal academic submission.
- Descriptor values are computed estimates and do not establish drug safety, efficacy, dosing, or suitability for any person.
- The app is not a prescribing or clinical decision-support tool.
- Student details are used in the current session to personalize the report; this starter app does not save them to a database.
