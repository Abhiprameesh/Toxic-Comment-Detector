# Toxic Comment Detector

Multi-label toxic comment classifier with an optional text rewrite step. The app is built with Streamlit and trains a TF‑IDF + One‑vs‑Rest Logistic Regression model on the Jigsaw Toxic Comment dataset (`train.csv`). Optionally, it can rewrite toxic text using the Groq API.

## Features
- **Multi‑label classification**: toxic, severe_toxic, obscene, threat, insult, identity_hate
- **Configurable thresholds**: global or per‑label
- **Text preprocessing**: lowercasing, URL removal, token filtering, stopword removal, lemmatization
- **On‑the‑fly training**: sample size selectable for speed
- **Interactive UI**: bar chart of per‑label probabilities
- **Optional rewrite**: Groq LLM rewrites text into non‑toxic language

## Project Structure
- `toxic_detector_app.py` — Streamlit UI + training/inference + optional Groq rewrite
- `toxic_detector.ipynb` — exploratory/analysis notebook (optional)
- `train.csv` — Jigsaw Toxic Comment dataset (expected columns: `comment_text` + 6 label columns)
- `requirements.txt` — Python dependencies
- `.env` — environment variables (optional; for `GROQ_API_KEY`)

## Requirements
- Python 3.10+
- OS: Tested on Windows (should work cross‑platform)
- A copy of `train.csv` in the project root (Jigsaw Toxic Comment dataset)

## Setup
1. (Recommended) Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `train.csv` in the repository root beside `toxic_detector_app.py`.
4. (Optional) To enable rewrite via Groq, set your API key:
   - Create `.env` in the repo root with:
     ```
     GROQ_API_KEY="<your-key>"
     ```
   - or export it in your shell environment.

## Run
Start the Streamlit app:
```bash
streamlit run toxic_detector_app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

## Usage Tips
- Use the sidebar to adjust:
  - **Training size** for faster startup
  - **Thresholds** (global or per‑label) to balance false positives/negatives
  - **Preprocessing**: stopword removal and lemmatization
  - **Groq Rewrite** settings (model, only‑if‑toxic)
- Click "Predict" to get:
  - A binary label (toxic/clean) or list of active toxic categories
  - Per‑label probability bar chart
  - Optional rewritten, polite version (if Groq enabled)

## Data Expectations
- `train.csv` must include columns:
  - `comment_text`
  - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (0/1)
- If an `id` column exists, it will be dropped during loading.

## Notes on Performance
- First run may download NLTK resources (`stopwords`, `wordnet`).
- Training caches across runs via Streamlit caching. Increasing sample size improves quality but increases time/memory.
- The default Logistic Regression (class_weight='balanced') helps with label imbalance.

## Troubleshooting
- Missing `train.csv`:
  - Place the file in the repo root; ensure required columns are present.
- NLTK lookup/download errors:
  - Ensure internet access for first run; Streamlit will download needed corpora.
- Out‑of‑memory or slow training:
  - Reduce Training size in the sidebar.
- Groq rewrite not working:
  - Ensure `GROQ_API_KEY` is set and valid; toggle "Rewrite with Groq" in the sidebar.

## Acknowledgements
- Jigsaw/Wikipedia Toxic Comment Classification Challenge dataset
- scikit‑learn, NLTK, Streamlit

## License
Specify your license here (e.g., MIT).
