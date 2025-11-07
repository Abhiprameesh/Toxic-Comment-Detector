import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from groq import Groq
from dotenv import load_dotenv


# ------------------------------
# Config & constants
# ------------------------------
LABEL_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TRAIN_PATH = 'train.csv'
load_dotenv()  # Load environment variables from .env if present


# ------------------------------
# NLTK setup (cached)
# ------------------------------
@st.cache_resource(show_spinner=False)
def _ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    return True


# ------------------------------
# Preprocessing
# ------------------------------
_ensure_nltk()
_STOPWORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str, remove_stopwords: bool = True, do_lemmatize: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Basic normalization
    text = text.lower()
    # Remove URLs
    text = re.sub(r'(http\S+|www\S+|https\S+)', ' ', text)
    # Keep only letters, numbers and spaces
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    # Collapse spaces
    text = ' '.join(text.split())
    if remove_stopwords:
        text = ' '.join(w for w in text.split() if w not in _STOPWORDS)
    if do_lemmatize:
        text = ' '.join(_LEMMATIZER.lemmatize(w) for w in text.split())
    return text


def preprocess_series(s: pd.Series, remove_stopwords: bool = True, do_lemmatize: bool = True) -> pd.Series:
    return s.astype(str).apply(lambda x: clean_text(x, remove_stopwords, do_lemmatize))


# ------------------------------
# Load data (cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find training file at {path}")
    df = pd.read_csv(path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    expected = ['comment_text'] + LABEL_NAMES
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}. Expected: {expected}")
    return df


# ------------------------------
# Vectorizer + Model (cached)
# ------------------------------
class ToxicModel:
    def __init__(self, vectorizer: TfidfVectorizer, clf: OneVsRestClassifier):
        self.vectorizer = vectorizer
        self.clf = clf

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        Xv = self.vectorizer.transform(texts)
        # predict_proba is available for logistic regression in OvR
        return self.clf.predict_proba(Xv)


@st.cache_resource(show_spinner=True)
def build_model(
    df: pd.DataFrame,
    sample_size: int | None = 50000,
    remove_stopwords: bool = True,
    do_lemmatize: bool = True,
) -> ToxicModel:
    # Optional sampling to speed up first build
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Preprocess
    texts = preprocess_series(df['comment_text'], remove_stopwords, do_lemmatize)
    y = df[LABEL_NAMES].astype(int)

    # Vectorize
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(texts)

    # Model
    base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    ovr = OneVsRestClassifier(base_lr)
    ovr.fit(X, y)

    return ToxicModel(tfidf, ovr)


# ------------------------------
# Inference utilities
# ------------------------------
def predict_labels(
    model: ToxicModel,
    texts: List[str],
    threshold: float = 0.7,
    per_label_thresholds: Dict[str, float] | None = None,
) -> List[Dict]:
    cleaned = [clean_text(t) for t in texts]
    proba = model.predict_proba(cleaned)
    results = []
    for p in proba:
        probs = {LABEL_NAMES[i]: float(p[i]) for i in range(len(LABEL_NAMES))}
        # Use per-label thresholds if provided, else the global threshold
        if per_label_thresholds:
            preds = {name: int(probs[name] >= per_label_thresholds.get(name, threshold)) for name in LABEL_NAMES}
        else:
            preds = {name: int(probs[name] >= threshold) for name in LABEL_NAMES}
        toxic_any = int(any(preds.values()))
        results.append({
            'prob': probs,
            'pred': preds,
            'toxic_any': toxic_any
        })
    return results


def format_single_label(res: Dict, mode: str = 'binary') -> str:
    if mode == 'binary':
        return 'toxic' if res['toxic_any'] == 1 else 'clean'
    if mode == 'multi':
        active = [k for k, v in res['pred'].items() if v == 1]
        return ', '.join(active) if active else 'clean'
    return 'clean'


# ------------------------------
# Groq rewrite utilities
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client():
    key = os.getenv('GROQ_API_KEY')
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


def rewrite_with_groq(text: str, model: str = 'llama-3.1-8b-instant') -> tuple[str | None, str | None]:
    """Return (rewritten_text, error). Uses Groq Chat Completions API with a simple, direct style.
    No explicit word replacement rules are applied in code; rely on the LLM to rewrite politely.
    """
    client = get_groq_client()
    if client is None:
        return None, 'Groq client not initialized. Provide a valid GROQ_API_KEY.'
    try:
        sys_msg = (
            "You rewrite comments into polite, non-toxic language while preserving meaning. "
            "Be concise and direct. If the input uses direct insults, replace them with a neutral but clear "
            "description (e.g., 'You are a cocksucker' -> 'You are a bad person')."
        )
        user_msg = (
            "Rewrite the following comment to make it polite and non-toxic. "
            "Do not change the meaning too much; just remove insults/offensive words and keep the rest intact.\n\n"
            f"Comment: \n{text}\n\n"
            "Return only the rewritten comment, nothing else."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        content = (resp.choices[0].message.content or "").strip()
        lowered = content.lower()
        # Heuristic: detect refusals/empty and ask caller to fallback
        if not content or any(phrase in lowered for phrase in ["i can't help", "cannot help", "can't assist", "unable to help", "i won't"]):
            return None, 'Groq returned a refusal or empty content.'
        return content, None
    except Exception as e:
        return None, str(e)

# Removed explicit local replacement functions per request; rely on LLM for rewriting.


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="🧪", layout="centered")
st.title("🧪 Toxic Comment Detector")
st.caption("Multi-label toxic comment classifier using TF-IDF + One-vs-Rest Logistic Regression")

with st.sidebar:
    st.header("Settings")
    sample_size = st.selectbox("Training size (for speed)", options=[10000, 20000, 50000, 100000, 'All'], index=2)
    sample_size = None if sample_size == 'All' else int(sample_size)
    threshold = st.slider("Toxic threshold (global)", min_value=0.1, max_value=0.95, value=0.7, step=0.05,
                          help="Increase this if you see false positives for neutral text.")
    use_per_label = st.checkbox("Use per-label thresholds", value=False,
                                help="Tune thresholds per class to reduce false positives.")
    per_label_thresholds = {}
    if use_per_label:
        st.caption("Per-label thresholds")
        for name in LABEL_NAMES:
            per_label_thresholds[name] = st.slider(f"{name}", 0.1, 0.95, 0.7, 0.05, key=f"thr_{name}")
    label_mode = st.radio("Output label format", options=['binary', 'multi'], index=0,
                          help="binary: toxic/clean; multi: list toxic categories")
    remove_sw = st.checkbox("Remove stopwords", value=True)
    do_lemma = st.checkbox("Lemmatize", value=True)
    st.divider()
    st.subheader("Groq Rewrite")
    use_groq = st.checkbox("Rewrite with Groq after prediction", value=False)
    rewrite_only_if_toxic = st.checkbox("Only rewrite if predicted toxic", value=True)
    groq_model = st.selectbox("Groq model", options=[
        'llama-3.1-8b-instant',
        'llama-3.1-70b-versatile',
        'mixtral-8x7b-32768',
    ], index=0)


@st.cache_resource(show_spinner=True)
def _train_cached(sample_size: int | None, remove_sw: bool, do_lemma: bool) -> ToxicModel:
    data = load_data(TRAIN_PATH)
    return build_model(data, sample_size=sample_size, remove_stopwords=remove_sw, do_lemmatize=do_lemma)


with st.spinner("Training/Loading model..."):
    model = _train_cached(sample_size, remove_sw, do_lemma)


st.subheader("Try it out")
default_text = "i love nlp"
user_text = st.text_area("Enter a comment", value=default_text, height=120)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if user_text.strip():
        results = predict_labels(
            model, [user_text], threshold=threshold,
            per_label_thresholds=per_label_thresholds if use_per_label else None
        )
        res = results[0]
        label = format_single_label(res, mode=label_mode)

        with col2:
            st.markdown(f"**Predicted:** `{label}`")
            st.progress(int(res['toxic_any'] * 100))

            with st.expander("Details (per-label probabilities)"):
                prob_series = pd.Series(res['prob']).sort_values(ascending=False)
                st.bar_chart(prob_series, use_container_width=True)

            if use_groq and (not rewrite_only_if_toxic or res['toxic_any'] == 1):
                st.markdown("**Polished (Groq):**")
                rewritten, err = rewrite_with_groq(user_text, model=groq_model)
                if err or not rewritten:
                    st.info("Groq was unavailable or declined. Enable Groq to generate a polite rewrite.")
                else:
                    st.code(rewritten)
            elif not use_groq and (not rewrite_only_if_toxic or res['toxic_any'] == 1):
                st.markdown("**Polished:**")
                st.info("Enable 'Rewrite with Groq' in the sidebar to generate a polite version.")
    else:
        with col2:
            st.warning("Please enter some text.")


st.divider()
st.caption("Model: TF-IDF (1-2 grams) + OneVsRest(LogReg, class_weight='balanced'). Trained on train.csv (Jigsaw Toxic).")

