# 🧠 MindPulse — Mental Health Emotion Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)

> **AI-powered, multi-label mental-health emotion detection from free-form text.**

MindPulse is an end-to-end NLP project that:

1. Loads the `DepressionEmo` Reddit dataset (`anger`, `sadness`, `emptiness`, `hopelessness`, `worthlessness`, `loneliness`, `suicide intent`, `brain dysfunction (forget)`).
2. Trains a **TF-IDF (1-2 grams, 10k features) + OneVsRest(RandomForest)** multi-label classifier.
3. Serves predictions through an interactive **Streamlit** dashboard with adjustable threshold, per-emotion probability chart, and downloadable report.

> ⚠️ **Disclaimer:** This project is for educational and awareness purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is in crisis, please contact a qualified mental-health professional or a crisis helpline in your country.

---

## ✨ Features

- **Multi-label classification** — a single text can trigger several emotion labels simultaneously.
- **Two-model design** — a RandomForest primary (best F1) and a LogisticRegression companion used only for per-word explanations.
- **Top contributing words** for every prediction — see *which* words pushed a label up, derived from TF-IDF × coefficient.
- **Color-coded word cloud** of the input text, tinted with the top predicted emotion.
- **Animated, confidence-coloured emotion cards** (green ≥0.7, amber ≥0.4, red ≥0.2, gray below) with a "high-risk" tag on `suicide intent` / `hopelessness`.
- **🌙 Dark-mode toggle** in the top bar (CSS variable swap, no full page reload).
- **Custom Streamlit theme** in `.streamlit/config.toml` (deep purple brand palette).
- **Loading skeleton** with shimmer animation while the model is running.
- **6 curated "Try this" examples** in the gallery for instant demos.
- **Adjustable prediction threshold** in the sidebar — sweep from 0.05 to 0.95.
- **Per-class probability bar chart** coloured by confidence with the threshold line and per-bar value labels.
- **High-risk warning banner** that escalates to `st.error` if `suicide intent` or `hopelessness` is detected.
- **Downloadable text report** of every prediction including the contributing words.
- **Live model metrics** (F1 micro/macro, precision, recall) for both models, shown in the sidebar.
- **Reproducible training script** (`train_model.py`) — re-train from scratch in one command.

---

## 🗂 Project structure

```
MindPulse/
├── app/
│   ├── app.py                 # Streamlit front-end
│   └── models/                # Trained artefacts (created by train_model.py)
│       ├── mental_health_model.pkl    # primary (RandomForest)
│       ├── interpretable_model.pkl    # companion (LogisticRegression)
│       └── mlb.pkl
├── .streamlit/
│   └── config.toml            # Custom theme (deep purple brand palette)
├── DepressionEmo/             # Raw dataset + research scripts
│   └── Dataset/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── Notebook/
│   ├── EDA.ipynb
│   └── Multi-label Classifier - Threshold Tuned.ipynb
├── models/
│   └── metrics.json           # Test-set metrics from train_model.py
├── train_model.py             # Reproducible training script (trains both models)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Clone & enter

```bash
git clone https://github.com/iamHimanshu-07/MindPulse.git
cd MindPulse
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (one-time, ~1 min on CPU)

```bash
python train_model.py
```

This produces `app/models/mental_health_model.pkl`, `app/models/mlb.pkl`, and `models/metrics.json`.

> The model artefacts are intentionally **not** committed to the repo — they are large, binary, and easy to regenerate. This keeps the repo light and reproducible. **For deployment, see the *Deploy to Streamlit Community Cloud* section below** — the app downloads the model on first run.

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

Then open <http://localhost:8501> in your browser.

---

## 📊 Model performance

Test set (20 % hold-out, `random_state=42`):

| Model | F1 (micro) | F1 (macro) | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| **Primary** — `OneVsRest(RandomForest, n=200)` | 0.7585 | 0.7069 | 0.6945 | 0.8354 |
| **Interpretable** — `OneVsRest(LogisticRegression, liblinear, C=1.0)` | 0.7635 | 0.6890 | 0.7128 | 0.8218 |

Both pipelines share the same `TfidfVectorizer(ngram_range=(1,2), max_features=10_000, stop_words='english')` and are trained on a concatenation of `train.json + val.json + test.json` (split 80/20).

The RandomForest is used for predictions; the LogisticRegression is used **only** to surface the words that pushed each label up — its coefficients × TF-IDF scores give a faithful "why this prediction" view that RF does not provide directly.

*Baseline only — see the **Future work** section for planned upgrades.*

---

## 🧪 Using the app

Type or paste any text into the **Your text** box.
 <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/cabdbf1e-0913-40f0-96a6-3fccb59e6110" />
 
---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/awesome`.
3. Commit your changes: `git commit -m "feat: ..."`.
4. Push: `git push origin feat/awesome`.
5. Open a Pull Request.

---

## 📄 License

[Apache-2.0](LICENSE)

---
