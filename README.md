# AI-Powered Financial Fraud Detection System

## Project Description
This project uses Machine Learning to detect fraudulent financial
transactions, with a generative-AI layer that explains each result in
plain language. Built with Python, Scikit-learn, SHAP, OpenAI, and
Streamlit.

## Technologies Used
- Python
- Pandas / NumPy
- Scikit-learn (Random Forest)
- SHAP (model explainability)
- OpenAI API (gpt-4o-mini) for generative explanations
- Streamlit (web app UI)
- Joblib

## How It Works
1. A Random Forest model predicts **Fraudulent** or **Genuine** for a
   transaction.
2. **SHAP** computes which anonymized features (V1–V28) pushed that
   specific transaction toward fraud or genuine, and by how much.
3. Those contributions are passed to an **LLM** (`gpt-4o-mini`), which
   writes a short, plain-language explanation grounded in the SHAP values
   — it never overrides or contradicts the model's prediction.
4. If the OpenAI API key isn't configured or the service is unreachable,
   the app automatically falls back to a template-based explanation built
   from the same SHAP values, so a live demo never breaks.
5. A follow-up Q&A box lets a user ask more about the result.

## Files
- `train.py` — Trains the Random Forest model
- `app.py` — Streamlit web application (prediction + GenAI explanation + Q&A)
- `fraud_model.pkl` — Trained machine learning model
- `creditcard.csv` — Dataset (also used to load sample transactions in the UI)
- `requirements.txt` — Python dependencies
- `secrets.toml.example` — Template for your OpenAI API key

## How to Run

Install required libraries:

```
pip install -r requirements.txt
```

(Optional, to enable live AI explanations instead of offline fallback)
Copy `secrets.toml.example` to `.streamlit/secrets.toml` and add your
OpenAI API key.

Run the training script:

```
python train.py
```

Run the web application:

```
python -m streamlit run app.py
```
