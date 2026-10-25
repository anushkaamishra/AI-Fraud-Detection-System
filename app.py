import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================
FEATURE_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------
# Light custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.5rem; }
        .result-card {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            font-size: 1.05rem;
            color: #ffffff !important;
        }
        .result-card b { color: #ffffff !important; }
        .fraud-card { background-color: #c0392b; border: 1px solid #7a2e2e; }
        .genuine-card { background-color: #2e8b4f; border: 1px solid #1f6b38; }
        .genai-card {
            background-color: #1b2a3d;
            border: 1px solid #2e4a7a;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            color: #f0f4f8 !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CACHED RESOURCES
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_reference_dataset():
    try:
        return pd.read_csv("creditcard.csv")
    except Exception:
        return None


model = load_model()
explainer = load_explainer(model)
reference_df = load_reference_dataset()


# =========================================================
# HELPERS
# =========================================================
def openai_key_available() -> bool:
    try:
        return bool(st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        return False


def get_shap_contributions(df_row: pd.DataFrame, top_n: int = 5):
    """Return top_n (feature, value, shap_contribution) tuples for the fraud class."""
    raw = explainer.shap_values(df_row)

    if isinstance(raw, list):
        # older shap versions: list[class] -> (n_samples, n_features)
        shap_row = np.array(raw[1])[0]
    else:
        arr = np.array(raw)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            shap_row = arr[0, :, 1]
        else:
            shap_row = arr[0]

    order = np.argsort(-np.abs(shap_row))[:top_n]
    contributions = [
        (FEATURE_COLUMNS[i], float(df_row.iloc[0, i]), float(shap_row[i]))
        for i in order
    ]
    return contributions


def build_context_prompt(prediction_text, fraud_probability, amount, contributions):
    lines = "\n".join(
        f"- {feat} (value {val:.3f}): SHAP contribution {shap_val:+.3f} "
        f"({'toward fraud' if shap_val > 0 else 'toward genuine'})"
        for feat, val, shap_val in contributions
    )
    return f"""
You are explaining the result of a financial fraud-detection ML model to a
non-technical user. The Random Forest model has ALREADY made the prediction.
Do not make a new prediction or contradict it.

Transaction summary:
- Model prediction: {prediction_text}
- Fraud probability: {fraud_probability}%
- Transaction amount: {amount}

Top contributing anonymized features (from SHAP, a model-explainability method):
{lines}

Instructions:
1. Explain in 3-5 short sentences, plain language, for a normal user.
2. Reference the SHAP contributions in general terms (e.g. "a few unusual
   transaction-pattern features pushed the score toward fraud") — never invent
   a real-world meaning for V1-V28, they are anonymized.
3. Do not claim certainty beyond what the probability implies.
4. If fraudulent, recommend verifying the transaction with the cardholder/bank.
5. If genuine, note the model did not find the pattern suspicious, without
   guaranteeing it's risk-free.
""".strip()


def get_ai_explanation(prediction_text, fraud_probability, amount, contributions):
    """Returns (explanation_text, mode) where mode is 'ai' or 'fallback'."""
    if not openai_key_available():
        return fallback_explanation(prediction_text, fraud_probability, contributions), "fallback"

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = build_context_prompt(prediction_text, fraud_probability, amount, contributions)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain fraud-detection ML results clearly and cautiously. "
                        "You never override the model's prediction and never invent "
                        "real-world meaning for anonymized features."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=220,
        )
        return response.choices[0].message.content, "ai"
    except Exception as e:
        st.session_state["last_ai_error"] = str(e)
        return fallback_explanation(prediction_text, fraud_probability, contributions), "fallback"


def fallback_explanation(prediction_text, fraud_probability, contributions):
    top_feats = ", ".join(f for f, _, _ in contributions[:3])
    if prediction_text == "Fraudulent":
        action = "We recommend verifying this transaction with the cardholder or bank before proceeding."
    else:
        action = "The model did not find this transaction's pattern suspicious, though no model is ever 100% certain."
    return (
        f"**Offline explanation** (AI explanation service unavailable).\n\n"
        f"The model flagged this transaction as **{prediction_text}** with an estimated "
        f"fraud probability of **{fraud_probability}%**. The anonymized features that "
        f"influenced this result the most were **{top_feats}**. {action}"
    )


def ask_followup(question, context_summary, history):
    if not openai_key_available():
        return (
            "The AI Q&A service needs an OpenAI API key configured in "
            "`st.secrets['OPENAI_API_KEY']` to answer follow-up questions."
        )
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer follow-up questions about a fraud-detection model's result. "
                    "Stay grounded in the transaction summary provided. Never invent real-world "
                    "meaning for anonymized V1-V28 features. Keep answers short."
                ),
            },
            {"role": "user", "content": f"Transaction context:\n{context_summary}"},
        ]
        for turn in history:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Couldn't reach the AI service ({e})."


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("💳 About this project")
    st.write(
        "AI-Powered Financial Fraud Detection System — a Random Forest "
        "classifier trained on anonymized credit-card transactions, paired "
        "with a generative-AI layer that explains each result in plain "
        "language, grounded in SHAP feature-contribution values."
    )
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Random Forest predicts Fraud / Genuine\n"
        "2. SHAP explains *why*, feature by feature\n"
        "3. An LLM turns that into a plain-language summary\n"
        "4. Fallback template if the AI service is unavailable"
    )
    st.caption(
        "Note: without an OpenAI key in `secrets.toml`, explanations run in "
        "offline fallback mode automatically."
    )


# =========================================================
# MAIN
# =========================================================
st.title("💳 AI-Powered Financial Fraud Detection System")

tab_predict, tab_about = st.tabs(["🔎 Check a Transaction", "ℹ️ How It Works"])

with tab_predict:
    st.write("Enter transaction details, or load a real sample from the dataset:")

    col_a, col_b = st.columns(2)
    load_genuine = col_a.button("🎲 Load random genuine sample", use_container_width=True)
    load_fraud = col_b.button("🎲 Load random fraud sample", use_container_width=True)

    if load_genuine or load_fraud:
        if reference_df is None:
            st.warning("creditcard.csv not found next to app.py — sample loading is disabled.")
        else:
            target_class = 0 if load_genuine else 1
            subset = reference_df[reference_df["Class"] == target_class]
            if len(subset) == 0:
                st.warning("No matching sample found in the dataset.")
            else:
                row = subset.sample(1).iloc[0]
                for col in FEATURE_COLUMNS:
                    st.session_state[f"input_{col}"] = float(row[col])
                st.success("Sample loaded into the fields below ⬇️")

    col_time, col_amount = st.columns(2)
    time_val = col_time.number_input(
        "Time", key="input_Time", value=st.session_state.get("input_Time", 0.0)
    )
    amount_val = col_amount.number_input(
        "Amount", key="input_Amount", value=st.session_state.get("input_Amount", 0.0)
    )

    features = []
    with st.expander("Advanced: anonymized features V1–V28", expanded=False):
        grid_cols = st.columns(4)
        for i in range(1, 29):
            col_name = f"V{i}"
            key = f"input_{col_name}"
            with grid_cols[(i - 1) % 4]:
                val = st.number_input(
                    col_name, key=key, value=st.session_state.get(key, 0.0)
                )
            features.append(val)

    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        data = [[time_val] + features + [amount_val]]
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]
        fraud_probability = round(float(probability) * 100, 2)
        prediction_text = "Fraudulent" if prediction[0] == 1 else "Genuine"

        st.session_state["last_result"] = {
            "prediction_text": prediction_text,
            "fraud_probability": fraud_probability,
            "amount": amount_val,
        }
        st.session_state["chat_history"] = []

        # ---- Result card ----
        card_class = "fraud-card" if prediction[0] == 1 else "genuine-card"
        icon = "🚨" if prediction[0] == 1 else "✅"
        st.markdown(
            f'<div class="result-card {card_class}"><b>{icon} '
            f'{"Fraudulent Transaction Detected" if prediction[0] == 1 else "Genuine Transaction"}'
            f"</b></div>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{fraud_probability}%")
        m2.metric("Transaction amount", f"{amount_val:,.2f}")
        st.progress(min(int(fraud_probability), 100))

        # ---- SHAP contributions ----
        contributions = get_shap_contributions(df, top_n=6)
        st.session_state["last_contributions"] = contributions

        st.subheader("📊 What influenced this result")
        feat_names = [f for f, _, _ in contributions][::-1]
        feat_vals = [s for _, _, s in contributions][::-1]
        colors = ["#d9534f" if v > 0 else "#5cb85c" for v in feat_vals]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(feat_names, feat_vals, color=colors)
        ax.set_xlabel("SHAP contribution (→ fraud is positive)")
        ax.axvline(0, color="#444444", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)

        # ---- GenAI explanation ----
        st.subheader("🤖 Generative AI Explanation")
        explanation, mode = get_ai_explanation(
            prediction_text, fraud_probability, amount_val, contributions
        )
        st.markdown(f'<div class="genai-card">{explanation}</div>', unsafe_allow_html=True)
        if mode == "fallback" and openai_key_available():
            st.caption(f"AI service error: {st.session_state.get('last_ai_error', 'unknown')}")

        st.session_state["last_explanation"] = explanation

        # ---- Downloadable report ----
        report = (
            f"AI-Powered Financial Fraud Detection — Report\n"
            f"Prediction: {prediction_text}\n"
            f"Fraud probability: {fraud_probability}%\n"
            f"Transaction amount: {amount_val}\n\n"
            f"Top contributing features:\n"
            + "\n".join(f"- {f}: value={v:.3f}, shap={s:+.3f}" for f, v, s in contributions)
            + f"\n\nExplanation:\n{explanation}\n"
        )
        st.download_button(
            "⬇️ Download report (.txt)", data=report, file_name="fraud_report.txt"
        )

    # ---- Follow-up Q&A ----
    if "last_result" in st.session_state:
        st.divider()
        st.subheader("💬 Ask a follow-up question")
        question = st.text_input("e.g. 'Why is the amount not the main factor?'")
        if st.button("Ask") and question:
            r = st.session_state["last_result"]
            context_summary = (
                f"Prediction: {r['prediction_text']}, "
                f"Fraud probability: {r['fraud_probability']}%, "
                f"Amount: {r['amount']}. "
                f"Top features: {st.session_state.get('last_contributions')}"
            )
            answer = ask_followup(
                question, context_summary, st.session_state.get("chat_history", [])
            )
            st.session_state.setdefault("chat_history", []).append(
                {"q": question, "a": answer}
            )

        for turn in reversed(st.session_state.get("chat_history", [])):
            st.markdown(f"**You:** {turn['q']}")
            st.markdown(f"**AI:** {turn['a']}")

with tab_about:
    st.markdown(
        """
### Model
A **Random Forest classifier** trained on an anonymized credit-card
transaction dataset (`Time`, `V1`–`V28` PCA-transformed features, `Amount`)
predicts whether a transaction is fraudulent.

### Generative AI layer
1. **SHAP** (SHapley Additive exPlanations) computes how much each feature
   pushed the prediction toward "fraud" or "genuine" for *this specific*
   transaction.
2. Those contributions — not the raw prediction alone — are passed to an
   **LLM (gpt-4o-mini)**, which turns them into a short, plain-language
   explanation for a non-technical user.
3. If the AI service is unreachable (no API key, network issue, quota),
   the app automatically falls back to a template-based explanation built
   from the same SHAP values, so the demo never breaks.
4. A follow-up **Q&A box** lets a user ask more about the result, with the
   LLM answering grounded in the transaction's own summary.

This keeps the generative-AI component **explanatory and grounded**, rather
than a model that could contradict or replace the ML prediction.
        """
    )