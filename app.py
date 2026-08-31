import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open("schema.json") as f:
        schema = json.load(f)
    return model, scaler, schema

model, scaler, schema = load_artifacts()

FEATURE_COLUMNS = schema["feature_columns"]
NUMERIC_COLUMNS = schema["numeric_columns"]
NUMERIC_RANGES = schema["numeric_ranges"]
CAT_COLUMNS = schema["categorical_dropdown_columns"]
CAT_CHOICES = schema["categorical_choices"]
BINARY_COLUMNS = schema["binary_columns"]
TARGET_LABELS = schema["target_labels_by_index"]
ONE_HOT_SOURCE_COLUMNS = schema["one_hot_source_columns"]

BINARY_LABELS = {
    "Gender": ("Male", "Female"),
    "Daytime/evening attendance": ("Evening", "Daytime"),
    "Displaced": ("No", "Yes"),
    "Educational special needs": ("No", "Yes"),
    "Debtor": ("No", "Yes"),
    "Tuition fees up to date": ("No", "Yes"),
    "International": ("No", "Yes"),
    "Scholarship holder": ("No", "Yes"),
}

st.title("🎓 Student Outcome Predictor")
st.caption(
    f"Random Forest classifier — Dropout / Enrolled / Graduate "
    f"(validation accuracy: {schema['validation_accuracy']:.1%}, "
    f"macro-F1: {schema['validation_macro_f1']:.2f})"
)
st.write("Fill in a student's profile to predict their likely academic outcome.")

with st.form("student_form"):
    st.subheader("Background")
    col1, col2 = st.columns(2)
    inputs = {}
    with col1:
        for c in CAT_COLUMNS[: len(CAT_COLUMNS) // 2]:
            inputs[c] = st.selectbox(c, CAT_CHOICES[c])
    with col2:
        for c in CAT_COLUMNS[len(CAT_COLUMNS) // 2 :]:
            inputs[c] = st.selectbox(c, CAT_CHOICES[c])

    st.subheader("Personal & administrative")
    col3, col4 = st.columns(2)
    with col3:
        for c in BINARY_COLUMNS[: len(BINARY_COLUMNS) // 2]:
            no_label, yes_label = BINARY_LABELS[c]
            choice = st.radio(c, [no_label, yes_label], horizontal=True)
            inputs[c] = 1 if choice == yes_label else 0
    with col4:
        for c in BINARY_COLUMNS[len(BINARY_COLUMNS) // 2 :]:
            no_label, yes_label = BINARY_LABELS[c]
            choice = st.radio(c, [no_label, yes_label], horizontal=True)
            inputs[c] = 1 if choice == yes_label else 0

    st.subheader("Academic & economic indicators")
    col5, col6 = st.columns(2)
    numeric_cols_list = NUMERIC_COLUMNS
    half = len(numeric_cols_list) // 2
    with col5:
        for c in numeric_cols_list[:half]:
            lo, hi, med = NUMERIC_RANGES[c]
            is_int = float(lo).is_integer() and float(hi).is_integer()
            if is_int:
                inputs[c] = st.number_input(c, min_value=int(lo), max_value=int(hi) * 2 if hi > 0 else 100, value=int(med), step=1)
            else:
                inputs[c] = st.number_input(c, min_value=float(lo), value=float(med), step=0.1, format="%.2f")
    with col6:
        for c in numeric_cols_list[half:]:
            lo, hi, med = NUMERIC_RANGES[c]
            is_int = float(lo).is_integer() and float(hi).is_integer()
            if is_int:
                inputs[c] = st.number_input(c, min_value=int(lo), max_value=int(hi) * 2 if hi > 0 else 100, value=int(med), step=1)
            else:
                inputs[c] = st.number_input(c, min_value=float(lo), value=float(med), step=0.1, format="%.2f")

    submitted = st.form_submit_button("Predict outcome", use_container_width=True)

if submitted:
    row = {c: inputs[c] for c in NUMERIC_COLUMNS}
    row.update({c: inputs[c] for c in BINARY_COLUMNS})
    for c in ONE_HOT_SOURCE_COLUMNS:
        row[c] = inputs[c]

    raw_df = pd.DataFrame([row])
    encoded = pd.get_dummies(raw_df, columns=ONE_HOT_SOURCE_COLUMNS, drop_first=True)

    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[FEATURE_COLUMNS]

    scaled = scaler.transform(encoded)
    pred_idx = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]

    pred_label = TARGET_LABELS[str(pred_idx)] if str(pred_idx) in TARGET_LABELS else TARGET_LABELS[pred_idx]

    st.divider()
    st.subheader("Prediction")
    color = {"Graduate": "green", "Enrolled": "blue", "Dropout": "red"}.get(pred_label, "gray")
    st.markdown(f"### Predicted outcome: :{color}[{pred_label}]")

    proba_df = pd.DataFrame({
        "Outcome": [TARGET_LABELS[str(i)] if str(i) in TARGET_LABELS else TARGET_LABELS[i] for i in range(len(proba))],
        "Probability": proba,
    }).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Outcome"))
    st.dataframe(proba_df.style.format({"Probability": "{:.1%}"}), hide_index=True, use_container_width=True)
