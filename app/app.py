from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_assets():
    model = joblib.load(MODELS / "disease_model.joblib")
    symptoms = joblib.load(MODELS / "symptom_list.joblib")
    scores = pd.read_csv(MODELS / "medicine_scores.csv")
    return model, symptoms, scores

model, symptom_list, scores = load_assets()
labels = {s: s.replace("_", " ").strip().title() for s in symptom_list}

st.title("Personalized Healthcare & Medicine Recommendation System")
st.caption("Select your symptoms to see likely conditions and the medicines patients rated best.")

chosen = st.multiselect("Select your symptoms", options=symptom_list,
                        format_func=lambda s: labels[s])

if st.button("Get recommendations"):
    if not chosen:
        st.warning("Please select at least one symptom.")
    else:
        if len(chosen) < 3:
            st.info("Only a few symptoms selected, so results are less certain. "
                    "Adding more symptoms improves accuracy.")

        x = pd.DataFrame([[1 if s in chosen else 0 for s in symptom_list]],
                         columns=symptom_list)
        probs = model.predict_proba(x)[0]

        for i in probs.argsort()[::-1][:3]:
            name = model.classes_[i].strip()
            st.subheader(f"{name} - {probs[i]*100:.0f}% likelihood")

            meds = scores[scores["disease"] == name].head(5)
            if meds.empty:
                st.write("No review data available for this disease.")
            else:
                st.dataframe(
                    meds[["drugName", "reviews", "avg_rating", "score"]].rename(columns={
                        "drugName": "Medicine", "reviews": "Reviews",
                        "avg_rating": "Avg rating", "score": "Score"}),
                    hide_index=True)

st.divider()
st.caption("Educational project, not medical advice. Medicine rankings come from patient "
           "reviews, not clinical evidence. Consult a qualified doctor for health concerns.")