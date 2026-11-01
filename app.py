
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from google import genai
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

st.title("Diabetes & Cardiometabolic Risk Assessment")
st.caption("Explainable AI + Retrieval-Augmented Generative AI Research Prototype")

# Load model
model = joblib.load("diabetes_model.pkl")

# Gemini connection
@st.cache_resource
def get_gemini_client():
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

# RAG knowledge base
documents = [
    {
        "topic": "Diabetes Risk Factors",
        "text": "Risk factors include overweight, physical inactivity, family history and increasing age.",
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/risk-factors/"
    },
    {
        "topic": "Diabetes Prevention",
        "text": "Healthy eating, regular physical activity and maintaining a healthy weight can help reduce type 2 diabetes risk.",
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/prevention-type-2/"
    },
    {
        "topic": "Cardiovascular Health",
        "text": "High blood pressure, unhealthy cholesterol, smoking and physical inactivity are cardiovascular risk factors.",
        "source": "WHO",
        "url": "https://www.who.int/health-topics/cardiovascular-diseases"
    }
]

texts = [d["topic"] + " " + d["text"] for d in documents]
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(texts)

def retrieve_information(question, top_k=2):
    q_vector = vectorizer.transform([question])
    scores = cosine_similarity(q_vector, doc_vectors)[0]
    indices = scores.argsort()[::-1][:top_k]
    return [documents[i] for i in indices]

def rag_answer(question):
    retrieved = retrieve_information(question)

    context = "\n\n".join([
        f"{d['topic']}: {d['text']} Source: {d['source']} ({d['url']})"
        for d in retrieved
    ])

    prompt = f"""
    You are a health education assistant.
    Answer using the retrieved information below.
    Do not diagnose or invent medical facts.
    Mention the sources and advise professional medical consultation.

    Retrieved information:
    {context}

    User question:
    {question}
    """

    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text, retrieved

# Prediction section
st.header("1. Diabetes Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    submitted = st.form_submit_button("Predict Diabetes Risk")

if submitted:
    sample = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Model predicts a higher likelihood of diabetes.")
    else:
        st.success("Model predicts a lower likelihood of diabetes.")

    st.metric("Model estimated probability", f"{probability*100:.2f}%")

    st.subheader("SHAP Explainability")

    transformed = model.named_steps["imputer"].transform(sample)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": sample.columns,
        "SHAP Value": values
    })

    explanation = explanation.sort_values(
        "SHAP Value", key=abs, ascending=False
    )

    st.bar_chart(explanation.set_index("Feature")["SHAP Value"])

# RAG section
st.divider()
st.header("2. RAG + Gemini Health Education Assistant")

question = st.text_area(
    "Ask a question about diabetes or cardiovascular health:"
)

if st.button("Ask Gemini AI"):
    if question.strip():
        try:
            with st.spinner("Retrieving information and generating answer..."):
                answer, sources = rag_answer(question)

            st.subheader("AI Generated Answer")
            st.write(answer)

            st.subheader("Retrieved Sources")
            for doc in sources:
                st.write(f"**{doc['topic']} — {doc['source']}**")
                st.write(doc["url"])

        except Exception as e:
            st.error(f"Could not generate answer: {e}")
    else:
        st.warning("Please enter a question.")

st.divider()
st.caption(
    "Research prototype for educational purposes only. "
    "Not a medical diagnosis or clinical risk assessment tool."
)
