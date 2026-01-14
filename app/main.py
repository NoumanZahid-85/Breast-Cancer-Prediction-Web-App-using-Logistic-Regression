import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../Data/data.csv"
MODEL_PATH = BASE_DIR / "../Model/logistic_model.pkl"
SCALER_PATH = BASE_DIR / "../Model/scaler.pkl"
CSS_PATH = BASE_DIR / "../assets/style.css"
# ===============================
# Load & Cache Data
# ===============================
@st.cache_data
def get_clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["id", "Unnamed: 32"])
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data

@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, scaler

# ===============================
# Sidebar Inputs
# ===============================
def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)", "fractal_dimension_mean"),

        ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"), ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"), ("Fractal dimension (se)", "fractal_dimension_se"),

        ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"), ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"), ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"), ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    with st.sidebar.expander("Mean Values", expanded=True):
        for label, key in slider_labels[:10]:
            input_dict[key] = st.slider(
                label,
                float(data[key].min()),
                float(data[key].max()),
                float(data[key].mean())
            )

    with st.sidebar.expander("Standard Error Values"):
        for label, key in slider_labels[10:20]:
            input_dict[key] = st.slider(
                label,
                float(data[key].min()),
                float(data[key].max()),
                float(data[key].mean())
            )

    with st.sidebar.expander("Worst Values"):
        for label, key in slider_labels[20:]:
            input_dict[key] = st.slider(
                label,
                float(data[key].min()),
                float(data[key].max()),
                float(data[key].mean())
            )

    return input_dict

# ===============================
# Radar Chart
# ===============================
def get_radar_chart(input_data):
    data = get_clean_data().drop(columns=["diagnosis"])
    scaled = {
        k: (v - data[k].min()) / (data[k].max() - data[k].min())
        for k, v in input_data.items()
    }

    categories = [
        "Radius", "Texture", "Perimeter", "Area", "Smoothness",
        "Compactness", "Concavity", "Concave Points",
        "Symmetry", "Fractal Dimension"
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[scaled[f"{k}_mean"] for k in
           ["radius","texture","perimeter","area","smoothness","compactness",
            "concavity","concave points","symmetry","fractal_dimension"]],
        theta=categories,
        fill="toself",
        name="Mean"
    ))

    fig.add_trace(go.Scatterpolar(
        r=[scaled[f"{k}_se"] for k in
           ["radius","texture","perimeter","area","smoothness","compactness",
            "concavity","concave points","symmetry","fractal_dimension"]],
        theta=categories,
        fill="toself",
        name="Standard Error"
    ))

    fig.add_trace(go.Scatterpolar(
        r=[scaled[f"{k}_worst"] for k in
           ["radius","texture","perimeter","area","smoothness","compactness",
            "concavity","concave points","symmetry","fractal_dimension"]],
        theta=categories,
        fill="toself",
        name="Worst"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        showlegend=True
    )

    return fig

# ===============================
# Prediction Output
# ===============================
def add_predictions(input_data):
    model, scaler = load_model()
    input_array = scaler.transform(np.array(list(input_data.values())).reshape(1, -1))
    prediction = model.predict(input_array)[0]
    probs = model.predict_proba(input_array)[0]

    st.markdown("### Cell Cluster Prediction")

    if prediction == 0:
        label, prob, cls = "Benign", probs[0], "benign"
    else:
        label, prob, cls = "Malignant", probs[1], "malignant"

    st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">Diagnosis</div>
            <div class="diagnosis-badge {cls}">{label}</div>
            <div class="probability-text">Probability: <b>{prob:.2%}</b></div>
        </div>
        <div class="disclaimer">
            This app assists medical professionals but must not replace clinical diagnosis.
        </div>
    """, unsafe_allow_html=True)

# ===============================
# Main App
# ===============================
def main():
    st.set_page_config("Breast Cancer Prediction", "🎗️", layout="wide")

    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_data = add_sidebar()

    st.title("Breast Cancer Prediction App 🎗️")
    st.write("Enter cell nuclei measurements using the sidebar.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(get_radar_chart(input_data), use_container_width=True)
    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()
