import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Species Classifier — Streamlit App")
st.sidebar.header("Carga del dataset")
uploaded = st.sidebar.file_uploader("Sube el archivo Iris.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Iris.csv")


# Eliminar columna Id si existe
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Vista previa
st.subheader("Vista previa del dataset")
st.dataframe(df, use_container_width=True)


# ---------------------------------------------------
# PREDICCIÓN MANUAL + VISUALIZACIÓN 3D
# ---------------------------------------------------
st.subheader("Predicción manual + Visualización 3D")

col1, col2 = st.columns(2)
with col1:
    s1 = st.number_input("SepalLengthCm", min_value=0.0, max_value=10.0, value=5.6)
    s2 = st.number_input("SepalWidthCm", min_value=0.0, max_value=10.0, value=2.8)
with col2:
    s3 = st.number_input("PetalLengthCm", min_value=0.0, max_value=10.0, value=4.0)
    s4 = st.number_input("PetalWidthCm", min_value=0.0, max_value=10.0, value=1.3)

if st.button("Predecir especie"):
    sample = [[s1, s2, s3, s4]]
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]

    st.success(f" La especie predicha es: **{pred}**")

    # Gráfico 3D con punto predicho
    st.subheader("Posición de la nueva muestra en el espacio 3D")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for species in df["Species"].unique():
        subset = df[df["Species"] == species]
        ax.scatter(
            subset["SepalLengthCm"],
            subset["SepalWidthCm"],
            subset["PetalLengthCm"],
            label=species
        )

    ax.scatter(
        s1, s2, s3,
        color="black",
        s=120,
        marker="X",
        label="Nueva muestra"
    )

    ax.set_xlabel("SepalLengthCm")
    ax.set_ylabel("SepalWidthCm")
    ax.set_zlabel("PetalLengthCm")
    ax.legend()

    st.pyplot(fig)



