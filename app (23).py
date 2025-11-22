import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.title(" Iris Species Classifier — Streamlit App")

# ---------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------
st.sidebar.header("Carga del dataset")
uploaded = st.sidebar.file_uploader("Sube el archivo Iris.csv", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Eliminar columna Id si existe
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    st.subheader("Vista previa del dataset")
    st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------
    # VISUALIZACIÓN 3D DEL DATASET (EDA)
    # ---------------------------------------------------
    st.subheader("Visualización 3D del dataset (EDA)")

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

    ax.set_xlabel("SepalLengthCm")
    ax.set_ylabel("SepalWidthCm")
    ax.set_zlabel("PetalLengthCm")
    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------------------
    # ENTRENAMIENTO DEL MODELO
    # ---------------------------------------------------
    st.subheader("Entrenamiento del Modelo")

    X = df.drop(columns=["Species"])
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train_scaled, y_train)

    st.success("Modelo entrenado correctamente")

    # ---------------------------------------------------
    # PREDICCIÓN MANUAL + 3D
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

        # Gráfico 3D con el punto predicho
        st.subheader("Posición de la nueva muestra en el espacio 3D")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Dataset original
        for species in df["Species"].unique():
            subset = df[df["Species"] == species]
            ax.scatter(
                subset["SepalLengthCm"],
                subset["SepalWidthCm"],
                subset["PetalLengthCm"],
                label=species
            )

        # Punto predicho
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

else:
    st.info("Sube el archivo Iris.csv para continuar.")

