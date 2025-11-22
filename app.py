import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------
st.set_page_config(page_title="Iris Classifier Dashboard", layout="wide")
st.title("🌸 Iris Species Classification Dashboard")

# ---------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------
st.sidebar.header("Carga del dataset")
uploaded = st.sidebar.file_uploader("Sube el archivo Iris.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Iris.csv")  # Carga automática sin mensajes

# Limpiar ID si existe
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# ---------------------------------------------------
# TABS SEGÚN LO QUE PIDE EL PDF
# ---------------------------------------------------
tabs = st.tabs(["📊 Exploración del Dataset", 
                "🤖 Modelo de Clasificación", 
                "🔮 Predicción Manual"])

# ===================================================
# 📊 TAB 1 — Exploración del Dataset (EDA)
# ===================================================
with tabs[0]:

    st.subheader("Vista previa del dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Distribución de especies")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="Species", ax=ax)
    st.pyplot(fig)

    st.subheader("Matriz de correlación")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===================================================
# 🤖 TAB 2 — Entrenamiento del Modelo (OBLIGATORIO PDF)
# ===================================================
with tabs[1]:

    st.subheader("Entrenamiento del modelo")

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

    st.success("Modelo entrenado con éxito")

    # -----------------------
    # MÉTRICAS
    # -----------------------
    y_pred = model.predict(X_test_scaled)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.3f}")
    col4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

# ===================================================
# 🔮 TAB 3 — Predicción Manual + Gráfico 3D (OBLIGATORIO PDF)
# ===================================================
with tabs[2]:

    st.subheader("Ingresar características para predecir la especie")

    colA, colB = st.columns(2)
    with colA:
        s1 = st.number_input("SepalLengthCm", min_value=0.0, max_value=10.0, value=5.4)
        s2 = st.number_input("SepalWidthCm", min_value=0.0, max_value=10.0, value=3.0)
    with colB:
        s3 = st.number_input("PetalLengthCm", min_value=0.0, max_value=10.0, value=4.5)
        s4 = st.number_input("PetalWidthCm", min_value=0.0, max_value=10.0, value=1.5)

    if st.button("Predecir especie"):
        sample = [[s1, s2, s3, s4]]
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]

        st.success(f"🌼 La especie predicha es: **{pred}**")

        # -------------------------
        # GRÁFICO 3D CON PREDICCIÓN
        # -------------------------
        st.subheader("Visualización 3D con la muestra predicha")

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





