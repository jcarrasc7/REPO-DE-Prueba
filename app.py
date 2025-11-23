import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Dashboard", layout="wide")

st.markdown("""
    <style>
        body { font-size: 10px; }
        .stPlotlyChart { height: 300px !important; }
        .css-1d391kg { padding: 0rem 1rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Iris Species Classification Dashboard")

df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# TABS
tabs = st.tabs([
    "Dataset Overview",
    "Model Training",
    "Prediction"
])

#  TAB 1 
with tabs[0]:

    st.subheader("Quick Preview")
    st.dataframe(df.head(8), use_container_width=True)

    colA, colB, colC = st.columns(3)
    colA.metric("Rows", df.shape[0])
    colB.metric("Columns", df.shape[1])
    colC.metric("Classes", df["Species"].nunique())

    st.subheader("Class Balance")
    fig_bar = px.bar(
        df,
        x="Species",
        color="Species",
        title="",
        height=370,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Relationships Between Variables")

    
    df_rename = df.rename(columns={
        "SepalLengthCm": "Sepal Length",
        "SepalWidthCm": "Sepal Width",
        "PetalLengthCm": "Petal Length",
        "PetalWidthCm": "Petal Width"
    })

    fig_sm = px.scatter_matrix(
        df_rename,
        dimensions=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        color="Species",
        title="",
        height=650
    )
    
    fig_sm.update_layout(
        xaxis_tickangle=45,
        yaxis_tickangle=45,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig_sm, use_container_width=True)


# TAB 2 
with tabs[1]:

    st.subheader("Model Training")

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

    st.success("Model trained successfully!")

    y_pred = model.predict(X_test_scaled)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.3f}")
    col4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

    
    st.subheader("Training vs Testing Accuracy")

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    fig_acc = px.bar(
    x=["Train Accuracy", "Test Accuracy"],
    y=[train_acc, test_acc],
    title="Train vs Test Accuracy",
    height=400
    )

    fig_acc.update_layout(yaxis=dict(range=[0,1]))
    st.plotly_chart(fig_acc, use_container_width=True)



# TAB 3
with tabs[2]:

    st.subheader("Enter measurements to predict species")

    colA, colB = st.columns(2)
    with colA:
        s1 = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.4)
        s2 = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    with colB:
        s3 = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.5)
        s4 = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.5)

    if st.button("Predict"):

        sample = [[s1, s2, s3, s4]]
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]

        st.success(f"Predicted Species: {pred}")

        st.subheader("3D Visualization")

        fig3d = px.scatter_3d(
            df,
            x="SepalLengthCm",
            y="SepalWidthCm",
            z="PetalLengthCm",
            color="Species",
            opacity=0.7,
            height=450
        )

        fig3d.add_scatter3d(
            x=[s1], y=[s2], z=[s3],
            mode="markers",
            marker=dict(size=6, color="black", symbol="x"),
            name="New Sample"
        )

        st.plotly_chart(fig3d, use_container_width=True)
