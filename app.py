import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.set_page_config(page_title="Iris Dashboard", layout="wide")

st.markdown("""
    <style>
        body { font-size: 10px; }
        .stPlotlyChart { height: 300px !important; }
        .css-1d391kg { padding: 0rem 1rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title(" Iris Species Classification Dashboard")

# LOAD DATASET
st.sidebar.header("Dataset Loader")
uploaded = st.sidebar.file_uploader("Upload Iris.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# TABS
tabs = st.tabs([
    "Dataset Overview",
    "Model Training",
    "Prediction"
])

# TAB 1 — Dataset Overview

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
        height=350,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

 
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        colorscale="Blues"
    )
    fig_corr.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Relationships Between Variables")
    fig_sm = px.scatter_matrix(
        df,
        dimensions=df.select_dtypes(include=["float", "int"]).columns,
        color="Species",
        title="",
        height=450
    )
    st.plotly_chart(fig_sm, use_container_width=True)
    
import plotly.graph_objects as go

corr = df.corr(numeric_only=True)

edges = []
for i in corr.columns:
    for j in corr.columns:
        if i != j:
            weight = corr.loc[i, j]
            edges.append((i, j, abs(weight)))

nodes = list(corr.columns)

edge_x = []
edge_y = []
for edge in edges:
    edge_x.append(nodes.index(edge[0]))
    edge_y.append(nodes.index(edge[2]))

fig = go.Figure(data=go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="markers+text",
    text=nodes,
    textposition="bottom center"
))

fig.update_layout(
    title="Correlation Network",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)










# TAB 2 — Model Training

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


# TAB 3 — Prediction

with tabs[2]:

    st.subheader("Enter measurements to predict species")

    colA, colB = st.columns(2)
    with colA:
        s1 = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.4)
        s2 = st.number_input("Sepal Width",  min_value=0.0, max_value=10.0, value=3.0)
    with colB:
        s3 = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.5)
        s4 = st.number_input("Petal Width",  min_value=0.0, max_value=10.0, value=1.5)

    if st.button("Predict"):
        sample = [[s1, s2, s3, s4]]
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]

        st.success(f" Predicted Species: **{pred}**")

        # ------------ 3D SCATTER ----------
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

        # Prediction point
        fig3d.add_scatter3d(
            x=[s1], y=[s2], z=[s3],
            mode='markers',
            marker=dict(size=6, color="black", symbol="x"),
            name="New Sample"
        )

        st.plotly_chart(fig3d, use_container_width=True)







