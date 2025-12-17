# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="User Behaviour Analysis", layout="wide")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "processed" not in st.session_state:
    st.session_state.processed = None
if "page" not in st.session_state:
    st.session_state.page = "home"
if "filename" not in st.session_state:
    st.session_state.filename = None

# 🌟 Custom top navigation bar using Streamlit buttons
st.markdown("""
    <style>
        div.nav-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        div.stButton > button {
            background-color: #e2e8f0;
            color: #1f2937;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: 500;
            border-radius: 6px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #cbd5e1;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    cols = st.columns(5)
    if cols[0].button("🏠 Home"):
        st.session_state.page = "home"
    if cols[1].button("📂 Upload"):
        st.session_state.page = "upload"
    if cols[2].button("⚙️ Detection"):
        st.session_state.page = "model"
    if cols[3].button("📈 Visualize"):
        st.session_state.page = "viz"
    if cols[4].button("💾 Export"):
        st.session_state.page = "download"

st.title("📊 User Behaviour Analysis using Isolation Forest")
st.markdown("An interactive system for detecting anomalies in user activity data using an unsupervised ML model.")

# 🏠 Home
if st.session_state.page == "home":
    st.subheader("Overview")
    st.write("""
    This tool allows you to upload a dataset of user activities, preprocess it, 
    and apply the **Isolation Forest algorithm** to detect anomalous user behaviours.

    The system includes:
    - Automatic data cleaning and encoding
    - Adjustable contamination rate for sensitivity
    - Multiple visualizations: Bar, Pie, and Scatter plots
    - Downloadable results with anomaly labels
    """)

# 📂 Upload
elif st.session_state.page == "upload":
    st.subheader("Upload your dataset (CSV format)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.session_state.filename = uploaded_file.name
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df_encoded = pd.get_dummies(df, drop_first=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_encoded)
        st.session_state.processed = (df_encoded, scaled_data)
        st.success(f"✅ File '{uploaded_file.name}' uploaded and processed successfully!")

    if st.session_state.data is not None:
        st.write(f"### 📋 Preview: {st.session_state.filename}")
        st.dataframe(st.session_state.data.head(), use_container_width=True)

        if st.button("🧹 Clear Uploaded Data"):
            st.session_state.data = None
            st.session_state.processed = None
            st.session_state.filename = None
            st.success("Data cleared. You can upload a new file.")

# ⚙️ Model & Detection
elif st.session_state.page == "model":
    if st.session_state.processed is None:
        st.warning("Please upload and preprocess your dataset first.")
    else:
        st.subheader("Model Configuration")
        contamination = st.slider("Set Contamination Rate (Expected % of anomalies):", 0.01, 0.20, 0.05, 0.01)
        df_encoded, scaled_data = st.session_state.processed

        if st.button("Run Isolation Forest Model"):
            model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
            st.session_state.data["Anomaly_Label"] = model.fit_predict(scaled_data)
            st.session_state.data["Anomaly_Score"] = model.decision_function(scaled_data)
            st.success("✅ Model executed successfully!")

            counts = st.session_state.data["Anomaly_Label"].value_counts()
            st.write("### Detection Summary")
            st.write(pd.DataFrame({
                "Label": ["Normal (1)", "Anomalous (-1)"],
                "Count": [counts.get(1, 0), counts.get(-1, 0)]
            }))

# 📈 Visualizations
elif st.session_state.page == "viz":
    if st.session_state.data is None or "Anomaly_Label" not in st.session_state.data.columns:
        st.warning("Please run the model first.")
    else:
        st.subheader("Visualization Dashboard")
        vis_option = st.selectbox(
            "Choose Visualization Type:",
            ["Bar Chart – Normal vs Anomalies", "Pie Chart – Proportion", "Scatter Plot – Feature Comparison"]
        )

        df = st.session_state.data

        if vis_option == "Bar Chart – Normal vs Anomalies":
            counts = df["Anomaly_Label"].value_counts().reset_index()
            counts.columns = ["Label", "Count"]
            fig = px.bar(
                counts,
                x="Label",
                y="Count",
                color="Label",
                color_discrete_map={1: "skyblue", -1: "salmon"},
                title="Normal vs Anomalous Data Count",
                width=600,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif vis_option == "Pie Chart – Proportion":
            pie_data = df["Anomaly_Label"].value_counts()
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Normal", "Anomalous"],
                    values=[pie_data.get(1, 0), pie_data.get(-1, 0)],
                    marker=dict(colors=["#7dd3fc", "#f87171"]),
                    hole=0.4
                )
            ])
            fig.update_layout(title="Proportion of Anomalies", width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif vis_option == "Scatter Plot – Feature Comparison":
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            x_axis = st.selectbox("Select X-axis feature:", numeric_cols, index=0)
            y_axis = st.selectbox("Select Y-axis feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=df["Anomaly_Label"].map({1: "Normal", -1: "Anomalous"}),
                color_discrete_map={"Normal": "#60a5fa", "Anomalous": "#ef4444"},
                title=f"Scatter Plot ({x_axis} vs {y_axis})",
                width=800,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

# 💾 Download Results
elif st.session_state.page == "download":
    if st.session_state.data is None or "Anomaly_Label" not in st.session_state.data.columns:
        st.warning("Please run the model first.")
    else:
        st.subheader("Download Processed Results")
        buffer = BytesIO()
        st.session_state.data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Download CSV with Anomaly Labels",
            data=buffer,
            file_name="user_behaviour_results.csv",
            mime="text/csv"
        )
        st.success("Your processed dataset is ready for download.")
