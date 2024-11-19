import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Dashboard
st.title("CSV/Excel Data Analytics Dashboard")

# Upload File
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        # Display Dataframe
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Basic Dataset Information
        st.subheader("Dataset Information")
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write(df.describe())

        # Select Columns for Analysis
        st.subheader("Select Columns for Analysis")
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        
        if numeric_columns:
            col1, col2 = st.columns(2)
            with col1:
                selected_x = st.selectbox("X-axis", numeric_columns)
            with col2:
                selected_y = st.selectbox("Y-axis", numeric_columns)

            # Scatter Plot
            st.subheader("Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=selected_x, y=selected_y, ax=ax)
            st.pyplot(fig)

            # Correlation Heatmap
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for analysis.")

    except Exception as e:
        st.error(f"Error: {e}")
