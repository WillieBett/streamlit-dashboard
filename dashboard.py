import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Dashboard
st.title("Exploratory Data Analysis (EDA) Dashboard")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        # Dataset Overview
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Dataset Information
        st.subheader("Dataset Information")
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write("**Column Data Types:**")
        st.write(df.dtypes)

        # Missing Values
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Summary Statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())

        # Unique Value Counts
        st.subheader("Unique Value Counts")
        st.write({col: df[col].nunique() for col in df.columns})

        # Select Columns for Visualizations
        st.subheader("Select Columns for Visualizations")
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_columns:
            col1, col2 = st.columns(2)
            with col1:
                selected_x = st.selectbox("X-axis (Numeric)", numeric_columns)
            with col2:
                selected_y = st.selectbox("Y-axis (Numeric)", numeric_columns)

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

        if categorical_columns:
            selected_cat = st.selectbox("Select a Categorical Column", categorical_columns)

            # Bar Plot
            st.subheader("Bar Plot")
            fig, ax = plt.subplots()
            df[selected_cat].value_counts().plot(kind='bar', ax=ax)
            plt.title(f"Bar Plot for {selected_cat}")
            plt.ylabel("Frequency")
            st.pyplot(fig)

        # Pair Plot
        if st.checkbox("Show Pair Plot"):
            st.subheader("Pair Plot")
            fig = sns.pairplot(df[numeric_columns])
            st.pyplot(fig)

        # Distribution Plot
        if numeric_columns:
            selected_num = st.selectbox("Select a Numeric Column for Distribution", numeric_columns)
            st.subheader("Distribution Plot")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num], kde=True, ax=ax)
            plt.title(f"Distribution of {selected_num}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV or Excel file.")
