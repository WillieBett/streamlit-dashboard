import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Streamlit App Title
st.title("Enhanced CSV/Excel Data Analytics Dashboard with ML Models")

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
        st.write(df.describe(include='all').T)

        # Missing Values Analysis
        st.subheader("Missing Values Analysis")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        if st.checkbox("Handle Missing Data"):
            method = st.selectbox("Choose a method", ["Drop rows", "Fill with mean", "Fill with median"])
            if method == "Drop rows":
                df = df.dropna()
            elif method == "Fill with mean":
                df = df.fillna(df.mean())
            elif method == "Fill with median":
                df = df.fillna(df.median())
            st.success("Missing values handled!")

        # Numeric and Categorical Columns
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(include="object").columns.tolist()

        # Visualization Options
        st.subheader("Data Visualizations")
        if numeric_columns:
            # Distribution Plots
            selected_dist_col = st.selectbox("Select Numeric Column for Distribution", numeric_columns)
            if selected_dist_col:
                fig, ax = plt.subplots()
                sns.histplot(df[selected_dist_col], kde=True, ax=ax)
                st.pyplot(fig)

            # Scatter Plot
            if len(numeric_columns) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    selected_x = st.selectbox("X-axis", numeric_columns)
                with col2:
                    selected_y = st.selectbox("Y-axis", numeric_columns)

                if selected_x and selected_y:
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
            # Bar Chart for Categorical Columns
            selected_cat_col = st.selectbox("Select Categorical Column for Bar Chart", categorical_columns)
            if selected_cat_col:
                fig, ax = plt.subplots()
                df[selected_cat_col].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)

        # Outlier Detection
        st.subheader("Outlier Detection")
        selected_outlier_col = st.selectbox("Select Numeric Column for Outlier Analysis", numeric_columns)
        if selected_outlier_col:
            Q1 = df[selected_outlier_col].quantile(0.25)
            Q3 = df[selected_outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[selected_outlier_col] < (Q1 - 1.5 * IQR)) | (df[selected_outlier_col] > (Q3 + 1.5 * IQR))]
            st.write(f"Outliers in {selected_outlier_col}:")
            st.write(outliers)

        # Export Processed Data
        st.subheader("Export Processed Data")
        if st.button("Download Processed Dataset"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")

        # Machine Learning Models
        st.subheader("Train a Machine Learning Model")
        target_column = st.selectbox("Select Target Column", df.columns)

        if target_column:
            features = df.drop(columns=[target_column])
            target = df[target_column]

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            if target.dtypes == 'object':
                # Classification Model
                st.write("Training a Random Forest Classifier...")
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"**Accuracy:** {accuracy:.2f}")
            else:
                # Regression Model
                st.write("Training a Linear Regression Model...")
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                st.write(f"**Mean Squared Error:** {mse:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
