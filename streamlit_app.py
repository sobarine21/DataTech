import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime
from io import BytesIO
import base64
import json

# Helper Functions
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Unsupported file type.")
        return None

def save_to_csv(df, filename):
    csv = BytesIO()
    df.to_csv(csv, index=False)
    b64 = base64.b64encode(csv.getvalue()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)

def visualize_data(df):
    st.subheader("Correlation Heatmap")
    correlation = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation, ax=ax, annot=True, cmap='coolwarm')
    st.pyplot(fig)

    st.subheader("Box Plot for Outliers")
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    column = st.selectbox("Select a column for box plot:", numerical_cols)
    if column:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        st.pyplot(fig)

def apply_one_hot_encoding(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# New Functions
def generate_summary_statistics(df):
    st.subheader("Summary Statistics")
    st.write(df.describe())

def calculate_skewness_kurtosis(df):
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    skewness = df[numerical_cols].skew()
    kurtosis = df[numerical_cols].kurt()
    st.write("Skewness:\n", skewness)
    st.write("Kurtosis:\n", kurtosis)

def feature_interaction(df):
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    col1 = st.selectbox("Select first column for interaction:", numerical_cols)
    col2 = st.selectbox("Select second column for interaction:", numerical_cols)
    if col1 and col2:
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        st.write("Interaction feature created:", df[[f'{col1}_x_{col2}']].head())

def cumulative_sum_analysis(df):
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    column = st.selectbox("Select a column for cumulative sum analysis:", numerical_cols)
    if column:
        df['Cumulative Sum'] = df[column].cumsum()
        st.line_chart(df['Cumulative Sum'])

def missing_values_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot(plt)

# Main Application
def main():
    st.title("Advanced Data Analysis and Automation Tool")

    # User Authentication
    st.sidebar.header("Login")
    user = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if user == "admin" and password == "admin":
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid login")
            return

    # File Upload
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json"])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:  # Ensure data was loaded successfully
            st.write("Data Preview:", df.head())

            # Data Profiling Report
            st.subheader("Data Profiling")
            st.write("Data Shape:", df.shape)
            st.write("Data Types:", df.dtypes)
            st.write("Missing values per column:", df.isna().sum())

            # Data Cleansing Options
            st.subheader("Data Cleansing")
            if st.checkbox("Remove Duplicates"):
                df = df.drop_duplicates()
                st.write("Duplicates removed.")

            if st.checkbox("Handle Missing Values"):
                missing_method = st.selectbox("Choose method", ("Mean", "Median", "Mode"))
                if missing_method == "Mean":
                    df.fillna(df.mean(), inplace=True)
                elif missing_method == "Median":
                    df.fillna(df.median(), inplace=True)
                elif missing_method == "Mode":
                    df.fillna(df.mode().iloc[0], inplace=True)
                st.write("Missing values handled.")

            # Data Scaling and Encoding
            st.subheader("Feature Engineering and Scaling")
            if st.checkbox("Scale Data"):
                scaler = st.selectbox("Choose scaling method", ("Standard Scaler", "Min-Max Scaler"))
                if scaler == "Standard Scaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                df[df.select_dtypes(include=['float', 'int']).columns] = scaler.fit_transform(df.select_dtypes(include=['float', 'int']))
                st.write("Data scaled.")

            if st.checkbox("One-Hot Encode Categorical Variables"):
                df = apply_one_hot_encoding(df)
                st.write("One-hot encoding applied.")

            # Data Visualization
            st.subheader("Data Visualization")
            visualize_data(df)

            if st.checkbox("Histogram"):
                column = st.selectbox("Select column for histogram", df.columns)
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

            # Summary Statistics and Additional Analysis
            if st.checkbox("Generate Summary Statistics"):
                generate_summary_statistics(df)

            if st.checkbox("Calculate Skewness and Kurtosis"):
                calculate_skewness_kurtosis(df)

            if st.checkbox("Create Feature Interaction"):
                feature_interaction(df)

            if st.checkbox("Cumulative Sum Analysis"):
                cumulative_sum_analysis(df)

            if st.checkbox("Missing Values Heatmap"):
                missing_values_heatmap(df)

            # Machine Learning Model Training
            st.subheader("Machine Learning")
            target = st.selectbox("Select target column", df.columns)

            if target and df.shape[0] > 1:  # Ensure there are enough samples
                if st.checkbox("Train Linear Regression Model"):
                    X = df.drop(target, axis=1)
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    st.write("Linear Regression Mean Squared Error:", mse)

                if st.checkbox("Train Classification Model (Random Forest)"):
                    X = df.drop(target, axis=1)
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    st.write("Random Forest Accuracy:", accuracy)

                if st.checkbox("Perform Clustering (KMeans)"):
                    num_clusters = st.slider("Number of clusters", 2, 10, 3)
                    if df.shape[0] > num_clusters:
                        kmeans = KMeans(n_clusters=num_clusters)
                        df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=['float', 'int']))
                        st.write("Clustering completed.")
                        st.write(df.head())
                    else:
                        st.warning("Number of samples is less than number of clusters.")

            # Download Processed Data
            st.subheader("Download Processed Data")
            if st.button("Download CSV"):
                save_to_csv(df, "processed_data")

if __name__ == "__main__":
    main()
