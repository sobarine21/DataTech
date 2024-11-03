import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime
from io import StringIO, BytesIO
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

def save_to_pdf(df, filename):
    pdf = BytesIO()
    df.to_csv(pdf, index=False)
    b64 = base64.b64encode(pdf.getvalue()).decode()
    href = f'<a href="data:file/pdf;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)

def preprocess_text(text_column):
    return text_column.str.lower().str.replace('[^\w\s]', '').str.strip()

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
        st.write("Data Preview:", df.head())

        # Data Profiling Report
        st.subheader("Data Profiling")
        st.write("Shape of data:", df.shape)
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

        # Feature Engineering and Scaling
        st.subheader("Feature Engineering and Scaling")
        if st.checkbox("Scale Data"):
            scaler = st.selectbox("Choose scaling method", ("Standard Scaler", "Min-Max Scaler"))
            if scaler == "Standard Scaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            df[df.select_dtypes(include=['float', 'int']).columns] = scaler.fit_transform(df.select_dtypes(include=['float', 'int']))
            st.write("Data scaled.")

        if st.checkbox("Feature Engineering"):
            feature_col1 = st.selectbox("Select first column for feature creation:", df.columns)
            feature_col2 = st.selectbox("Select second column for feature creation:", df.columns)
            if feature_col1 and feature_col2:
                df['new_feature'] = df[feature_col1] + df[feature_col2]
                st.write("New feature created:", df[['new_feature']].head())

        # Data Visualization
        st.subheader("Data Visualization")
        visualize_data(df)
        
        if st.checkbox("Histogram"):
            column = st.selectbox("Select column for histogram", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        # Machine Learning Model Training
        st.subheader("Machine Learning")
        target = st.selectbox("Select target column", df.columns)
        
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
            kmeans = KMeans(n_clusters=num_clusters)
            df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=['float', 'int']))
            st.write("Clustering completed.")
            st.write(df.head())

        # Download Processed Data
        st.subheader("Download Processed Data")
        if st.button("Download CSV"):
            save_to_pdf(df, "processed_data")

if __name__ == "__main__":
    main()
