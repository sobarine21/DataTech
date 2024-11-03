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
from sklearn.impute import KNNImputer
from datetime import datetime
from io import BytesIO
import base64
import json

# Helper Functions
def load_data(file):
    """Load data from various file formats."""
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
    """Save DataFrame to a CSV file for download."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)

def preprocess_text(text_column):
    """Preprocess text data: lowercase, remove punctuation, and strip whitespace."""
    return text_column.str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()

def visualize_data(df):
    """Visualize the correlation heatmap and box plots for outlier detection."""
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

def outlier_removal(df, column):
    """Remove outliers based on the IQR method."""
    if column:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df

def fill_missing_with_knn(df, n_neighbors=5):
    """Fill missing values using KNN imputation."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[df.select_dtypes(include=['float', 'int']).columns] = imputer.fit_transform(df.select_dtypes(include=['float', 'int']))
    return df

def generate_summary_statistics(df):
    """Generate and display summary statistics of the DataFrame."""
    st.subheader("Summary Statistics")
    st.write(df.describe())

def date_parser(df, column):
    """Convert specified column to datetime format."""
    if column:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        st.write(f"Converted {column} to datetime.")
    return df

def feature_importance(model, X, y):
    """Display feature importance from the trained model."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    st.bar_chart(feature_importance_df.set_index('Feature'))

def apply_one_hot_encoding(df):
    """Apply one-hot encoding to categorical variables."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def time_series_analysis(df, date_column):
    """Perform time series analysis on the selected date column."""
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    st.line_chart(df)

def get_data_shapes(df):
    """Display the shape of the DataFrame and each column."""
    st.write("Data Shape:", df.shape)
    st.write("Column Shapes:", {col: df[col].shape for col in df.columns})

def missing_values_heatmap(df):
    """Display a heatmap of missing values in the DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot(plt)

def correlation_matrix(df):
    """Display a correlation matrix heatmap."""
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
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
        st.write("Data Preview:", df.head())

        # Data Profiling Report
        st.subheader("Data Profiling")
        get_data_shapes(df)
        st.write("Data Types:", df.dtypes)
        st.write("Missing values per column:", df.isna().sum())

        # Data Cleansing Options
        st.subheader("Data Cleansing")
        if st.checkbox("Remove Duplicates"):
            df = df.drop_duplicates()
            st.write("Duplicates removed.")

        if st.checkbox("Handle Missing Values"):
            missing_method = st.selectbox("Choose method", ("Mean", "Median", "Mode", "KNN"))
            if missing_method == "Mean":
                df.fillna(df.mean(), inplace=True)
            elif missing_method == "Median":
                df.fillna(df.median(), inplace=True)
            elif missing_method == "Mode":
                df.fillna(df.mode().iloc[0], inplace=True)
            elif missing_method == "KNN":
                n_neighbors = st.slider("Select number of neighbors", 1, 10, 5)
                df = fill_missing_with_knn(df, n_neighbors)
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
            if column:
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
            feature_importance(model, X, y)

        if st.checkbox("Perform Clustering (KMeans)"):
            num_clusters = st.slider("Number of clusters", 2, 10, 3)
            if df.shape[0] > num_clusters:
                kmeans = KMeans(n_clusters=num_clusters)
                df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=['float', 'int']))
                st.write("Clustering completed.")
                st.write(df.head())
            else:
                st.warning("Number of samples is less than number of clusters.")

        # Time Series Analysis
        if st.checkbox("Time Series Analysis"):
            date_column = st.selectbox("Select date column", df.columns)
            if date_column:
                time_series_analysis(df, date_column)

        # Download Processed Data
        st.subheader("Download Processed Data")
        if st.button("Download CSV"):
            save_to_csv(df, "processed_data")

if __name__ == "__main__":
    main()
