import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# App title
st.title("Vomitoxin Prediction")

# File uploader
data_file = st.file_uploader("Upload your CSV file", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Check if target column exists
    if 'vomitoxin_ppb' in data.columns:
        X = data.drop('vomitoxin_ppb', axis=1)
        y = data['vomitoxin_ppb']

        # Convert non-numeric columns and handle missing values
        X = X.select_dtypes(include=[np.number]).fillna(0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA visualization
        st.subheader("PCA Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis')
        plt.colorbar(label='Vomitoxin (ppb)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)

        # Model training
        st.subheader("Model Training before applying any outlier detection technique or any hyperparameter techniques")
        model_choice = st.selectbox("Choose a model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        model = None

        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        elif model_choice == "XGBoost":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

        if st.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.4f}")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.xlabel("Actual Vomitoxin (ppb)")
            plt.ylabel("Predicted Vomitoxin (ppb)")
            plt.title(f"{model_choice} Predictions")
            st.pyplot(plt)

    else:
        st.error("The dataset must contain a 'vomitoxin_ppb' column.")
else:
    st.write("Please upload a CSV file.")

# Display Result Images
st.subheader("Model Results")

github_base_url = "https://raw.githubusercontent.com/Iamkrmayank/Vomitoxin-Prediction/main"

for i in range(1, 7):
    image_url = f"{github_base_url}/result{i}.png"
    st.image(image_url, caption=f"Result {i}", use_container_width=True)

st.write("Developed with Streamlit")

