import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data, split_data
from model import train_model
from plots import plot_distribution, plot_actual_vs_predicted, plot_residuals

st.title('California Housing Price Prediction')

# Filuppladdningsfunktion
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    housing = load_data(uploaded_file)
    housing = preprocess_data(housing)
    X, y = split_data(housing)
    
    rf_model, rf_mae, rf_mse, rf_r2, X_test, y_test, rf_y_pred = train_model(X, y)

    # Visa grundläggande information om datasetet
    st.write("Dataset Information:")
    st.write(housing.info())

    # Visa statistisk sammanfattning av datasetet
    st.write("Statistical Summary:")
    st.write(housing.describe())

    # Visa distributionen av målvariabeln
    st.write("Distribution of Median House Value:")
    st.pyplot(plot_distribution(housing))

    # Visa modellens prestanda
    st.write("Model Performance:")
    st.write(f'MAE: {rf_mae}')
    st.write(f'MSE: {rf_mse}')
    st.write(f'R²: {rf_r2}')

    # Scatter plot av faktiska vs förutsagda värden
    st.write("Actual vs Predicted Median House Values:")
    st.plotly_chart(plot_actual_vs_predicted(y_test, rf_y_pred))

    # Residual plot
    st.write("Residuals vs Actual Median House Values:")
    residuals = y_test - rf_y_pred
    st.plotly_chart(plot_residuals(y_test, residuals))
