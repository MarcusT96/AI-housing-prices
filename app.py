import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ladda datasetet
housing = pd.read_csv('./housing.csv')

# Skapa nya attribut
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']
housing['capped'] = (housing['median_house_value'] == 500001).astype(int)

# Omvandla kategoriska variabler till numeriska
housing = pd.get_dummies(housing, columns=['ocean_proximity'])

# Fyll saknade värden med medianvärden
housing = housing.fillna(housing.median())

# Uppdatera features och target variabler
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

# Dela upp data i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Träna modellen
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Utvärdera prestanda
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

# Streamlit applikation
st.title('California Housing Price Prediction')

# Visa grundläggande information om datasetet
st.write("Dataset Information:")
st.write(housing.info())

# Visa statistisk sammanfattning av datasetet
st.write("Statistical Summary:")
st.write(housing.describe())

# Visa distributionen av målvariabeln
st.write("Distribution of Median House Value:")
fig, ax = plt.subplots()
housing['median_house_value'].hist(bins=50, ax=ax, figsize=(10,5))
ax.set_xlabel('Median House Value')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Visa modellens prestanda
st.write("Model Performance:")
st.write(f'MAE: {rf_mae}')
st.write(f'MSE: {rf_mse}')
st.write(f'R²: {rf_r2}')

# Scatter plot av faktiska vs förutsagda värden
st.write("Actual vs Predicted Median House Values:")
fig, ax = plt.subplots()
ax.scatter(y_test, rf_y_pred, alpha=0.5)
ax.set_xlabel('Actual Median House Value')
ax.set_ylabel('Predicted Median House Value')
ax.set_title('Actual vs Predicted Median House Values (Random Forest) with Capped Indicator')
st.pyplot(fig)

# Residual plot
st.write("Residuals vs Actual Median House Values:")
residuals = y_test - rf_y_pred
fig, ax = plt.subplots()
ax.scatter(y_test, residuals, alpha=0.5)
ax.set_xlabel('Actual Median House Value')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Actual Median House Values (Random Forest) with Capped Indicator')
st.pyplot(fig)