import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# PAGE CONFIG
st.set_page_config(page_title="Supermarket Sales Forecast", layout="wide")

# Load static CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    df['week'] = pd.to_datetime(df['week'])
    return df

# Load
df = load_data()

# Title
st.title("🛒 Supermarket Sales Forecasting")
st.write("This app provides data visualization and sales prediction using Random Forest.")

# ================================
# SECTION 1: Data & Visualization
# ================================
st.subheader("📊 Data Overview")
st.dataframe(df.head())

# Chart 1: Total Sales by Week
sales_by_week = df.groupby('week')['sales'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=sales_by_week, x='week', y='sales', ax=ax1)
ax1.set_title("Total Sales by Week")
st.pyplot(fig1)

# Chart 2: Sales by Category
fig2, ax2 = plt.subplots()
sns.barplot(data=df, x='category', y='sales', estimator=sum, ci=None, ax=ax2)
ax2.set_title("Sales by Category")
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Chart 3: Correlation Matrix
fig3, ax3 = plt.subplots()
corr = df[['sales', 'price', 'revenue', 'promotion', 'holiday']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# ==================================
# SECTION 2: Machine Learning Model
# ==================================
st.subheader("📈 Sales Forecast with Random Forest")

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['region', 'category']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['region', 'category']))

df_model = df.drop(columns=['week', 'product_id', 'region', 'category'])
df_final = pd.concat([df_model.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Split data
X = df_final.drop(columns='sales')
y = df_final['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.markdown(f"""
- **MAE:** {mae:.2f}  
- **RMSE:** {rmse:.2f}  
- **R² Score:** {r2:.2f}
""")

# Scatterplot: Actual vs Predicted
fig4, ax4 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax4, alpha=0.6)
ax4.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
ax4.set_xlabel('Actual Sales')
ax4.set_ylabel('Predicted Sales')
ax4.set_title('Actual vs Predicted Sales')
st.pyplot(fig4)
