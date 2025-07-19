import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="🛒 Supermarket Sales Forecast", layout="wide")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    df['week'] = pd.to_datetime(df['week'])
    return df

df = load_data()

# Title
st.title("🛒 Supermarket Sales Forecasting App")
st.write("Visualize supermarket sales and forecast future values using machine learning.")

# =======================
# 📊 Section: Visualizations
# =======================
st.subheader("📊 Exploratory Data Analysis")

# Chart 1: Sales by Week
sales_by_week = df.groupby('week')['sales'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=sales_by_week, x='week', y='sales', ax=ax1)
ax1.set_title("1️⃣ Total Sales by Week")
st.pyplot(fig1)

# Chart 2: Sales by Category
fig2, ax2 = plt.subplots()
sns.barplot(data=df, x='category', y='sales', estimator=sum, ci=None, ax=ax2, palette='pastel')
ax2.set_title("2️⃣ Total Sales by Product Category")
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Chart 3: Sales by Region
fig3, ax3 = plt.subplots()
sns.barplot(data=df, x='region', y='sales', estimator=sum, ci=None, ax=ax3, palette='muted')
ax3.set_title("3️⃣ Total Sales by Region")
st.pyplot(fig3)

# Chart 4: Sales by Promotion
fig4, ax4 = plt.subplots()
sns.boxplot(data=df, x='promotion', y='sales', ax=ax4)
ax4.set_title("4️⃣ Sales by Promotion (0 = No, 1 = Yes)")
st.pyplot(fig4)

# Chart 5: Sales by Holiday
fig5, ax5 = plt.subplots()
sns.boxplot(data=df, x='holiday', y='sales', ax=ax5)
ax5.set_title("5️⃣ Sales on Holidays vs Normal Days")
st.pyplot(fig5)

# Chart 6: Price vs Sales
fig6, ax6 = plt.subplots()
sns.scatterplot(data=df, x='price', y='sales', hue='promotion', alpha=0.6, ax=ax6)
ax6.set_title("6️⃣ Price vs Sales (Colored by Promotion)")
st.pyplot(fig6)

# Chart 7: Sales vs Revenue
fig7, ax7 = plt.subplots()
sns.scatterplot(data=df, x='sales', y='revenue', hue='category', alpha=0.7, ax=ax7)
ax7.set_title("7️⃣ Sales vs Revenue (Colored by Category)")
st.pyplot(fig7)

# Chart 8: Correlation Matrix
fig8, ax8 = plt.subplots()
corr = df[['sales', 'price', 'revenue', 'promotion', 'holiday']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax8)
ax8.set_title("8️⃣ Correlation Matrix")
st.pyplot(fig8)

# ========================
# 📈 Section: Forecasting
# ========================
st.subheader("📈 Sales Forecasting Using Random Forest")

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['region', 'category']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['region', 'category']))

# Prepare model data
df_model = df.drop(columns=['week', 'product_id', 'region', 'category'])
df_final = pd.concat([df_model.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Split
X = df_final.drop(columns='sales')
y = df_final['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display results
st.markdown("### 🧮 Evaluation Metrics")
st.markdown(f"- **MAE (Mean Absolute Error):** `{mae:.2f}`")
st.markdown(f"- **RMSE (Root Mean Squared Error):** `{rmse:.2f}`")
st.markdown(f"- **R² Score:** `{r2:.2f}`")

# Prediction scatter plot
fig9, ax9 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax9)
ax9.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
ax9.set_xlabel("Actual Sales")
ax9.set_ylabel("Predicted Sales")
ax9.set_title("📌 Actual vs Predicted Sales")
st.pyplot(fig9)

# Final note
st.success("✅ Forecasting Complete! You can improve model performance by tuning parameters or trying other algorithms.")
