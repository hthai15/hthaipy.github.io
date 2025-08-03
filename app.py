import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Sales Analysis App", layout="wide")
sns.set(style='whitegrid')

st.title("📊 Sales Analysis and Prediction Dashboard")

# Đọc file CSV mẫu đã upload sẵn
@st.cache_data
def load_data():
    return pd.read_csv("supermarket_sales_forecast_sample.csv")

df = load_data()

st.subheader("🔍 Data Overview")
st.write("**Dataset Info:**")
st.dataframe(df.head())
st.write("**Null values:**", df.isnull().sum())
st.write("**Duplicate rows:**", df.duplicated().sum())
st.write("**Data types:**")
st.write(df.dtypes)

# Drop duplicates
df = df.drop_duplicates()

# Scale sales
scaler = StandardScaler()
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

st.subheader("📈 Statistical Summary")
st.write(df.describe())

# Insights
st.subheader("🔎 Key Insights")
col1, col2 = st.columns(2)

with col1:
    top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(5)
    st.write("🔝 **Top 5 sản phẩm bán chạy:**")
    st.dataframe(top_products)

    avg_sales_region = df.groupby('region')['sales'].mean().sort_values(ascending=False)
    st.write("📍 **Doanh số trung bình theo khu vực:**")
    st.dataframe(avg_sales_region)

with col2:
    weekly_sales = df.groupby('week')['sales'].sum()
    st.metric("📆 Số tuần có mặt trong dữ liệu", df['week'].nunique())
    st.metric("📦 Tổng số sản phẩm khác nhau", df['product_id'].nunique())
    st.metric("💰 Tổng doanh số", f"{df['sales'].sum():,.2f}")
    st.metric("📈 Doanh số trung bình", f"{df['sales'].mean():,.2f}")

    st.write(f"📈 **Tuần cao nhất**: Tuần {weekly_sales.idxmax()} với {weekly_sales.max():,.2f}")
    st.write(f"📉 **Tuần thấp nhất**: Tuần {weekly_sales.idxmin()} với {weekly_sales.min():,.2f}")

# Charts
st.subheader("📊 Visualizations")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['sales'], kde=True, color='skyblue', bins=30, ax=ax1)
ax1.set_title('Phân phối doanh số bán hàng')
st.pyplot(fig1)

top10 = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top10.index.astype(str), y=top10.values, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy')
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(7,5))
sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
ax3.set_title('Ảnh hưởng của Promotion đến Sales')
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(7,5))
sns.boxplot(x='holiday', y='sales', data=df, ax=ax4)
ax4.set_title('Ảnh hưởng của Holiday đến Sales')
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax5)
ax5.set_title('Xu hướng doanh số theo tuần')
st.pyplot(fig5)

# Modeling
st.subheader("🤖 Sales Prediction Model")
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.write("### 🔍 ĐÁNH GIÁ MÔ HÌNH")
st.write(f"📌 MAE: {mae:.2f}")
st.write(f"📌 MSE: {mse:.2f}")
st.write(f"📌 RMSE: {rmse:.2f}")
st.write(f"📌 R-squared (R²): {r2:.2f}")

st.write("### 📊 So sánh thực tế vs dự đoán")
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(y_test.values[:30], label='Thực tế', marker='o')
ax6.plot(y_pred[:30], label='Dự đoán', marker='x')
ax6.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
ax6.legend()
st.pyplot(fig6)

st.write("### 🔍 Phân phối sai số")
errors = y_test - y_pred
fig7, ax7 = plt.subplots(figsize=(10, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax7)
ax7.set_title("Phân phối sai số dự đoán")
st.pyplot(fig7)
