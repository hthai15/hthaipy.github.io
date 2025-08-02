import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình hiển thị
st.set_page_config(page_title="📊 Dự báo doanh số siêu thị", layout="wide")
sns.set(style='whitegrid')

# --- Tải dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv('supermarket_sales_forecast_sample.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

st.title("📈 Ứng dụng Dự báo Doanh số Bán hàng Siêu thị")
st.markdown("## 1. 🧾 Tổng quan dữ liệu")

# --- Tổng quan ---
st.write("### 🗂️ Dữ liệu mẫu")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.write("**Thông tin dữ liệu**")
    buffer = df.info(buf=None)
    st.text(str(buffer))

with col2:
    st.write("**Thống kê mô tả**")
    st.dataframe(df.describe())

# --- Phân tích Missing và Duplicate ---
st.write("### 📌 Kiểm tra dữ liệu thiếu & trùng")
st.write("**Dữ liệu thiếu:**")
st.write(df.isnull().sum())
st.write(f"**Số dòng trùng lặp:** {df.duplicated().sum()}")

# --- Chuẩn hoá cột Sales ---
scaler = StandardScaler()
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# --- Biểu đồ phân phối sales ---
st.markdown("## 2. 📊 Trực quan dữ liệu")

st.subheader("🔹 Phân phối doanh số")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df['sales'], kde=True, bins=30, color='skyblue', ax=ax1)
ax1.set_title('Phân phối Doanh số (Sales)')
st.pyplot(fig1)

# --- Top 10 sản phẩm ---
st.subheader("🔹 Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy nhất')
ax2.set_xlabel('Product ID')
ax2.set_ylabel('Tổng Sales')
st.pyplot(fig2)

# --- Boxplot ảnh hưởng promotion & holiday ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔹 Ảnh hưởng của Promotion")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
    ax3.set_title('Promotion vs Sales')
    st.pyplot(fig3)

with col4:
    st.subheader("🔹 Ảnh hưởng của Holiday")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='holiday', y='sales', data=df, ax=ax4)
    ax4.set_title('Holiday vs Sales')
    st.pyplot(fig4)

# --- Xây dựng mô hình ---
st.markdown("## 3. 🤖 Dự báo Doanh số với Linear Regression")

# One-hot encode
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)

# Feature & Label
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.write("### 📋 Đánh giá mô hình Linear Regression:")
st.markdown(f"""
- **📌 MAE (Sai số trung bình tuyệt đối):** `{mae:.2f}`  
- **📌 MSE (Sai số bình phương trung bình):** `{mse:.2f}`  
- **📌 RMSE (Căn bậc hai của MSE):** `{rmse:.2f}`  
- **📌 R² (Hệ số xác định):** `{r2:.2f}`
""")

# --- Biểu đồ dự đoán: thực tế vs dự đoán ---
st.subheader("📈 So sánh Doanh số Thực tế vs Dự đoán")
num_samples = st.slider("🔢 Chọn số mẫu hiển thị", min_value=10, max_value=min(100, len(y_test)), value=30, step=5)

fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(range(num_samples), y_test.values[:num_samples], label='Thực tế', marker='o', linestyle='-', color='blue')
ax5.plot(range(num_samples), y_pred[:num_samples], label='Dự đoán', marker='x', linestyle='--', color='orange')
ax5.set_title(f"📉 Thực tế vs Dự đoán ({num_samples} mẫu đầu)")
ax5.set_xlabel("Chỉ số mẫu")
ax5.set_ylabel("Sales")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# --- Biểu đồ sai số ---
st.subheader("📉 Phân phối sai số dự đoán")
errors = y_test - y_pred
fig6, ax6 = plt.subplots(figsize=(10, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax6, color='salmon')
ax6.set_title("Phân phối sai số")
ax6.set_xlabel("Sai số")
ax6.set_ylabel("Tần suất")
st.pyplot(fig6)
