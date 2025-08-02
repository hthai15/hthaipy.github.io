import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Thiết lập giao diện seaborn
sns.set(style='whitegrid')

st.title("📊 Phân Tích & Dự Báo Doanh Số Siêu Thị")

# Load dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv("supermarket_sales_forecast_sample.csv")

df = load_data()

st.subheader("1. Tổng Quan Dữ Liệu")
st.write(df.head())

# Xử lý dữ liệu trùng lặp
df = df.drop_duplicates()

# Chuẩn hóa cột sales
scaler = StandardScaler()
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# Trực quan hóa dữ liệu
st.subheader("2. Trực Quan Hóa Dữ Liệu")

# 1. Histogram doanh số
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['sales'], kde=True, color='skyblue', bins=30, ax=ax1)
ax1.set_title('Phân phối doanh số bán hàng (Sales)')
ax1.set_xlabel('Sales')
ax1.set_ylabel('Tần suất')
st.pyplot(fig1)

# 2. Top 10 sản phẩm bán chạy
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy nhất')
ax2.set_xlabel('Product ID')
ax2.set_ylabel('Tổng Sales')
st.pyplot(fig2)

# 3. Boxplot Promotion
fig3, ax3 = plt.subplots(figsize=(7,5))
sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
ax3.set_title('Ảnh hưởng của Promotion đến Sales')
ax3.set_xlabel('Promotion (0 = không, 1 = có)')
ax3.set_ylabel('Sales')
st.pyplot(fig3)

# 4. Boxplot Holiday
fig4, ax4 = plt.subplots(figsize=(7,5))
sns.boxplot(x='holiday', y='sales', data=df, ax=ax4)
ax4.set_title('Ảnh hưởng của Holiday đến Sales')
ax4.set_xlabel('Holiday (0 = không, 1 = có)')
ax4.set_ylabel('Sales')
st.pyplot(fig4)

# 5. Lineplot theo tuần
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax5)
ax5.set_title('Xu hướng doanh số theo tuần')
ax5.set_xlabel('Tuần')
ax5.set_ylabel('Doanh số')
ax5.grid(True)
st.pyplot(fig5)

# Dự báo doanh số
st.subheader("3. Dự Báo Doanh Số")

# Mã hóa biến categorical
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)

# Tách X, y
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write("### 🔍 ĐÁNH GIÁ MÔ HÌNH DỰ BÁO")
st.write(f"📌 MAE: {mae:.2f}")
st.write(f"📌 MSE: {mse:.2f}")
st.write(f"📌 RMSE: {rmse:.2f}")
st.write(f"📌 R-squared (R²): {r2:.2f}")

# Biểu đồ dự đoán vs thực tế
st.write("### 📈 So sánh Doanh số Thực tế và Dự đoán")
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(y_test.values[:30], label='Thực tế', marker='o')
ax6.plot(y_pred[:30], label='Dự đoán', marker='x')
ax6.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
ax6.set_xlabel("Chỉ số mẫu")
ax6.set_ylabel("Sales")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)

# Biểu đồ sai số
errors = y_test - y_pred
fig7, ax7 = plt.subplots(figsize=(10, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax7)
ax7.set_title("Phân phối sai số dự đoán")
ax7.set_xlabel("Sai số")
ax7.set_ylabel("Tần suất")
ax7.grid(True)
st.pyplot(fig7)
