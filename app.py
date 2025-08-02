import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cài đặt cấu hình
st.set_page_config(page_title="Supermarket Sales Forecast", layout="wide")

# Tiêu đề
st.title("🛒 Dự báo doanh số siêu thị & Trực quan hóa dữ liệu")

# Đọc dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    return df

df = load_data()

# ===========================
# 📌 Mô tả dữ liệu ban đầu
# ===========================
st.header("📄 Mô tả dữ liệu")
st.write("**Số dòng:**", df.shape[0])
st.write("**Số cột:**", df.shape[1])
st.write("**Các cột trong dữ liệu:**", list(df.columns))
st.dataframe(df.head())

# ===========================
# 🔧 Tiền xử lý dữ liệu
# ===========================
st.header("🔧 Tiền xử lý dữ liệu")
st.write("**Kiểm tra giá trị null:**")
st.dataframe(df.isnull().sum())

st.write("**Thông tin tổng quan về dữ liệu:**")
st.dataframe(df.describe())

# ===========================
# 📊 Phân tích dữ liệu
# ===========================
st.header("📊 Phân tích dữ liệu")

# Tổng doanh số
total_sales = df['sales'].sum()
st.metric("Tổng doanh số", f"{total_sales:,.0f}")

# Doanh số trung bình theo tuần
avg_weekly_sales = df.groupby('week')['sales'].sum().mean()
st.metric("Doanh số trung bình theo tuần", f"{avg_weekly_sales:,.0f}")

# Ảnh hưởng khuyến mãi
promo_sales = df[df['promotion'] == 1]['sales'].mean()
no_promo_sales = df[df['promotion'] == 0]['sales'].mean()
st.write(f"✅ Doanh số trung bình có khuyến mãi: **{promo_sales:,.0f}**, không khuyến mãi: **{no_promo_sales:,.0f}**")

# ===========================
# 📊 Trực quan hóa dữ liệu
# ===========================
st.header("📊 Trực quan hóa dữ liệu")

# 1. Biểu đồ phân phối doanh số
st.subheader("1. Phân phối doanh số")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df['sales'], bins=30, kde=True, color='skyblue', ax=ax1)
ax1.set_title('Phân phối doanh số')
ax1.set_xlabel('Doanh số')
ax1.set_ylabel('Tần suất')
st.pyplot(fig1)

# 2. Top 10 sản phẩm bán chạy
st.subheader("2. Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy')
ax2.set_xlabel('Tổng doanh số')
ax2.set_ylabel('Mã sản phẩm')
st.pyplot(fig2)

# 3. Ảnh hưởng của Promotion đến Sales
st.subheader("3. Doanh số theo chương trình khuyến mãi")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='promotion', y='sales', data=df, palette='Set2', ax=ax3)
ax3.set_title('Doanh số theo khuyến mãi')
ax3.set_xlabel('Khuyến mãi')
ax3.set_ylabel('Doanh số')
st.pyplot(fig3)

# 4. Ảnh hưởng của Holiday đến Sales
st.subheader("4. Doanh số theo ngày lễ")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='holiday', y='sales', data=df, palette='coolwarm', ax=ax4)
ax4.set_title('Doanh số theo ngày lễ')
ax4.set_xlabel('Ngày lễ')
ax4.set_ylabel('Doanh số')
st.pyplot(fig4)

# 5. Xu hướng doanh số theo tuần
st.subheader("5. Xu hướng doanh số theo tuần")
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax5)
ax5.set_title('Xu hướng doanh số theo tuần')
ax5.set_xlabel('Tuần')
ax5.set_ylabel('Doanh số')
ax5.grid(True)
st.pyplot(fig5)

# ===========================
# 📈 Dự báo doanh số
# ===========================
st.header("📈 Dự báo doanh số")

# Chuẩn bị dữ liệu
X = df[['week']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**MSE:** {mse:,.2f}")
st.write(f"**R2 Score:** {r2:.2f}")

# Biểu đồ dự báo
st.subheader("Biểu đồ dự báo doanh số")
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.scatter(X_test, y_test, label='Thực tế', color='blue')
ax6.plot(X_test, y_pred, label='Dự báo', color='red')
ax6.set_title('Dự báo doanh số theo tuần')
ax6.set_xlabel('Tuần')
ax6.set_ylabel('Doanh số')
ax6.legend()
st.pyplot(fig6)
