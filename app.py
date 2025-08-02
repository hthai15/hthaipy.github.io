import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="📊 Dự đoán Doanh số Bán hàng", layout="wide")

st.title("📊 Dự đoán Doanh số Bán hàng với Linear Regression")

# Đọc dữ liệu
st.subheader("1️⃣ Tổng quan dữ liệu")
df = pd.read_csv("supermarket_sales_forecast_sample.csv")

st.markdown(f"**Shape:** {df.shape}")
st.dataframe(df.head())

st.subheader("📌 Kiểu dữ liệu từng cột")
st.write(df.dtypes)

st.subheader("📌 Thống kê mô tả")
st.write(df.describe())

# Xử lý dữ liệu
st.subheader("2️⃣ Xử lý dữ liệu")

st.markdown("👉 Xoá giá trị null và trùng lặp")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

st.markdown("👉 Chuẩn hoá cột ngày (nếu có)")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

# Trực quan hóa dữ liệu
st.subheader("3️⃣ Trực quan hóa dữ liệu")

# Biểu đồ doanh số theo sản phẩm
fig1 = px.bar(df.groupby("product_id")["sales"].sum().reset_index(),
              x="product_id", y="sales", title="Tổng doanh số theo sản phẩm")
st.plotly_chart(fig1, use_container_width=True)

# Biểu đồ doanh số theo tuần
fig2 = px.line(df, x="week", y="sales", title="Doanh số theo tuần")
st.plotly_chart(fig2, use_container_width=True)

# Biểu đồ phân phối sales
fig3 = px.histogram(df, x="sales", nbins=30, title="Phân phối doanh số")
st.plotly_chart(fig3, use_container_width=True)

# Mối quan hệ các biến với sales
st.markdown("### 🔍 Tương quan giữa các biến với doanh số")
fig4 = px.scatter_matrix(df, dimensions=["promotion", "holiday", "sales"],
                         color="promotion", title="Mối quan hệ giữa các biến")
st.plotly_chart(fig4, use_container_width=True)

# Heatmap
st.markdown("### 🔥 Ma trận tương quan")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Huấn luyện mô hình
st.subheader("4️⃣ Huấn luyện mô hình dự báo")

feature_cols = ['week', 'promotion', 'holiday']
if 'month' in df.columns:
    feature_cols.append('month')
if 'year' in df.columns:
    feature_cols.append('year')

X = df[feature_cols]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
st.subheader("5️⃣ Đánh giá mô hình")
st.markdown(f"- 📉 **MAE (Mean Absolute Error):** {mean_absolute_error(y_test, y_pred):.2f}")
st.markdown(f"- 📉 **MSE (Mean Squared Error):** {mean_squared_error(y_test, y_pred):.2f}")
st.markdown(f"- 📈 **R² Score:** {r2_score(y_test, y_pred):.2f}")

# Biểu đồ so sánh
st.markdown("### 📊 So sánh Giá trị Dự đoán và Thực tế")
df_result = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})
fig5 = px.line(df_result.reset_index(drop=True), title="Thực tế vs Dự đoán")
st.plotly_chart(fig5, use_container_width=True)

st.success("✅ Dự đoán hoàn thành!")
