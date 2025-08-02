import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("Dự đoán doanh số siêu thị bán lẻ")

# --- Load dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    return df

df = load_data()
st.subheader("📊 Dữ liệu gốc")
st.dataframe(df.head(10), use_container_width=True)

# --- Tiền xử lý ---
st.subheader("🔧 Tiền xử lý dữ liệu")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

# Hiển thị mô tả dữ liệu
st.write("Mô tả dữ liệu:")
st.dataframe(df.describe())

# --- Trực quan hóa dữ liệu ---
st.subheader("📈 Trực quan hóa dữ liệu")

col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="sales", nbins=30, title="Phân phối doanh số")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if "promotion" in df.columns:
        fig2 = px.box(df, x="promotion", y="sales", title="Doanh số theo khuyến mãi")
        st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    if "holiday" in df.columns:
        fig3 = px.box(df, x="holiday", y="sales", title="Doanh số theo ngày lễ")
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "week" in df.columns:
        fig4 = px.line(df.groupby("week")["sales"].mean().reset_index(), x="week", y="sales", title="Doanh số trung bình theo tuần")
        st.plotly_chart(fig4, use_container_width=True)

# --- Chuẩn bị dữ liệu mô hình ---
st.subheader("🤖 Huấn luyện mô hình dự báo")

# Chọn các đặc trưng liên quan
features = ["week", "promotion", "holiday"]
target = "sales"

# Kiểm tra cột có tồn tại
if all(col in df.columns for col in features + [target]):
    X = df[features]
    y = df[target]

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.markdown("**🎯 Kết quả đánh giá mô hình:**")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- MSE: {mse:.2f}")
    st.write(f"- RMSE: {rmse:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    # --- Vẽ biểu đồ sai số ---
    st.subheader("📉 Phân phối sai số dự đoán")
    errors = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(errors, bins=30, kde=True, ax=ax)
    ax.set_title("Phân phối sai số dự đoán")
    ax.set_xlabel("Sai số")
    ax.set_ylabel("Tần suất")
    st.pyplot(fig)

else:
    st.warning("❌ Một số cột cần thiết không có trong file CSV. Vui lòng kiểm tra lại.")

