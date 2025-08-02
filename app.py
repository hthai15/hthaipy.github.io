import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Cài đặt giao diện ---
st.set_page_config(page_title="Dự đoán Doanh số Siêu thị", layout="wide")
st.title("📊 Ứng dụng Dự đoán Doanh số Siêu thị bằng Linear Regression")

# --- Tải dữ liệu ---
st.sidebar.header("📁 Upload dữ liệu CSV")
uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Xem trước dữ liệu")
    st.write(df.head())

    # --- Tiền xử lý ---
    st.subheader("🔧 Tiền xử lý dữ liệu")
    df = df.dropna()
    df = df.drop_duplicates()
    df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)
    st.write("✅ Đã mã hóa One-Hot Encoding")

    # --- Tách dữ liệu ---
    X = df_encoded.drop(['sales', 'week'], axis=1)
    y = df_encoded['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Huấn luyện mô hình ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Đánh giá ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Kết quả đánh giá mô hình")
    st.markdown(f"""
    - **MAE**: {mae:.2f}  
    - **MSE**: {mse:.2f}  
    - **RMSE**: {rmse:.2f}  
    - **R² (R-squared)**: {r2:.2f}
    """)

    # --- Biểu đồ dự đoán vs thực tế ---
    st.subheader("📊 So sánh Doanh số Thực tế vs Dự đoán")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(y_test.values[:30], label='Thực tế', marker='o')
    ax1.plot(y_pred[:30], label='Dự đoán', marker='x')
    ax1.set_title("Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
    ax1.set_xlabel("Chỉ số mẫu")
    ax1.set_ylabel("Sales")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # --- Biểu đồ sai số ---
    st.subheader("📉 Phân phối Sai số Dự đoán")
    errors = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(errors, bins=30, kde=True, ax=ax2)
    ax2.set_title("Phân phối sai số dự đoán")
    ax2.set_xlabel("Sai số")
    ax2.set_ylabel("Tần suất")
    ax2.grid(True)
    st.pyplot(fig2)

else:
    st.warning("📌 Vui lòng upload file CSV để bắt đầu.")
