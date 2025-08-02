import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title
st.title("Dự đoán Doanh số Bán hàng của Siêu thị")

# Upload data
uploaded_file = st.file_uploader("📁 Tải lên file dữ liệu CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1️⃣ Tổng quan Dữ liệu")
    st.write("**Kích thước:**", df.shape)
    st.write("**Thông tin:**")
    st.dataframe(df.head())

    # Show column types
    buffer = []
    df.info(buf=buffer)
    s = "\n".join(map(str, buffer))
    st.text(s)

    st.subheader("2️⃣ Tiền xử lý Dữ liệu")

    # Kiểm tra giá trị null
    st.write("**Số lượng giá trị thiếu theo cột:**")
    st.write(df.isnull().sum())

    # Xoá hàng thiếu dữ liệu
    df = df.dropna()

    # Kiểm tra và loại bỏ bản ghi trùng
    df = df.drop_duplicates()

    # Hiển thị lại sau xử lý
    st.write("**Sau khi xử lý:**")
    st.dataframe(df.head())

    st.subheader("3️⃣ Trực quan hóa dữ liệu")

    # Biểu đồ doanh số theo tuần
    if 'week' in df.columns and 'sales' in df.columns:
        fig1 = px.line(df, x='week', y='sales', title='Doanh số theo Tuần')
        st.plotly_chart(fig1)

    # Biểu đồ phân phối doanh số
    fig2, ax = plt.subplots()
    sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
    ax.set_title("Phân phối Doanh số")
    st.pyplot(fig2)

    # Boxplot doanh số theo khuyến mãi (nếu có)
    if 'promotion' in df.columns:
        fig3, ax = plt.subplots()
        sns.boxplot(x='promotion', y='sales', data=df, ax=ax)
        ax.set_title("Doanh số theo Khuyến mãi")
        st.pyplot(fig3)

    # Biểu đồ heatmap tương quan
    fig4, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Ma trận Tương quan")
    st.pyplot(fig4)

    st.subheader("4️⃣ Huấn luyện Mô hình Dự báo")

    # Chọn đặc trưng và nhãn
    features = ['week', 'product_id', 'promotion', 'holiday']
    target = 'sales'

    if all(col in df.columns for col in features):
        X = df[features]
        y = df[target]

        # One-hot encoding nếu cần
        X = pd.get_dummies(X, drop_first=True)

        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Dự đoán
        y_pred = model.predict(X_test)

        st.success("✅ Mô hình đã được huấn luyện!")

        st.subheader("5️⃣ Đánh giá Mô hình")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R²:** {r2:.2f}")

        st.subheader("6️⃣ So sánh Kết quả")

        fig5, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values[:30], label='Thực tế', marker='o')
        ax.plot(y_pred[:30], label='Dự đoán', marker='x')
        ax.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
        ax.set_xlabel("Chỉ số mẫu")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig5)

        st.subheader("7️⃣ Phân phối Sai số")

        errors = y_test - y_pred
        fig6, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(errors, bins=30, kde=True, ax=ax)
        ax.set_title("Phân phối Sai số Dự đoán")
        ax.set_xlabel("Sai số")
        ax.set_ylabel("Tần suất")
        ax.grid(True)
        st.pyplot(fig6)
    else:
        st.warning("⚠️ Thiếu một số cột cần thiết: 'week', 'product_id', 'promotion', 'holiday'")
