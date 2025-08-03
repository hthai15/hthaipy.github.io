import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")
sns.set(style='whitegrid')

st.title("📊 Sales Analysis and Prediction Dashboard")

# Load dataset
def load_data():
    return pd.read_csv("supermarket_sales_forecast_sample.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Bộ lọc dữ liệu")
selected_week = st.sidebar.multiselect("Chọn tuần:", sorted(df['week'].unique()), default=sorted(df['week'].unique()))
selected_region = st.sidebar.multiselect("Chọn khu vực:", df['region'].unique(), default=df['region'].unique())
selected_promotion = st.sidebar.multiselect("Khuyến mãi:", df['promotion'].unique(), default=df['promotion'].unique())

filtered_df = df[(df['week'].isin(selected_week)) &
                 (df['region'].isin(selected_region)) &
                 (df['promotion'].isin(selected_promotion))]

# Preprocess
filtered_df = filtered_df.drop_duplicates()
scaler = StandardScaler()
filtered_df['sales_scaled'] = scaler.fit_transform(filtered_df[['sales']])

# Tabs
overview, visual, model, interpret = st.tabs(["📄 Tổng quan", "📊 Biểu đồ", "🤖 Mô hình dự báo", "🧠 Diễn giải mô hình"])

with overview:
    st.subheader("🔍 Tổng quan dữ liệu")
    st.dataframe(filtered_df.head())
    st.write("**Thông tin cột:**")
    st.markdown("""
    - `week`: Số thứ tự tuần trong năm
    - `region`: Khu vực phân phối
    - `category`: Loại sản phẩm
    - `product_id`: Mã sản phẩm
    - `sales`: Doanh số bán ra (số lượng đơn vị hoặc tiền tệ)
    - `promotion`: 1 nếu có khuyến mãi, 0 nếu không
    - `holiday`: 1 nếu là tuần có ngày lễ, 0 nếu không
    """)
    st.write("**Thông tin tổng quát:**")
    st.write("Null values:", filtered_df.isnull().sum())
    st.write("Duplicate rows:", filtered_df.duplicated().sum())
    st.write("Data types:")
    st.write(filtered_df.dtypes)
    st.write("Thống kê mô tả:")
    st.write(filtered_df.describe())

with visual:
    st.subheader("📊 Phân tích trực quan")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['sales'], kde=True, color='skyblue', bins=30, ax=ax1)
    ax1.set_title('Phân phối doanh số bán hàng')
    st.pyplot(fig1)

    top10 = filtered_df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top10.index.astype(str), y=top10.values, palette='viridis', ax=ax2)
    ax2.set_title('Top 10 sản phẩm bán chạy')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.boxplot(x='promotion', y='sales', data=filtered_df, ax=ax3)
    ax3.set_title('Ảnh hưởng của Promotion đến Sales')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(7,5))
    sns.boxplot(x='holiday', y='sales', data=filtered_df, ax=ax4)
    ax4.set_title('Ảnh hưởng của Holiday đến Sales')
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_df, x='week', y='sales', color='green', ax=ax5)
    ax5.set_title('Xu hướng doanh số theo tuần')
    st.pyplot(fig5)

with model:
    st.subheader("🤖 Mô hình dự báo doanh số")
    df_encoded = pd.get_dummies(filtered_df, columns=['region', 'category', 'product_id'], drop_first=True)
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

    st.metric("📌 MAE", f"{mae:.2f}")
    st.metric("📌 MSE", f"{mse:.2f}")
    st.metric("📌 RMSE", f"{rmse:.2f}")
    st.metric("📌 R-squared", f"{r2:.2f}")

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.plot(y_test.values[:30], label='Thực tế', marker='o')
    ax6.plot(y_pred[:30], label='Dự đoán', marker='x')
    ax6.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
    ax6.legend()
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots(figsize=(10, 4))
    errors = y_test - y_pred
    sns.histplot(errors, bins=30, kde=True, ax=ax7)
    ax7.set_title("Phân phối sai số dự đoán")
    st.pyplot(fig7)

with interpret:
    st.subheader("🧠 Diễn giải mô hình")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    st.write("📌 **Hệ số ảnh hưởng của từng biến đầu vào**")
    st.dataframe(coef_df)

    st.markdown("""
    ### 📚 Kết luận:
    - Mô hình Linear Regression giúp xác định mối liên hệ giữa các yếu tố như khuyến mãi, ngày lễ, loại sản phẩm và doanh số.
    - Chỉ số R² cho biết mô hình giải thích được khoảng **{:.2%}** phương sai trong dữ liệu.
    - Các yếu tố ảnh hưởng lớn nhất gồm: `{}`
    - Mô hình phù hợp cho việc **dự báo sơ bộ** và **hiểu nguyên nhân chính ảnh hưởng doanh số**.
    """.format(r2, ', '.join(coef_df['Feature'].head(3)))
    )
