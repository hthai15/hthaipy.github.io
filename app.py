import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Supermarket Sales Forecast", layout="wide")
st.title("🛒 Dự báo doanh số siêu thị & Phân tích dữ liệu")

@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    return df

df = load_data()

# ===========================
# 🧹 Mô tả và tiền xử lý dữ liệu
# ===========================
st.header("🧹 Mô tả & Tiền xử lý dữ liệu")

st.subheader("1. Thông tin tổng quan")
st.write(df.describe())

st.subheader("2. Kiểm tra dữ liệu thiếu")
st.write(df.isnull().sum())

if df['week'].dtype != 'int64' and df['week'].dtype != 'float64':
    df['week'] = pd.to_numeric(df['week'], errors='coerce')
df = df.dropna(subset=['week', 'sales'])

# ===========================
# 🔍 Phân tích dữ liệu
# ===========================
st.header("🔍 Phân tích dữ liệu")

weekly_avg = df.groupby('week')['sales'].mean().reset_index()
st.write("### Trung bình doanh số theo tuần")
st.dataframe(weekly_avg.head())

promo_sum = df.groupby('promotion')['sales'].sum().reset_index()
st.write("### Tổng doanh số theo khuyến mãi")
st.dataframe(promo_sum)

# ===========================
# 📊 Trực quan hóa dữ liệu
# ===========================
st.header("📊 Trực quan hóa dữ liệu")

st.subheader("1. Phân phối doanh số")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df['sales'], bins=30, kde=True, color='skyblue', ax=ax1)
ax1.set_title('Phân phối doanh số')
ax1.set_xlabel('Doanh số')
ax1.set_ylabel('Tần suất')
st.pyplot(fig1)

st.subheader("2. Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy')
ax2.set_xlabel('Tổng doanh số')
ax2.set_ylabel('Mã sản phẩm')
st.pyplot(fig2)

st.subheader("3. Doanh số theo khuyến mãi")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='promotion', y='sales', data=df, palette='Set2', ax=ax3)
ax3.set_title('Doanh số theo khuyến mãi')
ax3.set_xlabel('Khuyến mãi')
ax3.set_ylabel('Doanh số')
st.pyplot(fig3)

st.subheader("4. Doanh số theo ngày lễ")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='holiday', y='sales', data=df, palette='coolwarm', ax=ax4)
ax4.set_title('Doanh số theo ngày lễ')
ax4.set_xlabel('Ngày lễ')
ax4.set_ylabel('Doanh số')
st.pyplot(fig4)

st.subheader("5. Xu hướng doanh số theo tuần")
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax5)
ax5.set_title('Xu hướng doanh số theo tuần')
ax5.set_xlabel('Tuần')
ax5.set_ylabel('Doanh số')
ax5.grid(True)
st.pyplot(fig5)

# ===========================
# 📈 Dự báo doanh số bằng Linear Regression
# ===========================
st.header("📈 Dự báo doanh số")

if df.shape[0] < 2:
    st.warning("⚠️ Không đủ dữ liệu để dự báo doanh số. Vui lòng kiểm tra lại file dữ liệu.")
else:
    try:
        X = df[['week']]
        y = df['sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        df_sorted = df.sort_values('week')
        df_sorted['predicted_sales'] = model.predict(df_sorted[['week']])

        fig6, ax6 = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=df_sorted['week'], y=df_sorted['sales'], label='Thực tế', ax=ax6)
        sns.lineplot(x=df_sorted['week'], y=df_sorted['predicted_sales'], label='Dự báo', ax=ax6)
        ax6.set_title("Dự báo doanh số theo tuần")
        ax6.set_xlabel("Tuần")
        ax6.set_ylabel("Doanh số")
        ax6.legend()
        ax6.grid(True)
        st.pyplot(fig6)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi dự báo doanh số: {e}")
