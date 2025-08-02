import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cấu hình trang
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
# 📊 Trực quan hóa dữ liệu
# ===========================
st.header("📊 Trực quan hóa dữ liệu")

# 1. Phân phối doanh số
st.subheader("1. Phân phối doanh số")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df['sales'], bins=30, kde=True, color='skyblue', ax=ax1)
ax1.set_title('Phân phối doanh số')
ax1.set_xlabel('Doanh số')
ax1.set_ylabel('Tần suất')
st.pyplot(fig1)

with st.expander("📊 Phân tích"):
    st.markdown("- Doanh số phân bố lệch phải, cho thấy phần lớn các giao dịch có giá trị doanh số thấp.")

# 2. Top 10 sản phẩm bán chạy
st.subheader("2. Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy')
ax2.set_xlabel('Tổng doanh số')
ax2.set_ylabel('Mã sản phẩm')
st.pyplot(fig2)

with st.expander("📊 Phân tích"):
    st.markdown("- Một số sản phẩm có doanh số vượt trội rõ rệt, phản ánh độ phổ biến hoặc hiệu quả marketing.")

# 3. Ảnh hưởng của Promotion đến Sales
st.subheader("3. Doanh số theo chương trình khuyến mãi")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='promotion', y='sales', data=df, palette='Set2', ax=ax3)
ax3.set_title('Doanh số theo khuyến mãi')
ax3.set_xlabel('Khuyến mãi')
ax3.set_ylabel('Doanh số')
st.pyplot(fig3)

with st.expander("📊 Phân tích"):
    st.markdown("- Nhìn chung, các đơn hàng có khuyến mãi có xu hướng đạt doanh số cao hơn.")

# 4. Ảnh hưởng của Holiday đến Sales
st.subheader("4. Doanh số theo ngày lễ")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='holiday', y='sales', data=df, palette='coolwarm', ax=ax4)
ax4.set_title('Doanh số theo ngày lễ')
ax4.set_xlabel('Ngày lễ')
ax4.set_ylabel('Doanh số')
st.pyplot(fig4)

with st.expander("📊 Phân tích"):
    st.markdown("- Có sự khác biệt nhẹ về doanh số giữa ngày lễ và ngày thường, tùy thuộc vào hành vi tiêu dùng.")

# 5. Xu hướng doanh số theo tuần
st.subheader("5. Xu hướng doanh số theo tuần")
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax5)
ax5.set_title('Xu hướng doanh số theo tuần')
ax5.set_xlabel('Tuần')
ax5.set_ylabel('Doanh số')
ax5.grid(True)
st.pyplot(fig5)

with st.expander("📊 Phân tích"):
    st.markdown(
        "- Biểu đồ đường cho thấy xu hướng doanh số theo thời gian (tuần).\n"
        "- Có thể thấy các tuần có mức tăng giảm khác nhau, phản ánh ảnh hưởng của các chiến dịch marketing, ngày lễ hoặc sự biến động nhu cầu."
    )

# ===========================
# 🤖 Dự báo doanh số đơn giản
# ===========================
st.header("🤖 Dự báo doanh số với Linear Regression")

# Chuẩn bị dữ liệu
features = ['week', 'promotion', 'holiday']
X = df[features]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hiển thị kết quả
st.subheader("Hiệu suất mô hình")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Biểu đồ so sánh dự báo
st.subheader("So sánh giá trị thực tế và dự báo")
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(y_test.values[:50], label='Thực tế')
ax6.plot(y_pred[:50], label='Dự báo')
ax6.set_title('So sánh doanh số thực tế và dự báo')
ax6.set_xlabel('Đơn hàng')
ax6.set_ylabel('Doanh số')
ax6.legend()
st.pyplot(fig6)

with st.expander("📊 Nhận xét"):
    st.markdown("- Mô hình hồi quy tuyến tính có thể mô phỏng xu hướng nhưng chưa hoàn toàn chính xác với dữ liệu hiện tại.")
