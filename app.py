import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình Streamlit
st.set_page_config(page_title="Dự báo doanh số siêu thị", layout="wide")
st.title("📊 Phân tích và Dự báo Doanh số Siêu thị")

# Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    df = df.drop_duplicates()
    return df

df = load_data()

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# Trực quan hóa dữ liệu
st.header("📈 Trực quan hóa dữ liệu")

# 1. Phân phối doanh số
st.subheader("1. Phân phối doanh số bán hàng")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['sales'], kde=True, color='skyblue', bins=30, ax=ax1)
ax1.set_title('Phân phối doanh số bán hàng (Sales)')
ax1.set_xlabel('Sales')
ax1.set_ylabel('Tần suất')
st.pyplot(fig1)

# 2. Top 10 sản phẩm bán chạy nhất
st.subheader("2. Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy nhất')
ax2.set_xlabel('Product ID')
ax2.set_ylabel('Tổng Sales')
st.pyplot(fig2)

# 3. Ảnh hưởng của Promotion đến Sales
st.subheader("3. Ảnh hưởng của Promotion đến Sales")
fig3, ax3 = plt.subplots(figsize=(7,5))
sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
ax3.set_title('Ảnh hưởng của Promotion đến Sales')
ax3.set_xlabel('Promotion (0 = không, 1 = có)')
ax3.set_ylabel('Sales')
st.pyplot(fig3)

# 4. Ảnh hưởng của Holiday đến Sales
st.subheader("4. Ảnh hưởng của Holiday đến Sales")
fig4, ax4 = plt.subplots(figsize=(7,5))
sns.boxplot(x='holiday', y='sales', data=df, ax=ax4)
ax4.set_title('Ảnh hưởng của Holiday đến Sales')
ax4.set_xlabel('Holiday (0 = không, 1 = có)')
ax4.set_ylabel('Sales')
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

# Mô hình dự báo doanh số
st.header("🤖 Dự báo Doanh số bằng Linear Regression")

# Mã hóa One-Hot
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hiển thị đánh giá
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("🔍 Đánh giá mô hình")
st.markdown(f"📌 **MAE:** {mae:.2f}")
st.markdown(f"📌 **MSE:** {mse:.2f}")
st.markdown(f"📌 **RMSE:** {rmse:.2f}")
st.markdown(f"📌 **R² Score:** {r2:.2f}")

# Vẽ biểu đồ kết quả dự đoán
st.subheader("📊 So sánh kết quả dự đoán với thực tế")
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(y_test.values[:30], label='Thực tế', marker='o')
ax6.plot(y_pred[:30], label='Dự đoán', marker='x')
ax6.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
ax6.set_xlabel("Chỉ số mẫu")
ax6.set_ylabel("Sales")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)
