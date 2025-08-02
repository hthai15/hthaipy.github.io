import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("📊 Dự đoán Doanh số Bán hàng với Linear Regression")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/sales_data.csv')

df = load_data()
st.subheader("1️⃣ Tổng quan dữ liệu")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# Info
with st.expander("📌 Thông tin DataFrame"):
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

# Describe
with st.expander("📌 Thống kê mô tả"):
    st.dataframe(df.describe())

# Top 10 products
st.subheader("2️⃣ Top 10 sản phẩm bán chạy nhất")
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)

fig1, ax1 = plt.subplots()
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax1)
ax1.set_title('Top 10 sản phẩm bán chạy nhất')
ax1.set_xlabel('Product ID')
ax1.set_ylabel('Tổng Sales')
st.pyplot(fig1)

# Boxplot: promotion
st.subheader("3️⃣ Ảnh hưởng của Promotion đến Sales")
fig2, ax2 = plt.subplots()
sns.boxplot(x='promotion', y='sales', data=df, ax=ax2)
ax2.set_title('Ảnh hưởng của Promotion đến Sales')
ax2.set_xlabel('Promotion (0 = không, 1 = có)')
ax2.set_ylabel('Sales')
st.pyplot(fig2)

# Boxplot: holiday
st.subheader("4️⃣ Ảnh hưởng của Holiday đến Sales")
fig3, ax3 = plt.subplots()
sns.boxplot(x='holiday', y='sales', data=df, ax=ax3)
ax3.set_title('Ảnh hưởng của Holiday đến Sales')
ax3.set_xlabel('Holiday (0 = không, 1 = có)')
ax3.set_ylabel('Sales')
st.pyplot(fig3)

# Encode
st.subheader("5️⃣ Tiền xử lý dữ liệu & One-Hot Encoding")
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)
st.write("✅ Dữ liệu sau khi encode:")
st.dataframe(df_encoded.head())

# Train/test split
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("6️⃣ Đánh giá mô hình Linear Regression")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R²", f"{r2:.2f}")

# Plot: actual vs predicted
st.subheader("7️⃣ So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
fig4, ax4 = plt.subplots()
ax4.plot(y_test.values[:30], label='Thực tế', marker='o')
ax4.plot(y_pred[:30], label='Dự đoán', marker='x')
ax4.set_title("Thực tế vs Dự đoán")
ax4.set_xlabel("Chỉ số mẫu")
ax4.set_ylabel("Sales")
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

# Plot: error distribution
st.subheader("8️⃣ Phân phối sai số dự đoán")
errors = y_test - y_pred
fig5, ax5 = plt.subplots()
sns.histplot(errors, bins=30, kde=True, ax=ax5)
ax5.set_title("Phân phối sai số")
ax5.set_xlabel("Sai số")
ax5.set_ylabel("Tần suất")
st.pyplot(fig5)
