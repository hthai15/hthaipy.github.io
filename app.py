import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit setup
st.set_page_config(layout="wide")
st.title("📊 Dự báo Doanh số Siêu thị bằng Linear Regression")

# Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("data/supermarket_sales.csv")
    return df

df = load_data()

# Thông tin dữ liệu
st.subheader("1. Thông tin dữ liệu")
st.write(df.info())
st.write(df.describe())

# Trực quan hóa
st.subheader("2. Trực quan hóa dữ liệu")

# Doanh số theo tuần
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='week', y='sales', color='green', ax=ax1)
ax1.set_title('Xu hướng doanh số theo tuần')
ax1.set_xlabel('Tuần')
ax1.set_ylabel('Doanh số')
ax1.grid(True)
st.pyplot(fig1)

# Top sản phẩm bán chạy
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax2)
ax2.set_title('Top 10 sản phẩm bán chạy nhất')
ax2.set_xlabel('Product ID')
ax2.set_ylabel('Tổng Sales')
st.pyplot(fig2)

# Promotion vs Sales
fig3, ax3 = plt.subplots(figsize=(7, 5))
sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
ax3.set_title('Ảnh hưởng của Promotion đến Sales')
ax3.set_xlabel('Promotion (0 = không, 1 = có)')
ax3.set_ylabel('Sales')
st.pyplot(fig3)

# Holiday vs Sales
fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.boxplot(x='holiday', y='sales', data=df, ax=ax4)
ax4.set_title('Ảnh hưởng của Holiday đến Sales')
ax4.set_xlabel('Holiday (0 = không, 1 = có)')
ax4.set_ylabel('Sales')
st.pyplot(fig4)

# Mã hóa dữ liệu
df_encoded = pd.get_dummies(df, columns=['region', 'category', 'product_id'], drop_first=True)

# Tách X và y
X = df_encoded.drop(['sales', 'week'], axis=1)
y = df_encoded['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("3. Đánh giá mô hình Linear Regression")
st.markdown(f"""
- 📌 **Mean Absolute Error (MAE)**: `{mae:.2f}`
- 📌 **Mean Squared Error (MSE)**: `{mse:.2f}`
- 📌 **Root Mean Squared Error (RMSE)**: `{rmse:.2f}`
- 📌 **R-squared (R²)**: `{r2:.2f}`
""")

# So sánh kết quả
st.subheader("4. So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.plot(y_test.values[:30], label='Thực tế', marker='o')
ax5.plot(y_pred[:30], label='Dự đoán', marker='x')
ax5.set_xlabel("Chỉ số mẫu")
ax5.set_ylabel("Sales")
ax5.set_title("Thực tế vs Dự đoán")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# Biểu đồ phân phối sai số
errors = y_test - y_pred
st.subheader("5. Phân phối Sai số Dự đoán")
fig6, ax6 = plt.subplots(figsize=(10, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax6)
ax6.set_xlabel("Sai số")
ax6.set_ylabel("Tần suất")
ax6.set_title("Phân phối sai số")
ax6.grid(True)
st.pyplot(fig6)
