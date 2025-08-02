import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình giao diện
st.set_page_config(page_title="Dự báo doanh số siêu thị", layout="wide")
st.title("📈 Dự báo Doanh số Bán hàng Siêu thị")

# 1. Đọc dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv("supermarket_sales_forecast_sample.csv")

df = load_data()
st.subheader("1. Dữ liệu đầu vào")
st.dataframe(df.head())

# 2. Tiền xử lý dữ liệu
st.subheader("2. Tiền xử lý dữ liệu")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Chuẩn hóa nếu cần
df['promotion'] = df['promotion'].astype(int)
df['holiday'] = df['holiday'].astype(int)

st.success("✅ Đã xử lý thiếu và trùng dữ liệu.")

# 3. Phân tích dữ liệu
st.subheader("3. Phân tích dữ liệu và Trực quan hóa")

# Biểu đồ 1: Doanh số theo tuần
fig1 = px.line(df, x='week', y='sales', title="Doanh số theo tuần")
st.plotly_chart(fig1, use_container_width=True)

# Biểu đồ 2: Phân phối doanh số
fig2 = px.histogram(df, x='sales', nbins=50, title="Phân phối Doanh số")
st.plotly_chart(fig2, use_container_width=True)

# Biểu đồ 3: Boxplot theo Promotion
fig3 = px.box(df, x='promotion', y='sales', title="Doanh số theo Chương trình Khuyến mãi")
st.plotly_chart(fig3, use_container_width=True)

# Biểu đồ 4: Boxplot theo Holiday
fig4 = px.box(df, x='holiday', y='sales', title="Doanh số trong và ngoài dịp lễ")
st.plotly_chart(fig4, use_container_width=True)

# Biểu đồ 5: Heatmap tương quan
st.write("Biểu đồ 5: Ma trận tương quan")
corr = df.corr(numeric_only=True)
fig5, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
st.pyplot(fig5)

# 4. Huấn luyện mô hình
st.subheader("4. Huấn luyện mô hình")

X = df[['week', 'promotion', 'holiday']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 5. Đánh giá mô hình
st.subheader("5. Đánh giá mô hình")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# 6. So sánh kết quả dự đoán
st.subheader("6. So sánh thực tế vs dự đoán (30 mẫu đầu)")

fig6, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values[:30], label='Thực tế', marker='o')
ax.plot(y_pred[:30], label='Dự đoán', marker='x')
ax.set_title("So sánh Doanh số Thực tế vs Dự đoán (30 mẫu đầu)")
ax.set_xlabel("Chỉ số mẫu")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)
st.pyplot(fig6)

# 7. Biểu đồ sai số
st.subheader("7. Phân tích sai số dự đoán")

errors = y_test - y_pred
fig7, ax = plt.subplots(figsize=(8, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax)
ax.set_title("Phân phối sai số dự đoán")
ax.set_xlabel("Sai số")
ax.set_ylabel("Tần suất")
st.pyplot(fig7)

st.success("🎉 Hoàn thành dự báo và đánh giá mô hình!")
