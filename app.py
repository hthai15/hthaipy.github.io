import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("📊 Dự Báo Doanh Số Siêu Thị")

# --- Đọc dữ liệu ---
df = pd.read_csv("supermarket_sales_forecast_sample.csv")

# --- Sidebar lọc dữ liệu ---
st.sidebar.header("🔎 Bộ lọc dữ liệu")

week_range = st.sidebar.slider("Chọn tuần", int(df['week'].min()), int(df['week'].max()), (int(df['week'].min()), int(df['week'].max())))
regions = st.sidebar.multiselect("Chọn khu vực", options=df['region'].unique(), default=df['region'].unique())
promo_filter = st.sidebar.selectbox("Khuyến mãi", ["Tất cả", "Có khuyến mãi", "Không khuyến mãi"])

df_filtered = df[
    (df['week'] >= week_range[0]) & 
    (df['week'] <= week_range[1]) & 
    (df['region'].isin(regions))
]

if promo_filter == "Có khuyến mãi":
    df_filtered = df_filtered[df_filtered['promotion'] == 1]
elif promo_filter == "Không khuyến mãi":
    df_filtered = df_filtered[df_filtered['promotion'] == 0]

# --- Giới thiệu & mô tả dữ liệu ---
st.subheader("📄 Giới thiệu Dữ liệu")
st.markdown("""
**Nguồn dữ liệu:** `supermarket_sales_forecast_sample.csv`

**Ý nghĩa các cột:**
- `week`: Tuần trong năm
- `region`: Khu vực bán hàng
- `product_id`: Mã sản phẩm
- `category`: Danh mục sản phẩm
- `promotion`: Có khuyến mãi (1) hoặc không (0)
- `holiday`: Tuần có ngày lễ (1) hoặc không (0)
- `sales`: Doanh số bán hàng

**Mục tiêu:** Phân tích doanh số, tìm hiểu yếu tố ảnh hưởng & xây dựng mô hình dự đoán doanh số.
""")

# --- Tổng quan dữ liệu ---
st.subheader("📊 Tổng quan dữ liệu đã lọc")
st.dataframe(df_filtered.head())

col1, col2, col3 = st.columns(3)
col1.metric("📦 Số sản phẩm", df_filtered['product_id'].nunique())
col2.metric("💰 Tổng doanh số", f"{df_filtered['sales'].sum():,.0f}")
col3.metric("📆 Số tuần", df_filtered['week'].nunique())

# --- Biểu đồ phân tích ---
st.subheader("📈 Phân tích doanh số")

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_filtered['sales'], kde=True, bins=30, color='skyblue', ax=ax1)
ax1.set_title("Phân phối Doanh số")
st.pyplot(fig1)

top_products = df_filtered.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette='viridis', ax=ax2)
ax2.set_title("Top 10 sản phẩm bán chạy nhất")
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(x='promotion', y='sales', data=df_filtered, ax=ax3)
ax3.set_title("Ảnh hưởng của Khuyến mãi đến Doanh số")
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.boxplot(x='holiday', y='sales', data=df_filtered, ax=ax4)
ax4.set_title("Ảnh hưởng của Ngày lễ đến Doanh số")
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(10, 4))
weekly_sales = df_filtered.groupby('week')['sales'].sum()
sns.lineplot(x=weekly_sales.index, y=weekly_sales.values, color='green', ax=ax5)
ax5.set_title("Xu hướng Doanh số theo Tuần")
st.pyplot(fig5)

# --- Mô hình dự báo ---
st.subheader("🤖 Mô hình dự báo doanh số")

df_model = df_filtered.copy()
df_model = pd.get_dummies(df_model, columns=['region', 'category', 'product_id'], drop_first=True)
X = df_model.drop(['sales', 'week'], axis=1)
y = df_model['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("📌 MAE", f"{mae:.2f}")
col2.metric("📌 RMSE", f"{rmse:.2f}")
col3.metric("📌 R²", f"{r2:.2f}")

# --- So sánh thực tế vs dự đoán ---
st.subheader("📊 So sánh Thực tế vs Dự đoán")

fig6, ax6 = plt.subplots(figsize=(10, 4))
ax6.plot(y_test.values[:30], label='Thực tế', marker='o')
ax6.plot(y_pred[:30], label='Dự đoán', marker='x')
ax6.legend()
ax6.set_title("So sánh 30 mẫu đầu")
st.pyplot(fig6)

# --- Hệ số hồi quy ---
st.subheader("🧠 Hệ số ảnh hưởng từ mô hình")

coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
st.dataframe(coefs.head(10).reset_index().rename(columns={'index': 'Feature', 0: 'Hệ số'}))

# --- Kết luận ---
st.subheader("📌 Kết luận & Nhận định")

most_impact = coefs.abs().idxmax()
st.markdown(f"""
✅ **Yếu tố ảnh hưởng mạnh nhất:** `{most_impact}`  
✅ **R² = {r2:.2f}** ⇒ {"Mô hình dự báo tốt." if r2 > 0.7 else "Mô hình chưa lý tưởng, cần cải tiến thêm."}

👉 Bạn có thể cải tiến bằng cách thử thêm các mô hình khác như Random Forest, XGBoost hoặc thêm feature mới (ví dụ: giá, thời tiết...).
""")
