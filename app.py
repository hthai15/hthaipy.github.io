import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Dự đoán Doanh số Bán lẻ", layout="wide")

st.title("📈 Dự đoán Doanh số Bán hàng siêu thị")
st.write("Ứng dụng sử dụng Linear Regression để dự đoán doanh số dựa trên dữ liệu lịch sử.")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    return df

df = load_data()
st.subheader("🔍 Dữ liệu gốc")
st.dataframe(df.head())

# Tiền xử lý dữ liệu
st.subheader("🧹 Tiền xử lý dữ liệu")

# Xử lý missing values
df = df.dropna()

# Xóa duplicated rows
df = df.drop_duplicates()

# Chuyển đổi cột ngày nếu có
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Chọn các biến số
feature_cols = ['week', 'product_id', 'promotion', 'holiday']
target_col = 'sales'

X = df[feature_cols]
y = df[target_col]

# Nếu các cột chưa phải số, thì encode
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hiển thị kết quả đánh giá
st.subheader("📊 Đánh giá mô hình")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Biểu đồ thực tế vs dự đoán
fig1 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Giá trị thực tế', 'y': 'Giá trị dự đoán'})
fig1.update_layout(title="Giá trị thực tế vs Giá trị dự đoán")
st.plotly_chart(fig1)

# Vẽ biểu đồ sai số
errors = y_test - y_pred
fig2, ax = plt.subplots(figsize=(10, 4))
sns.histplot(errors, bins=30, kde=True, ax=ax)
ax.set_title("Phân phối sai số dự đoán")
ax.set_xlabel("Sai số")
ax.set_ylabel("Tần suất")
st.pyplot(fig2)

# Dự đoán mới
st.subheader("🧪 Dự đoán thử")
with st.form("predict_form"):
    week = st.number_input("Tuần", min_value=1, max_value=53, value=10)
    product_id = st.number_input("ID Sản phẩm", min_value=1, value=1)
    promotion = st.selectbox("Khuyến mãi?", [0, 1])
    holiday = st.selectbox("Ngày lễ?", [0, 1])
    submitted = st.form_submit_button("Dự đoán")

    if submitted:
        new_data = pd.DataFrame([[week, product_id, promotion, holiday]], columns=feature_cols)
        prediction = model.predict(new_data)[0]
        st.success(f"💰 Doanh số dự đoán: {prediction:.2f}")
