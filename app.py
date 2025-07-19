import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="🛒 Sales Forecast", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    df['week'] = pd.to_datetime(df['week'])
    return df

df = load_data()

st.title("🛒 Supermarket Sales Forecast")
st.write("Analyze sales data and forecast future sales with Random Forest.")

# ======= Visualization =======
st.subheader("📊 Exploratory Data Analysis")

fig1 = plt.figure(figsize=(10, 4))
sns.lineplot(data=df.groupby("week")["sales"].sum().reset_index(), x="week", y="sales")
plt.title("Total Sales by Week")
st.pyplot(fig1)

fig2 = plt.figure()
sns.barplot(data=df, x="category", y="sales", estimator=sum, ci=None)
plt.title("Sales by Category")
plt.xticks(rotation=45)
st.pyplot(fig2)

fig3 = plt.figure()
sns.barplot(data=df, x="region", y="sales", estimator=sum, ci=None)
plt.title("Sales by Region")
st.pyplot(fig3)

fig4 = plt.figure()
sns.boxplot(data=df, x="promotion", y="sales")
plt.title("Promotion vs Sales")
st.pyplot(fig4)

fig5 = plt.figure()
sns.boxplot(data=df, x="holiday", y="sales")
plt.title("Holiday vs Sales")
st.pyplot(fig5)

fig6 = plt.figure()
sns.scatterplot(data=df, x="price", y="sales", hue="promotion")
plt.title("Price vs Sales (Promotion)")
st.pyplot(fig6)

fig7 = plt.figure()
sns.scatterplot(data=df, x="sales", y="revenue", hue="category")
plt.title("Sales vs Revenue by Category")
st.pyplot(fig7)

fig8 = plt.figure()
sns.heatmap(df[["sales", "price", "revenue", "promotion", "holiday"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
st.pyplot(fig8)

# ======= Preprocess =======
le_region = LabelEncoder()
le_category = LabelEncoder()

df['region_encoded'] = le_region.fit_transform(df['region'])
df['category_encoded'] = le_category.fit_transform(df['category'])

features = ['price', 'revenue', 'promotion', 'holiday', 'region_encoded', 'category_encoded']
X = df[features]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📈 Model Evaluation")
st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

fig9 = plt.figure()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
st.pyplot(fig9)

# ======= Predict from user input =======
st.subheader("🔮 Predict Sales with Your Input")

col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Product Price", value=20.0)
    promotion = st.selectbox("Promotion", [0, 1])
    holiday = st.selectbox("Holiday", [0, 1])
with col2:
    region = st.selectbox("Region", df['region'].unique())
    category = st.selectbox("Category", df['category'].unique())
    revenue = st.number_input("Expected Revenue", value=5000.0)

# Encode inputs
region_code = le_region.transform([region])[0]
category_code = le_category.transform([category])[0]

new_data = pd.DataFrame([[price, revenue, promotion, holiday, region_code, category_code]], columns=features)
predicted_sales = model.predict(new_data)[0]

st.success(f"📊 Predicted Sales: **{predicted_sales:.2f} units**")
