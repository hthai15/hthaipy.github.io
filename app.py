import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="🛒 Supermarket Sales Forecast", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_forecast_sample.csv")
    df['week'] = pd.to_datetime(df['week'])
    return df

df = load_data()

st.title("🛒 Supermarket Sales Forecasting App")
st.write("Visualize supermarket sales and forecast values using machine learning models.")

# =======================
# 📊 Section: Visualizations
# =======================
st.subheader("📊 Exploratory Data Analysis")

# Chart 1
sales_by_week = df.groupby('week')['sales'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=sales_by_week, x='week', y='sales', ax=ax1)
ax1.set_title("1️⃣ Total Sales by Week")
st.pyplot(fig1)

# Chart 2
fig2, ax2 = plt.subplots()
sns.barplot(data=df, x='category', y='sales', estimator=sum, ci=None, ax=ax2, palette='pastel')
ax2.set_title("2️⃣ Sales by Category")
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Chart 3
fig3, ax3 = plt.subplots()
sns.barplot(data=df, x='region', y='sales', estimator=sum, ci=None, ax=ax3, palette='muted')
ax3.set_title("3️⃣ Sales by Region")
st.pyplot(fig3)

# Chart 4
fig4, ax4 = plt.subplots()
sns.boxplot(data=df, x='promotion', y='sales', ax=ax4)
ax4.set_title("4️⃣ Sales by Promotion")
st.pyplot(fig4)

# Chart 5
fig5, ax5 = plt.subplots()
sns.boxplot(data=df, x='holiday', y='sales', ax=ax5)
ax5.set_title("5️⃣ Sales by Holiday")
st.pyplot(fig5)

# Chart 6
fig6, ax6 = plt.subplots()
sns.scatterplot(data=df, x='price', y='sales', hue='promotion', alpha=0.6, ax=ax6)
ax6.set_title("6️⃣ Price vs Sales (Promotion)")
st.pyplot(fig6)

# Chart 7
fig7, ax7 = plt.subplots()
sns.scatterplot(data=df, x='sales', y='revenue', hue='category', alpha=0.7, ax=ax7)
ax7.set_title("7️⃣ Sales vs Revenue")
st.pyplot(fig7)

# Chart 8
fig8, ax8 = plt.subplots()
corr = df[['sales', 'price', 'revenue', 'promotion', 'holiday']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax8)
ax8.set_title("8️⃣ Correlation Matrix")
st.pyplot(fig8)

# ========================
# 📈 Section: Forecasting
# ========================
st.subheader("📈 Forecasting Models")

encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['region', 'category']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['region', 'category']))

df_model = df.drop(columns=['week', 'product_id', 'region', 'category'])
df_final = pd.concat([df_model.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

X = df_final.drop(columns='sales')
y = df_final['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🌲 Random Forest")
    st.write(f"**MAE:** {mean_absolute_error(y_test, rf_pred):.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
    st.write(f"**R²:** {r2_score(y_test, rf_pred):.2f}")
with col2:
    st.markdown("#### 📈 Linear Regression")
    st.write(f"**MAE:** {mean_absolute_error(y_test, lr_pred):.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
    st.write(f"**R²:** {r2_score(y_test, lr_pred):.2f}")

# Comparison charts
tab1, tab2 = st.tabs(["Random Forest", "Linear Regression"])
with tab1:
    fig_rf, ax_rf = plt.subplots()
    sns.scatterplot(x=y_test, y=rf_pred, alpha=0.6, ax=ax_rf)
    ax_rf.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax_rf.set_title("Random Forest: Actual vs Predicted")
    st.pyplot(fig_rf)

with tab2:
    fig_lr, ax_lr = plt.subplots()
    sns.scatterplot(x=y_test, y=lr_pred, alpha=0.6, ax=ax_lr)
    ax_lr.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax_lr.set_title("Linear Regression: Actual vs Predicted")
    st.pyplot(fig_lr)

# ========================
# 🔮 Predict New Input (fixed)
# ========================
st.subheader("🔮 Try Forecasting with Custom Input")

col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Product Price", min_value=0.0, value=20.0)
    promotion = st.selectbox("Promotion Applied?", [0, 1])
    holiday = st.selectbox("Is Holiday?", [0, 1])
with col2:
    region = st.selectbox("Region", df['region'].unique())
    category = st.selectbox("Category", df['category'].unique())
    revenue = st.number_input("Expected Revenue", min_value=0.0, value=5000.0)

input_data = pd.DataFrame({
    'price': [price],
    'revenue': [revenue],
    'promotion': [promotion],
    'holiday': [holiday],
    'region': [region],
    'category': [category]
})

encoded_input = encoder.transform(input_data[['region', 'category']])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['region', 'category']))
numeric_df = input_data[['price', 'revenue', 'promotion', 'holiday']].reset_index(drop=True)
final_input = pd.concat([numeric_df, encoded_df], axis=1)
final_input = final_input[X_train.columns]

model_choice = st.radio("Choose model for prediction", ["Random Forest", "Linear Regression"])
if model_choice == "Random Forest":
    prediction = rf.predict(final_input)[0]
else:
    prediction = lr.predict(final_input)[0]

st.success(f"📈 **Predicted Sales:** {prediction:.2f} units")
