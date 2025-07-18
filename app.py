import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Weekly Sales Forecast", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sales_weekly_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Promotion'] = df['Promotion'].map({'Yes': 1, 'No': 0})
    df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})
    df['Week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

df = load_data()

# Aggregate weekly
weekly_df = df.groupby('Week').agg({
    'Sales': 'sum',
    'Promotion': 'mean',
    'Holiday': 'mean'
}).reset_index()

st.title("📊 Weekly Sales Forecasting App")

# Show raw data
with st.expander("🔍 View Raw Weekly Data"):
    st.dataframe(weekly_df)

# ------------------------ Charts ------------------------
st.subheader("📈 Data Visualization")

# Line chart: Sales over time
fig1, ax1 = plt.subplots()
sns.lineplot(data=weekly_df, x='Week', y='Sales', marker='o', ax=ax1)
plt.xticks(rotation=45)
ax1.set_title("Weekly Sales Over Time")
st.pyplot(fig1)

# Correlation heatmap
fig2, ax2 = plt.subplots()
sns.heatmap(weekly_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)

# Sales vs Promotion
fig3, ax3 = plt.subplots()
sns.scatterplot(data=weekly_df, x='Promotion', y='Sales', ax=ax3)
ax3.set_title("Sales vs Promotion")
st.pyplot(fig3)

# Sales vs Holiday
fig4, ax4 = plt.subplots()
sns.boxplot(x='Holiday', y='Sales', data=weekly_df, ax=ax4)
ax4.set_title("Sales by Holiday/Non-Holiday")
ax4.set_xticklabels(['No Holiday', 'Holiday'])
st.pyplot(fig4)

# ------------------------ Modeling ------------------------
st.subheader("🤖 Predict Weekly Sales")

# Features and target
X = weekly_df[['Promotion', 'Holiday']]
y = weekly_df['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**✅ R² Score:** {r2:.2f}")
st.write(f"**📉 Mean Squared Error (MSE):** {mse:.2f}")

# Show actual vs predicted
st.markdown("#### 🔄 Actual vs Predicted Sales")
result_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
st.dataframe(result_df.reset_index(drop=True))

# ------------------------ User Input ------------------------
st.subheader("📊 Try Your Own Prediction")

promo_input = st.slider("Promotion Rate (%)", 0, 100, 50) / 100
holiday_input = st.radio("Is it a Holiday Week?", ['No', 'Yes'])
holiday_val = 1 if holiday_input == 'Yes' else 0

user_pred = model.predict([[promo_input, holiday_val]])
st.success(f"📈 Predicted Weekly Sales: **${user_pred[0]:,.2f}**")
