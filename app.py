import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# UI title
st.title("🛒 Supermarket Sales Forecasting App")
st.write("This app visualizes sales data and predicts sales using a machine learning model.")

# Upload file
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['week'] = pd.to_datetime(df['week'])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # ---- Visualization ----
    st.subheader("📊 Visualizations")

    # Chart 1: Total sales per week
    sales_by_week = df.groupby('week')['sales'].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10,4))
    sns.lineplot(data=sales_by_week, x='week', y='sales', ax=ax1)
    ax1.set_title("Total Sales by Week")
    st.pyplot(fig1)

    # Chart 2: Sales by category
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x='category', y='sales', estimator=sum, ci=None, ax=ax2)
    ax2.set_title("Total Sales by Category")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Chart 3: Correlation Heatmap
    fig3, ax3 = plt.subplots()
    corr = df[['sales', 'price', 'revenue', 'promotion', 'holiday']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # ---- Model Section ----
    st.subheader("📈 Sales Prediction (Random Forest)")

    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[['region', 'category']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['region', 'category']))

    df_model = df.drop(columns=['week', 'product_id', 'region', 'category'])
    df_final = pd.concat([df_model.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Model pipeline
    X = df_final.drop(columns='sales')
    y = df_final['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    # Scatter plot
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax4)
    ax4.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax4.set_xlabel('Actual Sales')
    ax4.set_ylabel('Predicted Sales')
    ax4.set_title("Actual vs Predicted Sales")
    st.pyplot(fig4)
else:
    st.info("👈 Please upload a CSV file to continue.")
