# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Title ---
st.title("☕ Café Sales Prediction App")
st.write("Predict your café’s daily sales based on customer count, pricing, and marketing!")

# --- Generate synthetic training data ---
np.random.seed(42)
data = {
    'num_customers': np.random.randint(50, 200, 100),
    'avg_coffee_price': np.random.uniform(80, 150, 100),
    'avg_starter_price': np.random.uniform(120, 250, 100),
    'marketing_spend': np.random.uniform(500, 3000, 100),
    'is_weekend': np.random.choice([0, 1], size=100)
}

data['sales'] = (
    data['num_customers'] * (data['avg_coffee_price'] * 0.4 + data['avg_starter_price'] * 0.6)
    + data['marketing_spend'] * 0.3
    + data['is_weekend'] * 500
    + np.random.normal(0, 1000, 100)
)

df = pd.DataFrame(data)

# --- Train Model ---
X = df[['num_customers', 'avg_coffee_price', 'avg_starter_price', 'marketing_spend', 'is_weekend']]
y = df['sales']

model = LinearRegression()
model.fit(X, y)

# --- Sidebar Inputs ---
st.sidebar.header("Input Café Details")

num_customers = st.sidebar.slider("Number of Customers", 50, 300, 120)
avg_coffee_price = st.sidebar.slider("Average Coffee Price (₹)", 50, 200, 120)
avg_starter_price = st.sidebar.slider("Average Starter Price (₹)", 80, 300, 180)
marketing_spend = st.sidebar.slider("Marketing Spend (₹)", 0, 5000, 1500)
is_weekend = st.sidebar.selectbox("Is it Weekend?", ["No", "Yes"])

is_weekend_val = 1 if is_weekend == "Yes" else 0

# --- Prediction ---
new_data = pd.DataFrame({
    'num_customers': [num_customers],
    'avg_coffee_price': [avg_coffee_price],
    'avg_starter_price': [avg_starter_price],
    'marketing_spend': [marketing_spend],
    'is_weekend': [is_weekend_val]
})

predicted_sales = model.predict(new_data)[0]

# --- Display Output ---
st.subheader("Predicted Café Sales")
st.success(f"💰 Estimated Daily Sales: ₹ {predicted_sales:,.2f}")

# --- Optional Chart ---
st.write("### Training Data Overview")
st.line_chart(df['sales'])
