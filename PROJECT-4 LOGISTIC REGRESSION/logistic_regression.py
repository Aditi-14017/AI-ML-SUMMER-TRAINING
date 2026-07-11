import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Page Configuration

st.set_page_config(
    page_title="Insurance purchase Prediction",
    page_icon="💵",
    layout="centered"
)
st.title("💵Insurance Purchase Prediction")
st.write("💵Insurance Purchase Prediction")

st.markdown("---")
st.markdown("""
 👤**Developed By:** Aditi singh

📧 Branch:**INFORMATION_TECHNOLOGY**,
    Room:**G-612**
""")

from datetime import datetime
st.write("📅", datetime.now().strftime("%d-%m-%Y"))

# 1. Load Data
# Note: Ensure "insurance_data.csv" is in the same directory as this script!
try:
    df = pd.read_csv("insurance_data.csv")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Please place 'insurance_data.csv' in the same folder as this script.")
    st.stop()

# # 2. Plotting the Data in Streamlit
# st.write("### Age vs Insurance Purchase Visualization")
# fig, ax = plt.subplots()
# ax.scatter(df.age, df.bought_insurance, marker='+', color='red')
# ax.set_xlabel("Age")
# ax.set_ylabel("Bought Insurance (1=Yes, 0=No)")
# st.pyplot(fig)

# 3. Train Test Split (Locked with random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], 
    df.bought_insurance, 
    train_size=0.8, 
    random_state=42
)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
st.write("### Model Performance")
st.write(f"**Model Accuracy Score:** {model.score(X_test, y_test):.2f}")

# Extracting dynamic coefficients so manual math always matches the model
m = model.coef_[0][0]
b = model.intercept_[0]

st.write("### Model Parameters")
st.write(f"**Coefficient (m):** {m:.4f}")
st.write(f"**Intercept (b):** {b:.4f}")

 # # 6. Manual Math & Sigmoid Function
 # def sigmoid(x):
 #     return 1 / (1 + math.exp(-x))

# # def prediction_function(age):
# z = mx + b
# z = m * age + b 
# y = sigmoid(z)
#     # return y

import streamlit as st

st.write("### Try Your Own Prediction")

# User input
user_age = st.number_input(
    "Enter Age:",
    min_value=1,
    max_value=95,
    value=24
)

# Prediction function
def prediction_function(age):
    # Example probability (replace with your ML model)
    probability = age / 90
    probability = min(max(probability, 0), 1)
    return probability

# Predict button
if st.button("Predict"):
    probability = prediction_function(user_age)

    st.write(f"**Prediction Probability:** {probability:.2f}")

    if probability >= 0.5: 
        st.success("Prediction: Age valid for insurance ✅")
        st.balloons()
    else: 
        st.error("Prediction:Age not valid for insurance   ❌")
        st.warning("👤,## you are not eligible for insurance ##.")
