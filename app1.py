import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- LOGIN SYSTEM ----------------
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["login"] = True
        else:
            st.error("Invalid credentials")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# ---------------- MAIN APP ----------------

import streamlit as st
import time

st.set_page_config(page_title="Dashboard", layout="wide")

# Loading animation
with st.spinner("Loading Dashboard..."):
    time.sleep(2)

# Logo + Title
col1, col2 = st.columns([1,4])

with col1:
    st.image("Logo.png", width=100)

with col2:
    st.markdown("<h1 style='color:#00FFAA;'>📊 UrbanNest Studio Sales Dashboard</h1>", unsafe_allow_html=True)

# Load data
df = pd.read_csv("C:/Users/AALIMA/Documents/M.Tech_BigData/sales.csv")

# Fix date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Sidebar Filters
st.sidebar.image("logo.png", width=120)
st.sidebar.title("🔍 Filters")
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Analysis", "📈 Prediction", "🤖 Chatbot"]
)

if page == "🏠 Home":
    st.markdown("<h1 style='color:#00FFAA;'>📊 Sales Dashboard</h1>", unsafe_allow_html=True)
    st.success("Welcome to your AI-powered dashboard 🚀")

region = st.sidebar.multiselect("Region", df["Region"].unique(), df["Region"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), df["Category"].unique())

filtered_df = df[(df["Region"].isin(region)) & (df["Category"].isin(category))]

# ---------------- KPI SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <p class="big-font">Total Sales</p>
        <p>${filtered_df['Sales'].sum():,.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <p class="big-font">Total Orders</p>
        <p>{filtered_df.shape[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <p class="big-font">Top Region</p>
        <p>{filtered_df.groupby("Region")["Sales"].sum().idxmax()}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CHARTS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Sales by Region")
    st.bar_chart(filtered_df.groupby("Region")["Sales"].sum())

with col2:
    st.subheader("📦 Sales by Category")
    st.bar_chart(filtered_df.groupby("Category")["Sales"].sum())

# ---------------- PIE CHART ----------------
st.subheader("Category Distribution (Pie Chart)")
st.write(filtered_df["Category"].value_counts())
st.pyplot(filtered_df["Category"].value_counts().plot.pie(autopct='%1.1f%%').figure)

# ---------------- MONTHLY TREND ----------------
filtered_df["Month"] = filtered_df["Order Date"].dt.month
st.subheader("📅 Monthly Trend")
st.line_chart(filtered_df.groupby("Month")["Sales"].sum())

# ---------------- DOWNLOAD BUTTON ----------------
st.subheader("Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# ---------------- ML PREDICTION ----------------
st.subheader("📈 Sales Prediction (Simple ML)")

# Prepare data
monthly = filtered_df.groupby("Month")["Sales"].sum().reset_index()

if len(monthly) > 1:
    X = monthly["Month"].values.reshape(-1,1)
    y = monthly["Sales"].values

    model = LinearRegression()
    model.fit(X, y)

    next_month = st.slider("Select Month for Prediction", 1, 12, 6)

    prediction = model.predict([[next_month]])

    st.success(f"Predicted Sales for Month {next_month}: ${prediction[0]:,.2f}")
else:
    st.warning("Not enough data for prediction")

# ---------------- DATA TABLE ----------------
st.subheader("Data Preview")
st.dataframe(filtered_df.head(20))

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}

.card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

#Add Chatbot Code
from groq import Groq

client = Groq(api_key="gsk_xxxx")

# Initialize memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if page == "🤖 Chatbot":

    st.subheader("🤖 AI Assistant")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    user_input = st.chat_input("Ask about your sales data...")

    if user_input:

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        response_text = ""

        # 👉 Total Sales
        if "total sales" in user_input.lower():
            total = df["Sales"].sum()
            response_text = f"Total Sales = ${total:,.2f}"

        # 👉 Top Region
        elif "top region" in user_input.lower():
            region = df.groupby("Region")["Sales"].sum().idxmax()
            response_text = f"Top Region is {region}"

        # 👉 Category Chart
        elif "category chart" in user_input.lower():
            response_text = "Here is sales by category:"
            chart_data = df.groupby("Category")["Sales"].sum()
            st.bar_chart(chart_data)

        # 👉 Monthly Trend
        elif "monthly" in user_input.lower():
            df["Month"] = df["Order Date"].dt.month
            response_text = "Monthly Sales Trend:"
            chart_data = df.groupby("Month")["Sales"].sum()
            st.line_chart(chart_data)

        # 👉 AI response (Groq)
        else:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": user_input}]
            )
            response_text = response.choices[0].message.content

        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Show response
        with st.chat_message("assistant"):
            st.write(response_text)
