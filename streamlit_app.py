import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="Support Ticket Analyzer", layout="wide")

st.title("Support Ticket Analyzer")
st.write("This app classifies customer support issues and generates a summary using Transformer models.")

# Input box
text = st.text_area("Paste customer support conversation or ticket:", height=280)

# Button
if st.button("Analyze"):
    if not text.strip():
        st.error("Please enter some ticket text.")
    else:
        with st.spinner("Analyzing.."):
            response = requests.post(API_URL, json={"text": text})

            if response.status_code != 200:
                st.error("Error calling API")
            else:
                data = response.json()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment")
                    st.write(f"**Sentiment:** {data['sentiment']}")

                with col2:
                    st.subheader("Summary")
                    st.success(data['summary'])

