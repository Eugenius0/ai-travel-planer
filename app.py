import streamlit as st

# Streamlit app layout
st.title("Simple Streamlit App")
st.subheader("Testing Streamlit Deployment")

# User input
user_input = st.text_input("Enter something:")

# Display user input
if st.button("Submit"):
    st.write("You entered:", user_input)