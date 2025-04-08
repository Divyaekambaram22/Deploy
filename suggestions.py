import streamlit as st

def simple_feedback_form():
    st.title("📝 We Value Your Suggestions!")
    
    st.write("Your Response helps us improve. Please share your thoughts below!")
    
    name = st.text_input("👤 Your Name")
    email = st.text_input("📧 Your Email")
    rating = st.slider("⭐ How would you rate your experience?", 1, 5, 3)
    comments = st.text_area("💬 Your Comments")
    
    if st.button("Submit"):
        if name and email and comments:
            st.success("✅ Thank you, {}! Your Response has been submitted successfully.".format(name))
        else:
            st.warning("⚠️ Please fill in all fields before submitting.")

if __name__ == "__main__":
    simple_feedback_form()



