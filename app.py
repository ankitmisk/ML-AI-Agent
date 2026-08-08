import streamlit as st

st.sidebar.image(
    "logo.jpg",
    width=180
)

st.sidebar.title("My Dashboard")

st.set_page_config(layout="wide")

st.sidebar.title("Login")

username = st.sidebar.text_input("Username")

password = st.sidebar.text_input(
    "Password",
    type="password"
)

if st.sidebar.button("Login"):
    st.success(f"Welcome {username}")


# ---------- Navbar ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("🏠 Home")

with col2:
    st.button("📊 Dashboard")

with col3:
    st.button("📈 Analytics")

with col4:
    st.button("📞 Contact")

# ---------- Sidebar ----------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Home",
        "EDA",
        "Charts",
        "ML Model"
    ]
)

# ---------- Main Content ----------
st.title(page)

if page == "Home":
    st.write("Welcome")

elif page == "EDA":
    st.write("Dataset Analysis")

elif page == "Charts":
    st.write("Visualization")

elif page == "ML Model":
    st.write("Prediction Model")

# ---------- Footer ----------
st.write("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)

st.markdown("""
<style>

.footer{
position:fixed;
left:0;
bottom:0;
width:100%;
background:#262730;
color:white;
text-align:center;
padding:10px;
}

</style>

<div class="footer">

© 2026 Your Company | Built using Streamlit

</div>

""", unsafe_allow_html=True)
