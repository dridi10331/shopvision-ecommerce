import streamlit as st
import os

st.set_page_config(page_title="ShopVision", page_icon="🛍️")

st.title("🛍️ ShopVision")
st.write("E-commerce with AI Object Detection")

# Test environment variables
st.subheader("Environment Check")
if os.environ.get("SUPABASE_URL"):
    st.success("✅ SUPABASE_URL configured")
else:
    st.error("❌ SUPABASE_URL missing")

if os.environ.get("SUPABASE_KEY"):
    st.success("✅ SUPABASE_KEY configured")
else:
    st.error("❌ SUPABASE_KEY missing")

if os.environ.get("GEMINI_API_KEY"):
    st.success("✅ GEMINI_API_KEY configured")
else:
    st.error("❌ GEMINI_API_KEY missing")

st.info("If all checks pass, the Space is working correctly!")
