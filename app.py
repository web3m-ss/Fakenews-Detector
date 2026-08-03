import streamlit as st
import joblib
import re
st.set_page_config(
    page_title="Fake News Detector",
    layout= "centered"
)

@st.cache_resource
def load_model():
    model = joblib.load(r"Fakenews.pkl")
    tfidf = joblib.load(r"Tfidf.pkl")
    return model,tfidf
model, tfidf = load_model()
def clean_news_text(text):
    text = str(text)
    text = re.sub(r'^.*?\([Aa][Rr]|Reuters\)\s*-\s*', '', text)
    text = re.sub(r'^.*?\bReuters\b\s*-\s*', '', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 # -- UI 
st.title("Fake News Detector")
st.markdown("Fake News Detector Using TFIDF (only ENG news)")
st.divider()
with st.form("news_form"):
    news_title = st.text_input(
        "News Title",
        placeholder="Enter news title here..."
    )
    
    news_body = st.text_area(
        "Paste news full article text here...",
        height=200,
        placeholder="Paste full article text here..."
    )
    analyze_btn = st.form_submit_button("Analyze news..", use_container_width=True)

if analyze_btn:
    if not news_title.strip() and not news_body.strip():
        st.warning("⚠️ Please enter news title or article text first.")
    else:
        with st.spinner("Analyzing news content..."):
            full_text = f"{news_title} {news_body}"
            cleaned_text = clean_news_text(full_text)
            vectorized_text = tfidf.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            
            st.divider()
            if prediction == 0:
                st.success("Result: This news is a **Real news**")
                st.balloons()
            else:
                st.error("Result: This news is a **Fake news**")