import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Arabic → English Translator",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "./final_marian_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model


tokenizer, model = load_model()


# ------------------------------
# Translation Function
# ------------------------------
def translate_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("⚙️ Project Info")
    st.markdown("""
    ### Arabic → English Translation

    This app uses a fine-tuned MarianMT model for
    Arabic to English machine translation.

    ### Model
    `Helsinki-NLP/opus-mt-ar-en`

    ### Dataset
    `OPUS-100`

    ### BLEU Score
    **33.88**

    ### Built With
    - Hugging Face Transformers
    - MarianMT
    - Streamlit
    """)


# ------------------------------
# Main UI
# ------------------------------
st.title("🌍 Arabic → English Translator")
st.caption("Professional NLP Project using MarianMT + Hugging Face")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Arabic Input")
    user_input = st.text_area(
        "Enter Arabic text",
        height=250,
        placeholder="اكتب النص العربي هنا..."
    )

    translate_button = st.button("🚀 Translate", use_container_width=True)

with col2:
    st.subheader("📘 English Translation")

    if translate_button:
        if user_input.strip() == "":
            st.warning("Please enter Arabic text first.")
        else:
            with st.spinner("Translating..."):
                translated_text = translate_text(user_input)
            st.success(translated_text)
    else:
        st.info("Translation will appear here.")


# ------------------------------
# Example Section
# ------------------------------
st.divider()
st.subheader("✨ Example Sentences")

examples = [
    "أنا أحب الذكاء الاصطناعي",
    "الجامعة مهمة جدًا في بناء المستقبل",
    "أريد تعلم معالجة اللغات الطبيعية"
]

for example in examples:
    if st.button(example):
        with st.spinner("Translating example..."):
            result = translate_text(example)
        st.write("### Result")
        st.success(result)


# ------------------------------
# Footer
# ------------------------------
st.divider()
st.markdown(
    """
    <center>
        <p style='font-size:16px;'>
            Built for Moamen hamed AI Engineer 🚀
        </p>
    </center>
    """,
    unsafe_allow_html=True
)
