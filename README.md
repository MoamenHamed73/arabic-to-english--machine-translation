# 🌍 Arabic → English Machine Translation (MarianMT)

This project is a **Neural Machine Translation (NMT)** system that translates Arabic text into English using a Transformer-based model (MarianMT).

It demonstrates a full NLP pipeline including data preprocessing, model fine-tuning, evaluation, and deployment using Hugging Face Hub.

---

## 🚀 Live Model

The trained model is available on Hugging Face:

👉 https://huggingface.co/moamehamed7/arabic-to-english-model

---

## 🧠 Model Architecture

This project is based on a **Transformer Seq2Seq architecture**:

- Encoder-Decoder structure  
- Self-Attention mechanism  
- Pretrained MarianMT model fine-tuned on Arabic–English dataset  
- Tokenization using Hugging Face Tokenizer  

---

## 📊 Evaluation Results

The model is evaluated using BLEU score:

- **BLEU Score:** ~33+

This indicates a reasonable translation quality for a fine-tuned medium-scale model.

---
## Install required libraries:

```bash
pip install transformers torch sentencepiece huggingface_hub

 from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "moamehamed7/arabic-to-english-model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "أنا أحب تعلم الذكاء الاصطناعي"

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_beams=5
)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("English Translation:", translation)

## 🛠 Tech Stack

Python 🐍
PyTorch
Hugging Face Transformers 🤗
MarianMT Model
Google Colab / VS Code

## 🎯 Project Highlights

End-to-End NLP pipeline
Fine-tuned Transformer model
Real-world machine translation system
Model deployment using Hugging Face Hub
Production-style ML project

# 👨‍💻 Author

Moamen Hamed
AI & Machine Learning Enthusiast

