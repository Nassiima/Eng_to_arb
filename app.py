import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load pre-trained model and tokenizer
model_name = "ahmed792002/Finetuning_MBart_English_Arabic_Translation"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = "en_XX"  # Source language
tokenizer.tgt_lang = "ar_AR"  # Target language

# Streamlit App
st.title("English to Arabic Translation")
st.write("Enter text in English to translate it to Arabic:")

# Input box for English text
english_text = st.text_area("Enter English Text")

# Translate the text when the button is clicked
if st.button("Translate"):
    if english_text.strip():
        # Tokenize the input
        inputs = tokenizer(english_text, return_tensors="pt", padding=True, src_lang="en_XX")
        st.write(f"Tokenized inputs: {inputs}")  # Debugging log

        # Generate translation
        translated = model.generate(**inputs)
        st.write(f"Generated tokens: {translated}")  # Debugging log

        # Decode the translated text
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        # Display the translated text
        st.write(f"Translated text: {translated_text}")
    else:
        st.write("Please enter some English text to translate.")
