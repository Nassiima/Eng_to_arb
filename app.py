import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load pre-trained model and tokenizer
model_name = "ahmed792002/Finetuning_MBart_English_Arabic_Translation"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Streamlit App
st.title("English to Arabic Translation")
st.write("Enter text in English to translate it to Arabic:")

# Input box for English text
english_text = st.text_area("Enter English Text")

# Translate the text when the button is clicked
if st.button("Translate"):
    if english_text:
        # Tokenize the input
        inputs = tokenizer(english_text, return_tensors="pt", padding=True)
        
        # Generate translation
        translated = model.generate(**inputs)
        
        # Decode the translated text
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Display the translated text
        st.write(f"Translated text: {translated_text}")
    else:
        st.write("Please enter some English text to translate.")
