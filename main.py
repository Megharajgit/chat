import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def getLLamaResponse(input_text, no_words, blog_style):
    # Load the pre-trained model/tokenizer
    model_path = "meta-llama/Llama-2-7b-chat-hf"

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return "Model loading error. Please check configuration."

    # Setup prompt
    prompt = f"Write a blog for {blog_style} job profile for a topic '{input_text}' within {no_words} words."

    # Tokenize input and generate output
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=int(no_words))

    # Decode the output and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI setup
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic", "")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words', "")

with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

submit = st.button("Generate")

# Final response
if submit:
    if input_text and no_words.isdigit():
        response = getLLamaResponse(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.error("Please ensure topic is entered and number of words is a valid integer.")