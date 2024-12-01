import streamlit as st
from openai import OpenAI
import spacy
import nltk
from PIL import Image
import requests
from io import BytesIO
try:
    spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    spacy.load("en_core_web_sm")

# Download NLTK datasets (words and stopwords)
try:
    nltk.data.find('corpora/words.zip')
except:
    nltk.download("words")

try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download("stopwords")

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md or .pdf)", type=("txt", "md", "pdf")
    )



    if uploaded_file is not None:
        import tempfile
        from pydparser import ResumeParser
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name  # Get the temporary file path
        
        try:
            # Pass the temporary file path to ResumeParser
            data = ResumeParser(temp_file_path).get_extracted_data()
            # Display the extracted data
            st.write(data)
        finally:
            # Ensure the temporary file is deleted after processing
            import os
            os.remove(temp_file_path)


        # Generate an answer using the OpenAI API.
        response = client.images.generate(
        model="dall-e-3",
        prompt="a white siamese cat",
        size="1024x1024",
        quality="standard",
        n=1,
        )
        image_url = response.data[0].url

           # Fetch the image
        response = requests.get(image_url)
        if response.status_code == 200:
            # Open the image from the response content
            img = Image.open(BytesIO(response.content))
            
            # Display the image in Streamlit
            st.image(img, caption="Generated Image", use_column_width=True)
        else:
            st.error("Failed to fetch the image.")
