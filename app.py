import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
# Load the trained model and vectorizer
model_file = 'modle.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

vectorizer = model['vectorizer']  # Extract the vectorizer from the model
classifier = model['logreg']  # Extract the classifier from the model

# Function for text preprocessing
def text_preprocess(text):
    # Make text lowercase, remove punctuation, links, square brackets, and words containing numbers
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    # Tokenize and lemmatize
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in token]

    return ' '.join(text)

# Streamlit app
st.title('Sentiment Analysis App')
st.write("Enter a review and the model will predict its sentiment.")

# Input review
review_text = st.text_area("Enter the review text", "")

# Predict button
if st.button('Predict Sentiment'):
    if review_text:
        # Preprocess the input text
        processed_text = text_preprocess(review_text)
        
        # Vectorize the input text
        text_vector = vectorizer.transform([processed_text])
        
        # Predict sentiment
        prediction = classifier.predict(text_vector)[0]
        
        # Display prediction
        st.write(f"Predicted Sentiment: **{prediction}**")
    else:
        st.write("Please enter some text to analyze.")

# Add a sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info("""
    Enter a product review in the text box and click "Predict Sentiment".
    The model will classify the review as Positive, Negative, or Neutral based on its content.
""")
