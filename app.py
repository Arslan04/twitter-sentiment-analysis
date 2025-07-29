import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load the pre-trained Logistic Regression model and Vectorizer
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectori.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess the tweet
# def preprocess_tweet(tweet):
#     review = re.sub('[^a-zA-Z]', ' ', tweet)
#     review = review.lower()
#     review = review.split()
    
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    
#     return ' '.join(review)
def preprocess_tweet(tweet):
    # Remove non-alphabetical characters
    review = re.sub('[^a-zA-Z]', ' ', tweet)
    # Convert to lowercase
    review = review.lower()
    # Tokenize
    review = review.split()
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    review = [word for word in review if word not in stop_words]
    # Rejoin words into a single string
    return ' '.join(review)


# Streamlit app
st.title('Twitter Sentiment Analysis')

# Input tweet from user
tweet = st.text_input("Enter Tweet:")

# Button to trigger prediction
if st.button("Predict Sentiment"):
    if tweet:
        # Preprocess the tweet
        processed_tweet = preprocess_tweet(tweet)
        st.write("Tweet",tweet)
        
        # Transform the tweet using the same vectorizer used during training
        tweet_transformed = vectorizer.transform([processed_tweet])
        # st.write("processed tweet",processed_tweet)
        
        # Predict the sentiment using the Logistic Regression model
        prediction = model.predict(tweet_transformed)
        # st.write("tweet transformed",tweet_transformed)
        # st.write(prediction)
        
        # Display the result
        if prediction == 1:
            st.write("Prediction: Positive")
        else:
            st.write("Prediction: Negative")
    else:
        st.write("Please enter a tweet to predict sentiment.")
