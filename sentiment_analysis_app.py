import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st
import numpy as np

# Set the number of words to consider as features
num_words = 10000
maxlen = 256

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Decode the word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Pad the sequences
x_train = pad_sequences(x_train, value=word_index['the'], padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, value=word_index['the'], padding='post', maxlen=maxlen)

# Define the model architecture
model = Sequential([
    Embedding(num_words, 16, input_length=maxlen),
    GlobalAveragePooling1D(),
    
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_split=0.2, epochs=5, verbose=1)

# Define a tokenizer with the same num_words as the dataset
tokenizer = Tokenizer(num_words=num_words)

# Fit the tokenizer on the training data
tokenizer.fit_on_texts([' '.join([reverse_word_index.get(i, '?') for i in review]) for review in x_train])

# Streamlit app
st.title("Sentiment Analysis on Movie Reviews")
st.write("Enter a movie review and the model will predict whether the sentiment is positive or negative.")

# Input text box
user_input = st.text_area("Enter your review here:")

# Predict sentiment
if st.button("Predict Sentiment"):
    if user_input:
        # Convert the custom review to sequences
        custom_review_sequence = tokenizer.texts_to_sequences([user_input])

        # Pad the sequences
        custom_review_padded = pad_sequences(custom_review_sequence, maxlen=maxlen, padding='post')

        # Make prediction
        custom_prediction = model.predict(custom_review_padded)
        custom_predict = custom_prediction[0]

        # Show the result
        class_names = ['Negative', 'Positive']
        result = class_names[int(np.squeeze(custom_predict) > 0.5)]
        confidence = custom_predict[0] if result == 'Positive' else 1 - custom_predict[0]
        st.write(f"Sentiment: **{result}**")
        st.write(f"Prediction Confidence: **{confidence:.2f}**")
    else:
        st.write("Please enter a review to get a prediction.")
