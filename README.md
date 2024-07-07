# Sentiment Analysis on Movie Reviews

This project demonstrates a sentiment analysis model that predicts whether a given movie review is positive or negative using a neural network. The model is trained on the IMDB dataset and deployed using a Streamlit app.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

## Introduction

Sentiment analysis is a natural language processing (NLP) task where we determine the sentiment or emotion expressed in a piece of text. In this project, we classify movie reviews from the IMDB dataset as either positive or negative. We use a neural network model for this task and deploy the model using a Streamlit app, where users can input their own movie reviews and get the sentiment prediction.

## Technologies Used

- Python 3
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib

## Model Architecture

The neural network model used for sentiment analysis consists of the following layers:

1. **Embedding Layer**: Converts word indices to dense vectors of fixed size.
2. **Global Average Pooling Layer**: Reduces the dimensions of the input, computing the average of all the elements.
3. **Dense Layer**: A fully connected layer with ReLU activation.
4. **Output Layer**: A fully connected layer with sigmoid activation to output a probability score.

## Data Preprocessing

### Loading the IMDB Dataset

We use the IMDB dataset provided by Keras, which contains 50,000 movie reviews labeled as positive or negative. We limit the dataset to the top 10,000 most frequent words.

### Tokenizing and Padding

We tokenize the text data into sequences of word indices and pad them to ensure uniform input size for the model. We use a maximum length of 256 words per review.

### Building the Tokenizer

We create a tokenizer that maps words to indices and fits it on the training data. This tokenizer is used to preprocess custom input reviews in the Streamlit app.

## Training the Model

The model is compiled with the Adam optimizer and binary crossentropy loss function. It is trained for 5 epochs with a validation split of 20%. The training process includes evaluating the model on a validation set to monitor its performance.

## Running the Streamlit App

To run the Streamlit app, follow these steps:

1. Ensure all dependencies are installed:
   
   pip install tensorflow streamlit numpy matplotlib

2.Save the project script as sentiment_analysis_app.py.

3.Run the Streamlit app:

  streamlit run sentiment_analysis_app.py
  
4.Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).


## How It Works

**User Input:**  
The user enters a movie review in the text area of the Streamlit app.

**Preprocessing:**  
The input text is tokenized and padded using the pre-fitted tokenizer.

**Prediction:**  
The preprocessed input is fed into the trained model to get the sentiment prediction.

**Output:**  
The sentiment (positive or negative) and the prediction confidence are displayed on the app.

## Results

The trained model achieves reasonable accuracy on the IMDB test set. The Streamlit app allows for real-time sentiment analysis of user-provided movie reviews.

## Future Improvements

- **Increase Model Complexity:** Experiment with more complex models such as LSTM or GRU for better performance.
- **Hyperparameter Tuning:** Perform hyperparameter tuning to optimize the model.
- **Data Augmentation:** Use data augmentation techniques to enhance the training dataset.
- **Deployment:** Deploy the model as a web service using cloud platforms like AWS, GCP, or Azure for wider accessibility.

## Acknowledgements

- The IMDB dataset is provided by Keras.
- The project utilizes TensorFlow and Keras for building and training the neural network model.
- Streamlit is used for creating the web application.

