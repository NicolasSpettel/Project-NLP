#!/usr/bin/env python3
"""
Fake News Detection - Prediction Script
This script loads test data, makes predictions, and saves results with labels.
"""

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression 

# =============================================================================
# CONFIGURATION - UPDATE THESE WITH YOUR BEST MODEL SETTINGS
# =============================================================================

# Using CountVectorizer for text vectorization
BEST_VECTORIZER = CountVectorizer(max_features=25000, ngram_range=(1, 2))
# Using Logistic Regression as the classification model
BEST_MODEL = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
# Feature to be used: 'text', 'title', or 'text+title'
BEST_FEATURE = 'text+title'

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def clean_text(text):
    """
    Comprehensive text cleaning function.
    Removes HTML tags, converts to lowercase, removes b-prefixed strings,
    punctuation, numbers, and cleans whitespace.
    """
    # Remove HTML tags and content (style, script, comments)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove b-prefixed strings (e.g., b'hello' or b"world")
    text = re.sub(r"b['|\"](.*?)['|\"]", r'\1', text)
    text = text.lstrip('b')
    
    # Remove punctuation and numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Clean whitespace and remove single characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space and strip leading/trailing
    text = ' '.join([word for word in text.split() if len(word) > 1]) # Remove single character words

    #remove quotation marks like " ' ^ and so on
    text = re.sub(r'[\'\"^]', '', text)

    #remove ... aswell
    text = re.sub(r'\.\.\.', '', text)
    
    return text

def remove_stopwords(text):
    """
    Removes common English stopwords from text.
    Uses NLTK's list of stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text) # Tokenize the text into words
    return [word for word in tokens if word not in stop_words]

def lemmatize_text(tokens):
    """
    Lemmatizes tokens to their base form.
    Uses NLTK's WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text_data(df):
    """
    Applies the full text preprocessing pipeline to 'text' and 'title' columns.
    This includes cleaning, stopword removal, and lemmatization.
    Handles empty strings by replacing them with a placeholder.
    """
    for col in ['text', 'title']:
        # Apply cleaning, stopword removal, and lemmatization
        df[col] = (
            df[col]
            .apply(clean_text)
            .apply(remove_stopwords)
            .apply(lemmatize_text)
            .apply(lambda tokens: ' '.join(tokens)) # Join tokens back into a string
        )
        # Handle cases where preprocessing results in empty strings
        df[col] = df[col].replace('', f'empty_{col}')
    return df

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def train_and_predict():
    """
    Trains the configured model on training data and makes predictions on test data.
    Saves the test data with added prediction labels to a new CSV file.
    """
    
    print("Loading and preprocessing training data...")
    
    # Load training data from CSV
    try:
        train_data = pd.read_csv('train_data.csv')
        print(f"Training data loaded: {train_data.shape}")
    except FileNotFoundError:
        print("ERROR: train_data.csv not found! Please ensure it's in the same directory.")
        return
    
    # Load test data (without labels) from CSV
    try:
        test_data = pd.read_csv('test_data_no_labels.csv')
        print(f"Test data loaded: {test_data.shape}")
    except FileNotFoundError:
        print("ERROR: test_data_no_labels.csv not found! Please ensure it's in the same directory.")
        return
    
    # Preprocess both training and test datasets
    train_data = preprocess_text_data(train_data.copy())
    test_data = preprocess_text_data(test_data.copy())
    
    # Prepare text features based on BEST_FEATURE configuration
    print("Preparing text features...")
    
    if BEST_FEATURE == 'text':
        train_text = train_data['text']
        test_text = test_data['text']
    elif BEST_FEATURE == 'title':
        train_text = train_data['title']
        test_text = test_data['title']
    else:  # BEST_FEATURE == 'text+title'
        train_text = train_data['text'] + ' ' + train_data['title']
        test_text = test_data['text'] + ' ' + test_data['title']
    
    # Vectorize the prepared text data using the configured BEST_VECTORIZER
    print("Vectorizing text...")
    X_train_vec = BEST_VECTORIZER.fit_transform(train_text)
    X_test_vec = BEST_VECTORIZER.transform(test_text)
    
    # Assign vectorized features directly, as additional numerical features are no longer included
    X_train_combined = X_train_vec
    X_test_combined = X_test_vec
    
    # Train the model using the combined features and training labels
    print("Training model...")
    y_train = train_data['label']
    BEST_MODEL.fit(X_train_combined, y_train)
    
    # Make predictions on the preprocessed test data
    print("Making predictions...")
    predictions = BEST_MODEL.predict(X_test_combined)
    
    # Load the original test data again to preserve its original structure
    result_data = pd.read_csv('test_data_no_labels.csv')
    # Add the predicted labels as a new column
    result_data['label'] = predictions
    
    # Define the output file name
    output_file = 'test_data_with_labels.csv'
    # Save the results to a new CSV file without the DataFrame index
    result_data.to_csv(output_file, index=False)
    
    # Print a summary of the predictions
    print(f"\nPrediction Summary:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted Real (1): {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
    print(f"Predicted Fake (0): {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    print("="*60)
    print("FAKE NEWS DETECTION - PREDICTION SCRIPT")
    print("="*60)
    print(f"Using vectorizer: {type(BEST_VECTORIZER).__name__}")
    print(f"Using model: {type(BEST_MODEL).__name__}")
    print(f"Using feature: {BEST_FEATURE}")
    print("="*60)
    
    # --- Instructions for the user ---
    print("\n")
    print("==================================================================")
    print("                       HOW TO RUN THIS SCRIPT                     ")
    print("==================================================================")
    print("To successfully run this fake news detection script, please follow these steps:")
    print("\n")
    print("1.  **Prepare your data files:**")
    print("    - Ensure you have your training data in a CSV file named **'train_data.csv'**.")
    print("    - Ensure you have your test data (without labels) in a CSV file named **'test_data_no_labels.csv'**.")
    print("    - Both files should be in the **same directory** as this Python script.")
    print("    - The 'train_data.csv' file must contain a column named **'label'** (0 for fake, 1 for real).")
    print("    - Both 'train_data.csv' and 'test_data_no_labels.csv' must contain columns named **'text'** and **'title'**.")
    print("\n")
    print("2.  **Install necessary libraries:**")
    print("    - If you haven't already, install the required Python libraries using pip:")
    print("      `pip install pandas scikit-learn nltk`")
    print("    - You also need to download NLTK data (stopwords and wordnet). Open a Python interpreter and run:")
    print("      `import nltk`")
    print("      `nltk.download('stopwords')`")
    print("      `nltk.download('punkt')`")
    print("      `nltk.download('wordnet')`")
    print("\n")
    print("3.  **Run the script:**")
    print("    - Open your terminal or command prompt.")
    print("    - Navigate to the directory where you saved this script (e.g., `cd path/to/your/script`).")
    print("    - Execute the script using Python:")
    print("      `python predict_fake_news.py`")
    print("\n")
    print("4.  **Check the output:**")
    print("    - Upon successful execution, a new CSV file named **'test_data_with_labels.csv'** will be created in the same directory.")
    print("    - This file will contain your original test data along with a new column named **'label'**, which holds the predicted fake (0) or real (1) labels for each entry.")
    print("    - The script will also print a summary of the predictions to your terminal.")
    print("==================================================================")
    print("\n")
    # --- End of Instructions ---

    train_and_predict()
