import gradio as gr
import joblib
import re
import string
import warnings

warnings.filterwarnings("ignore")

# --- Load Model and Vectorizer ---
try:
    final_vectorizer = joblib.load('final_bow_vectorizer.pkl')
    final_model = joblib.load('final_logistic_regression_model.pkl')
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    print("Please ensure 'final_bow_vectorizer.pkl' and 'final_logistic_regression_model.pkl' exist in the same directory.")
    final_model = None
    final_vectorizer = None


# --- Preprocessing Function with no external libraries ---
def preprocess_text_light(text):
    """
    Cleans and preprocesses text using only built-in Python libraries.
    This function removes punctuation, numbers, and excess whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Prediction Function for Gradio ---
def predict_fake_news(title, body):
    """
    Gradio function to take a title and body text, combine them,
    preprocess, and predict the label (0=Fake, 1=Real).
    """
    if final_model is None or final_vectorizer is None:
        return "Error: Model or vectorizer not loaded. Please check the file paths."

    # Combine title and body text, as per your notebook's best configuration
    combined_text = f"{title} {body}"
    
    # Preprocess the combined text using the new light function
    processed_text = preprocess_text_light(combined_text)
    
    # Vectorize the preprocessed text
    text_vectorized = final_vectorizer.transform([processed_text])
    
    # Make a prediction
    prediction = final_model.predict(text_vectorized)
    prediction_proba = final_model.predict_proba(text_vectorized)[0]
    
    if prediction[0] == 0:
        result = "FAKE"
        confidence = prediction_proba[0] * 100
    else:
        result = "REAL"
        confidence = prediction_proba[1] * 100
        
    return f"Prediction: {result} (Confidence: {confidence:.2f}%)"


# --- Gradio Interface Setup ---
iface = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter the news article's title...", label="News Title"),
        gr.Textbox(lines=15, placeholder="Paste the news article body here...", label="News Article Body")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="NLP Project: Fake News Detector",
    description="This application uses a Logistic Regression model to classify a news article as fake or real based on its title and body text. (Using lightweight preprocessing)",
    examples=[
        ["Headline: Breaking News! Man Flies To Moon Using Only A Balloon.", "The man, a local eccentric, managed to defy the laws of physics and travel to the moon."],
        ["Headline: FDA Approves New Drug for Common Cold.", "The Food and Drug Administration announced today the approval of a new medication that shortens the duration of the common cold."],
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)