import model_training as tr
import pandas as pd
import joblib

def predict_gender(name):
    """
    Predict gender for a given name using the trained ML model.
    """
    # Normalize the name
    clean_name = tr.normalize_name(name)
    
    # Convert to vector
    name_vector = vectorizer.transform([clean_name])
    
    # Predict
    prediction = model.predict(name_vector)[0]
    
    return prediction

model = joblib.load("gender_model_enhanced.pkl")
vectorizer = joblib.load("vectorizer_enhanced.pkl")


print(predict_gender("فاطمة"))  # Example usage