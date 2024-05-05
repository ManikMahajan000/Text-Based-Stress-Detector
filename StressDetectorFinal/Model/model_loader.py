import os
import joblib

def load_model(folder_path):
    """
    Load a machine learning model and vectorizer from files within a folder.

    Parameters:
    - folder_path (str): Path to the folder containing the model and vectorizer files.

    Returns:
    - model: Loaded machine learning model.
    - vectorizer: Loaded vectorizer for preprocessing input features.
    """
    # Construct file paths for the model and vectorizer within the folder
    model_file = os.path.join(folder_path, 'model.pkl')
    model_file2 = os.path.join(folder_path, 'model2.pkl')
    model_file3 = os.path.join(folder_path, 'model3.pkl')
    vectorizer_file = os.path.join(folder_path, 'vectorizer.pkl')

    # Load the model and vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    model2 = joblib.load(model_file2)
    model3 = joblib.load(model_file3)
    return model, vectorizer, model2, model3


