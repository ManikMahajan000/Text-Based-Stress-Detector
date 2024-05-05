import nltk
from flask import Flask, render_template, request, redirect, url_for
from Model.model_loader import load_model  # Assuming load_model function loads both model and vectorizer

app = Flask(__name__)

# Load your model and vectorizer
model, vectorizer, model2, model3 = load_model('Model/models')  # Modify load_model to return both model and vectorizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input feature from the form
    input_feature = request.form['input_feature']

    # Transform input feature using the vectorizer
    input_vector = vectorizer.transform([input_feature])

    # Use your model to make predictions
    prediction = model.predict(input_vector)[0]
    prediction2 = model2.predict(input_vector)[0]
    prediction3 = model3.predict(input_vector)[0]
    threshold = 0 # Adjust threshold as needed

    if prediction > threshold and prediction2 > threshold and prediction3 > threshold:
        output_text = "A Stressful Text"
    else:
        output_text = "Not A Stressful Text"

    return render_template('result.html', Ans=output_text)

@app.route('/predict_next', methods=['GET'])
def predict_next():
    return redirect(url_for('index'))  # Redirect to the index page to predict the next sentence

if __name__ == '__main__':
    app.run(debug=True)
