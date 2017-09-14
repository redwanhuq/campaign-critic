from flask import render_template
from flaskexample import app
from flask import request
import feature_engineering
from sklearn.externals import joblib
from math import floor

@app.route('/')
@app.route('/index')
def index():
    return render_template('input.html')

@app.route('/input')
def critic_input():
    return render_template('input.html')

@app.route('/output')
def critic_output():
    # Pull 'hyperlink' from the input field and store it
    hyperlink = request.args.get('hyperlink')

    # Load model and scaler from pickle files
    final_clf = joblib.load('trained_classifier.pkl')
    scaler = joblib.load('trained_scaler.pkl')

    # Set the computed features and standardize them
    computed_features = feature_engineering.process_project(hyperlink)
    scaled_features = scaler.transform([computed_features])
    
    # Display the probability of being funded with the given features
    prob = floor(100 * final_clf.predict_proba(scaled_features)[0, 1])

    return render_template('output.html', the_result = prob)