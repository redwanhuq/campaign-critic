from flask import render_template
from flaskexample import app
from flask import request
import feature_engineering
from sklearn.externals import joblib
from math import floor
from scipy import sparse

@app.route('/')
@app.route('/index')
def index():
    return render_template('input.html')

@app.route('/input')
def critic_input():
    return render_template('input.html')

@app.route('/output')
def critic_output():
    # Pull model choice from the input field and store it   
    model_choice = request.args.get('model_choice')

    # Pull 'hyperlink' from the input field and store it
    hyperlink = request.args.get('hyperlink')

    # Load model, vectorizer and scaler from pickle files
    scaler = joblib.load('trained_scaler.pkl')
    
    # Set the computed features and standardize them
    meta_features_s1, preprocessed_text = feature_engineering.process_project(
        hyperlink, model_choice
        )

    scaled_features = scaler.transform([meta_features_s1])
    
    if model_choice in 'Yy':
        final_clf = joblib.load('trained_classifier.pkl')
        vectorizer = joblib.load('vectorizer_250.pkl')
        X_ngrams = vectorizer.transform([preprocessed_text])
        X_std_sparse = sparse.csr_matrix(scaled_features)
        X_full = sparse.hstack([X_std_sparse, X_ngrams])

        # Display the probability of being funded with the given features
        prob = floor(100 * final_clf.predict_proba(X_full)[0, 1])
    else:
        final_clf2 = joblib.load('trained_classifier_meta_only.pkl')
        prob = floor(100 * final_clf2.predict_proba(scaled_features)[0, 1])

    return render_template('output.html', the_result = prob)