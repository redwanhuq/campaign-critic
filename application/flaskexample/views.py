# Load required libraries
from flask import render_template
from flaskexample import app
from flask import request
import feature_engineering
import prediction_results
from sklearn.externals import joblib
from math import floor
from scipy import sparse
import pandas as pd
import numpy as np
from random import random

# Load scaler and vectorizer
scaler = joblib.load('trained_scaler.pkl')
vectorizer = joblib.load('vectorizer_250.pkl')

# Load model trained on meta features and n-grams
clf = joblib.load('trained_classifier.pkl')

# Load the feature vector for the average top performing project. This will be 
# used as the gold standard for weighted scores.
top_project_std = joblib.load('top_5_percent_vector.pkl')

@app.route('/')
@app.route('/index')
def index():
    """Render the index.html page
    
    Args:
        Nothing
    
    Returns:
        the index.html template and the hashed source link to the CSS"""

    # Create randomized links to prevent Flask from caching the CSS
    css_link = '../static/css/custom.css?q=' + str(random())
    return render_template('index.html', css_hash=css_link)

@app.route('/output')
def critic_output():    
    """Render the output.html page
    
    Args:
        Nothing
    
    Returns:
        the output.html template, the probability prediction, the hashed source
        link for the results graph, the hashed source link to the CSS, the URL
        of the project, the custom results response, and the custom color for
        the results response"""

    # Pull the URL from the input field and store it
    hyperlink = request.args.get('hyperlink')

    # Create randomized links to prevent Flask from caching the results image
    # and CSS
    img_link = '../static/images/figure.png?q=' + str(random())
    css_link = '../static/css/custom.css?q=' + str(random())
    
    # Scrape HTML content from hyperlink and identify if the scraping was 
    # successful
    scraped_html, success = feature_engineering.scrape(hyperlink)
    
    # Ensure input URL is from the Kickstarter domain
    if 'www.kickstarter.com' not in hyperlink:
    	success = False
    
    # If the scraping was successfully performed on a Kickstarter project page,
    # compute the probability of reaching the funding goal, otherwise lead the user
    # to an error page
    if success and (scraped_html.status_code != 404):
        # Scrape project page and extract campaign
        soup = feature_engineering.parse(scraped_html)
        campaign = feature_engineering.get_campaign(soup)

        # Extract title of the project
        title = soup.find('title').get_text(' ')
        title = title.replace(' —Kickstarter', '')

        # Normalize the text and extract meta features of the 
        # "About this project" section
        campaign['about'] = feature_engineering.normalize(campaign['about'])
        meta_features = feature_engineering.extract_meta_features(
            soup,
            campaign,
            'about'
        )
    
        # Standardize the meta features
        scaled_meta_features = scaler.transform([meta_features])
    
        # Collect the parameters weights of the trained model
        coefficients = clf.coef_.T.ravel()

        # Create a results graph and save it
        prediction_results.construct_graph(
            scaled_meta_features,
            coefficients,
            top_project_std
        )

        # Prepare the campaign text for n-gram vectorization
        preprocessed_text = feature_engineering.preprocess_text(campaign['about'])

        # Compute n-grams from the preprocessed text using the vectorizer
        X_ngrams = vectorizer.transform([preprocessed_text])

        # Combine the meta features with the n-grams
        X_meta_features = sparse.csr_matrix(scaled_meta_features)
        X_full = sparse.hstack([X_meta_features, X_ngrams])

        # Compute the probability of the project reaching the funding goal
        prob = floor(100 * clf.predict_proba(X_full)[0, 1])
        
        # Select a custom response based on the probability value and its
        # corresponding custom color 
        if prob >= 67:
            blurb = "Your campaign is in good shape!"
            color_choice = "#20C863"
        elif prob >= 33:
            blurb = "There's definitely room for improvement!"
            color_choice = "#FEB308"
        else:
            blurb = "Your campaign needs a lot of work!"
            color_choice = "#CB3335"
            
        return render_template(
            'output.html',
            the_result=prob,
            project_title=title,
            img_hash=img_link,
            css_hash=css_link,
            link=hyperlink,
            blurb=blurb,
            color_choice=color_choice
        )
    else:
        return render_template('error.html', css_hash=css_link)

@app.route('/')
@app.route('/about')
def about():
    """Render the about.html page
    
    Args:
        Nothing
    
    Returns:
        the about.html template and the hashed link to the CSS"""

    # Create randomized links to prevent Flask from caching the CSS
    css_link = '../static/css/custom.css?q=' + str(random())
    return render_template('about.html', css_hash=css_link)

@app.route('/')
@app.route('/info')
def info():
    """Render the info.html page
    
    Args:
        Nothing
    
    Returns:
        the info.html template and the hashed link to the CSS"""

    # Create randomized links to prevent Flask from caching the CSS
    css_link = '../static/css/custom.css?q=' + str(random())
    return render_template('info.html', css_hash=css_link)