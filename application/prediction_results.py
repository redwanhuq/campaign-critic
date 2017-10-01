# Load required libraries
import matplotlib as plt
plt.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Set figure display options
sns.set(context='notebook', style='darkgrid')
sns.set(font_scale=1.4)

def construct_graph(scaled_meta_features, coefficients, top_project_std):
    """Constructs and saves a plot displaying the weighted scores of the user's
    project compared to the weighted scores of the average project from the top
    5% of projects
    
    Args:
        scaled_meta_features (ndarray): a NumPy array containing the values of the
            19 meta features standardized by the scaler trained on the training
            set
        feature_ranks (ndarray): a NumPy array containing the weights of the 
            trained model
        top_project_std (ndarray): a NumPy array containing the values of the 19
            features from the average project from the top 5%, standardized by the
            scaler trained on the training set
    
    Returns:
        Nothing"""

    # List of meta features
    features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
                'num_exclms', 'percent_exclms', 'num_apple_words',
                'percent_apple_words', 'avg_words_per_sent', 'num_paragraphs',
                'avg_sents_per_paragraph', 'avg_words_per_paragraph',
                'num_images', 'num_videos', 'num_youtubes', 'num_gifs',
                'num_hyperlinks', 'num_bolded', 'percent_bolded']
    
    # Compute feature importances of the meta features
    feature_ranks = pd.Series(
        coefficients[:len(features)],
        index=features
    )
    
    # List of meta features that were most predictive of funded projects
    predictive_features = ['num_hyperlinks', 'num_images', 'num_apple_words',
                     'num_exclms', 'percent_bolded', 'num_words']

    # Transform the standardized feature vector into a Series
    feature_vector_std = pd.Series(scaled_meta_features.ravel(), index=features)

    # Compute the weighted score of the meta features of the user's project
    user_project_score = np.multiply(
        feature_vector_std[predictive_features],
        feature_ranks[predictive_features]
    )

    # Compute the weighted score of the meta features of the average top project
    top_project_score = np.multiply(
        top_project_std[predictive_features],
        feature_ranks[predictive_features]
    )

    # Combine the weighted score into a single DataFrame
    messy = pd.DataFrame(
        [user_project_score, top_project_score], 
        index=['Your project', 'Top projects']
    ).T.reset_index()

    # Transform the combined data into tidy format
    tidy = pd.melt(
        messy,
        id_vars='index',
        value_vars=['Your project', 'Top projects'],
        var_name=' '
    )

    # Draw a grouped bar plot of the weighted scores, and remove axes labels and 
    # x-axis tick marks
    fig = sns.factorplot(
        data=tidy,
        y='index',
        x='value',
        hue=' ',
        kind='bar',
        size=5,
        aspect=1.5,
        palette='Set1',
        legend_out=False
    ).set(
        xlabel='score',
        ylabel='',
        xticks=[]
    )

    # Re-label the y-axis and re-position legend
    labels = ['hyperlinks', 'images', 'innovation words', 'exclamation marks',
        'bolded text', 'length of description']
    plt.yticks(np.arange(6), labels)
    fig.ax.legend(loc='lower right')

    # Save the figure
    plt.savefig(
        'flaskexample/static/images/figure.png',
        dpi=300,
        bbox_inches='tight'
    );