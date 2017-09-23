# Load required packages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Set figure display options
sns.set(context='notebook', style='darkgrid')
sns.set(font_scale=1.3)

def construct_graph(scaled_meta_features, feature_ranks, top_project_std):
    
    # List of meta features
    features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
                'num_exclms', 'percent_exclms', 'num_apple_words',
                'percent_apple_words', 'avg_words_per_sent', 'num_paragraphs',
                'avg_sents_per_paragraph', 'avg_words_per_paragraph',
                'num_images', 'num_videos', 'num_youtubes', 'num_gifs',
                'num_hyperlinks', 'num_bolded', 'percent_bolded']

    # List of meta features that were most predictive of funded projects
    predictive_features = ['num_hyperlinks', 'num_images', 'num_apple_words',
                     'num_exclms', 'percent_bolded', 'num_words']

    # Transform the scaled meta features as a Series
    feature_vector_std = pd.Series(scaled_meta_features.ravel(), index=features)

    # Compute the strength of the meta features of the user's project
    user_project_strength = np.multiply(
        feature_vector_std[predictive_features],
        feature_ranks[predictive_features]
    )

    # Compute the strength of the meta features of the average top project
    top_project_strength = np.multiply(
        top_project_std[predictive_features],
        feature_ranks[predictive_features]
    )

    # Combine the strength metrics into a single DataFrame
    messy = pd.DataFrame(
        [user_project_strength, top_project_strength], 
        index=['My project', 'Top 5%']
    ).T.reset_index()

    # Transform the combined data into tidy format
    tidy = pd.melt(
        messy,
        id_vars='index',
        value_vars=['My project', 'Top 5%'],
        var_name=' '
    )

    # Draw a grouped bar plot
    fig = sns.factorplot(
        data=tidy,
        y='index',
        x='value',
        hue=' ',
        kind='bar',
        size=5,
        aspect=1.5,
        palette='Set2'
    ).set(xlabel='relative strength', ylabel='');

    # Re-label the y-axis
    labels = ['# of hyperlinks', '# of images', '# of innovation words',
              '# of exclamation marks', '% of text bolded', '# of words']
    
    plt.yticks(np.arange(6), labels)
    plt.savefig('data/figure.png')