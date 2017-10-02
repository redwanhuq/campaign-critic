# CampaignCritic
![Landing page](landing_page.PNG)

## What is CampaignCritic?
A free-to-use web app that helps Kickstarter creators increase their chances of reaching their campaign's funding goal by analyzing the project description, its structure and content.

## How do you use CampaignCritic?
If you've recently launched a Kickstarter, visit the [app](http://campaigncritic.com/) and add the URL to your project. If you're still drafting your campaign, the preview URL should work too. After you submit your project link, you'll receive 3 pieces of information:
- The probability that your project will reach its funding goal
- How to improve your chances by tweaking 6 structural components of your project's description
- How to improve your chances by adding specific language or topics that are highly predictive of successfully funded projects

## How does CampaignCritic work?
When you submit a link to your Kickstarter project, CampaignCritic:
1. Analyzes the structure and content of your project's description using natural language processing
2. Feeds the analysis into a machine learning model that has learned from having combed through over 24,000 past Kickstarters
3. Reports the probability of reaching your funding goal based on the structure and content of your project's description
4. Computes weighted scores reflecting how effectively your campaign utilizes 6 structural components that were found to be most predictive of funded projects
5. Compares your project's scores against the top 5% of projects from the past five years

For those still curious about how I built CampaignCritic, check out these [slides](https://docs.google.com/presentation/d/e/2PACX-1vQtciH4cJu_f81dpL2XCjvI-39WRlAomIqf2dfXUNlgI1wGre2Qj_e-tBWVR5GShQeFeFQL_idfM4Nj/pub?start=false&loop=false&delayms=3000)

## Why did I build CampaignCritic?
- I want to help Kickstarter creators maximize their chances of reaching their funding goal. Too many innovative projects are crippled by a lackluster project description.
- I'm a big fan of the crowdfunding world&mdash;it enables everyday folks to become entrepreneurs and sometimes even gives rise to a revolutionary new genre of [products](https://www.pebble.com/).
- I love natural language processing and want to use it to build tools that enable machines to decode and interpret the most unstructured data of all&mdash;human language. By doing so, perhaps we'll someday teach a machine to design Kickstarter campaigns, or even business plans for new companies as well.

## What does this repo include?
- Jupyter notebooks that illustrate every aspect of building this project with complete documentation. Each notebook can be run to generate data and content needed for the subsquent notebook.
   0. Extracting Kickstarter URLs from the Web Robots database
   1. Building a web scraper to build a dataset of Kickstarter projects
   2. Performing feature engineering on the Kickstarter dataset
   3. Uploading the complete training set into a PostgreSQL database
   4. Visually exploring meta features and their correlations with one another and class
   5. Training a validated machine learning model on meta features and n-grams
   6. Building a bar graph that identifies how effectively a project utilized key structural components that were most predictive of funded projects
- Jupyter notebooks used for prototyping (found in the [prototyping](https://github.com/redwanhuq/campaign-critic/tree/master/prototyping) folder and fully documented) the following:
   1. Designing an efficient web scraper to collect Kickstarter projects
   2. Constructing a feature engineering strategy for meta features
   3. Researching, testing and validating machine learning models
- Web app based on Bootstrap (frontend) and Flask (backend), containing the trained model, scaler, and vectorizer (found in the [application](https://github.com/redwanhuq/campaign-critic/tree/master/application) folder).
