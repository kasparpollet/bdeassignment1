from math import log
import pandas as pd
import numpy as np
import pickle
# import plotly.express as px
from dotenv import load_dotenv
from scipy.sparse.sputils import matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from scrapper import Scrapper
from database import DataBase
from clean import Clean
from model import Model


def get_reviews_from_file():
    """
    Get and clean positive and negative reviews from a kaggle csv data set
    And returns them as a pandas dataframe
    """
    hotels = pd.read_csv('reviews/Hotel_Reviews.csv')

    positive_reviews = (
        hotels[['Positive_Review']].copy()
        .rename({"Positive_Review": "Review"}, axis="columns")
    )

    positive_reviews['Label'] = 'Positive'
    # drop rows that contain any value in the list
    values = ['No Positive', '']
    positive_reviews = positive_reviews[positive_reviews['Review'].isin(values) == False]
        
    negative_reviews = (
        hotels[['Negative_Review']].copy()
        .rename({"Negative_Review": "Review"}, axis="columns")
    )    
    negative_reviews['Label'] = 'Negative'

    # drop rows that contain any value in the list
    values = ['No Negative', 'Nothing', '']
    negative_reviews = negative_reviews[negative_reviews['Review'].isin(values) == False]

    # Combine the 2 dataframes  
    reviews = pd.concat([positive_reviews, negative_reviews])

    return reviews

def get_written_reviews():
    """
    Get reviews into a pandas dataframe from a self written reviews csv file
    """
    return pd.read_csv('reviews/written_reviews.csv', sep=';')

def get_scraped_reviews():
    """
    Webscrape data from hotel review websites: trustpilot and hostelworld 
    And returns them as a pandas dataframe
    """
    url = 'https://www.trustpilot.com/search?query=hotels'
    reviews = Scrapper(url).get_reviews()

    # TODO get urls from file
    hostel_url = 'https://www.hostelworld.com/pwa/hosteldetails.php/St-Christopher-s-Village/London/502?from=2021-09-21&to=2021-09-24&guests=1&display=reviews'
    hostel_reviews = Scrapper(hostel_url).hostel_world()

    df = pd.DataFrame.from_dict(reviews)
    df2 = pd.DataFrame.from_dict(hostel_reviews)
    return pd.concat([df, df2])

def get_data():
    """
    Get and combine the different sources of hotel reviews
    And returns it as a combined pandas dataframe
    """
    # Get all dataframes
    data_set_reviews = get_reviews_from_file()
    written_reviews = get_written_reviews()
    scraped_data = get_scraped_reviews()
    print('scraped data: \n', scraped_data.count())
    print('written data: \n', written_reviews.count())
    print('kaggle data: \n', data_set_reviews.count())

    # Combine dataframes
    reviews = pd.concat([data_set_reviews, written_reviews, scraped_data])

    # Shuffle and reindex final dataframe
    reviews = reviews.sample(frac=1)
    reviews.reset_index(inplace=True, drop=True)

    return reviews

def logstic_regression(df):
    model = Model(df, model=LogisticRegression(solver='liblinear'), k_fold=5, vec='tfid')

    best_params = model.grid_search({
        'model__penalty': ['l1','l2'], 
        'model__C': np.logspace(-4, 4, 50)
    })

    C = best_params.best_estimator_.get_params()['model__C']
    penalty = best_params.best_estimator_.get_params()['model__penalty']
    print(f'Best C:', C)
    print(f'Best Penalty:', penalty)
    model.model.C = C
    model.model.penalty = penalty

    logistic = model.create_model()

    return logistic

def random_forest(df):
    model = Model(df, model=RandomForestClassifier(), k_fold=5, vec='tfid')

    best_params = model.grid_search({
        'model__n_estimators': list(range(30,121,10)), 
        'model__max_features': list(range(3,19,5))
    })

    n_estimators = best_params.best_estimator_.get_params()['model__n_estimators']
    max_features = best_params.best_estimator_.get_params()['model__max_features']
    print(f'Best n_estimators:', n_estimators)
    print(f'Best max_features:', max_features)
    model.model.n_estimators = n_estimators
    model.model.max_features = max_features

    forest = model.create_model()

    return forest

def __init__():
    # Load passwords and sensitive data into the environment
    load_dotenv()


if __name__ == "__main__":
    __init__()

    # Create a database connection
    db = DataBase()

    # df = pd.read_csv('reviews/cleaned_reviews.csv')
    df = db.get_filtered_from_db()

    # df = df.iloc[:2000]

    clean = Clean(df)
    print(clean.df)
    print(clean.df['Label'].value_counts())

    # clean.df.to_csv('reviews/cleaned_reviews.csv', index=False)

    # model = Model(clean.df, model=KNeighborsClassifier(n_neighbors=11))
    # model = Model(clean.df, model=MultinomialNB(), vec='tfid')

    try:
        forest = random_forest((clean.df))
        pickle.dump(forest, open('models/forest_model_gridsearch.pickle', 'wb'))
    except Exception as e:
        print(e)

    try:
        pass
        logistic = logstic_regression(clean.df)
        pickle.dump(logistic, open('models/logitic_model_gridsearch.pickle', 'wb'))
    except Exception as e:
        print(e)

    # clean.display_wordcloud()
