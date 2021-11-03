from math import log
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# import plotly.express as px
from dotenv import load_dotenv
from scipy.sparse.sputils import matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

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
    # written_reviews = get_written_reviews()
    scraped_data = get_scraped_reviews()
    print('scraped data: \n', scraped_data.count())
    # print('written data: \n', written_reviews.count())
    print('kaggle data: \n', data_set_reviews.count())

    # Combine dataframes
    reviews = pd.concat([data_set_reviews, scraped_data])

    # Shuffle and reindex final dataframe
    reviews = reviews.sample(frac=1)
    reviews.reset_index(inplace=True, drop=True)

    return reviews

def create_basic_models(df):
    model = Model(df, model=MultinomialNB(), k_fold=5, vec='tfid')
    model.create_model()
    # pickle.dump(model.model, open('models/multinomial_model.pickle', 'wb'))
    # pickle.dump(model.vec, open('models/multinomial_vec.pickle', 'wb'))

    model = Model(df, model=LogisticRegression())
    model.create_model()
    # pickle.dump(model.model, open('models/logistic_regression_model.pickle', 'wb'))
    # pickle.dump(model.vec, open('models/logistic_regression_vec.pickle', 'wb'))

    model = Model(df, model=RandomForestClassifier())
    model.create_model()
    # pickle.dump(model.model, open('models/random_forest_model.pickle', 'wb'))
    # pickle.dump(model.vec, open('models/random_forest_vec.pickle', 'wb'))

    # model = Model(df, model=SVC(), k_fold=5)
    # model.create_model()
    # pickle.dump(model.model, open('models/knn_model.pickle', 'wb'))
    # pickle.dump(model.vec, open('models/knn_vec.pickle', 'wb'))




def logstic_regression(df):
    model = Model(df, model=LogisticRegression(), k_fold=5, vec='tfid')

    best_params = model.grid_search({
        'model__solver': ['sag', 'saga'],
        'model__penalty': ['l1','l2'], 
        'model__C': np.logspace(-2,2,5),
        # 'model__max_iter':[100, 200, 300]
    })

    solver = best_params.best_estimator_.get_params()['model__solver']
    C = best_params.best_estimator_.get_params()['model__C']
    penalty = best_params.best_estimator_.get_params()['model__penalty']
    max_iter = best_params.best_estimator_.get_params()['model__max_iter']
    print('Best solver:', solver)
    print('Best C:', C)
    print('Best Penalty:', penalty)
    print('Best max_iter:', max_iter)
    model.model.solver = solver
    model.model.C = C
    model.model.penalty = penalty
    model.model.max_iter = max_iter

    logistic = model.create_model()

    return logistic

def random_forest(df):
    model = Model(df, model=RandomForestClassifier(), k_fold=5, vec='tfid')

    best_params = model.grid_search({
        'model__n_estimators': list(range(31,121,10)), 
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

    db = DataBase()
    df = db.get_filtered_from_db()

    df['Label'] = df['Label'].apply(lambda x: 1 if x=='Positive' else 0)

    # df = df.iloc[:10000]

    clean = Clean(df)
    # print(clean.df)
    # print(clean.df['Label'].value_counts())
    # create_basic_models(clean.df)

    plt.show()
    # try:
    #     forest = random_forest((clean.df))
    #     pickle.dump(forest.model, open('models/forest_model_gridsearch.pickle', 'wb'))
    #     pickle.dump(forest.vec, open('models/forest_vec_gridsearch.pickle', 'wb'))
    # except Exception as e:
    #     print(e)

    try:
        logistic = logstic_regression(clean.df)
        # pickle.dump(logistic.model, open('models/logitic_model_gridsearch.pickle', 'wb'))
        # pickle.dump(logistic.vec, open('models/logitic_vec_gridsearch.pickle', 'wb'))
    except Exception as e:
        print(e)

    # clean.display_wordcloud()
