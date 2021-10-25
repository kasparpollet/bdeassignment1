import pandas as pd
import os
import pymysql
from sqlalchemy import create_engine

class DataBase:
    """
    The DataBase class creates a connections with a MySQL database
    And holds methodes to interact with the database
    """
    
    def __init__(self):
        print('\nCreating database connection...')
        self.engine = pymysql.connect(host='localhost',user=os.getenv('DATABASE_USERNAME'), password=os.getenv('DATABASE_PASSWORD'), database='assignment1', cursorclass=pymysql.cursors.DictCursor)
        self.put_engine = create_engine(os.getenv('DATABASE_URL'))

    def upload_data(self, df, name, error='fail'):
        """
        Upload a given pandas dataframe to the database wth a given table name
        """
        df.to_sql(name=name,con=self.put_engine,if_exists=error,index=False,chunksize=1000) 

    def get_from_db(self):
        """
        Get all reviews from the database
        """
        print('\nGetting reviews...')
        return pd.read_sql_query('CALL get_all_reviews();', self.engine)

    def get_filtered_from_db(self):
        """
        Get all usable reviews from the database
        """
        print('\nGetting reviews...')
        return pd.read_sql_query('CALL get_reviews();', self.engine)