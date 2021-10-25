from os import remove
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

import nltk           
from nltk.corpus import stopwords
# nltk.download("stopwords")


class Clean:
    """
    The Clean class cleans a given dataframe with Reviews
    """

    def __init__(self, df):
        self.df = self.__clean(df)

    def __clean(self, df):
        """
        Cleans the Reviews of the given pandas dataframe
        """
        print('\nStart cleaning the dataframe...')
        start = time.time()

        stop_words = self.__get_stopwords()
        stemmer = nltk.stem.SnowballStemmer('english')
        df['Review'] = df['Review'].apply(lambda x: self.__clean_text(str(x), stop_words, stemmer))

        # Remove empty reviews
        df = df.loc[lambda x: x['Review'] != '']

        df.reset_index(inplace=True, drop=True)

        end = time.time()
        print(f'Finished cleaning the dataframe in {str(end-start)} seconds')
        return df

    def __clean_text(self, text, stop_words=[], stemmer=None):
        """
        Cleans a given text
        Remove <br>, \n and make the text lowercase
        Remove emojis
        Remove all stopwords if a stopwords list is given
        Transform all words to their stem so they are the same
        Remove all non english alphabetic charachters
        """
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        clean_text = text.replace("<br>", " ").replace("\n", " ")
        clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
        if stemmer:
            clean_text = ''.join(i + ' ' for i in [stemmer.stem(word) for word in text.lower().split() if word not in stop_words])
        else:
            clean_text = ''.join(i + ' ' for i in [word for word in text.lower().split() if word not in stop_words])
        return ''.join(filter(whitelist.__contains__, clean_text))

    def __get_stopwords(self):
        """
        Cobine nltk's and hotel reviews specific stopwords and returns these as a set
        """
        stop_words = stopwords.words('english')
        with open('./files/stop_words.txt') as f:
            stop_words += [line.split('\n')[0] for line in f.readlines()]
        return list(set(stop_words))

    def display_wordcloud(self):
        """
        Create and display a wordclooud with the cleaned dataframe
        """
        Mask = np.array(Image.open('files/hotel.png'))
        image_colors = ImageColorGenerator(Mask)

        all_text = " ".join(review for review in self.df.Review)
        wordcloud = WordCloud(background_color='white', mask=Mask).generate(all_text)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
        plt.axis("off")
        plt.show()
