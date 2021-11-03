import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self, df, model, test_size=0.2, k_fold=None, vec='count'):
        self.df = df
        self.test_size = test_size
        self.k_fold = k_fold
        self.model = model
        self.vec = self.__get_vecorizer(vec)

    def __get_vecorizer(self, vec):
        if vec == 'tfid':
            return self.__tfidf_vectorize()
        elif vec == 'count':
            return self.__count_vectorize()
        else:
            return self.__count_vectorize()

    def __count_vectorize(self):
        print('\nCreating CountVectorizer...')
        vec = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_features=500)
        matrix = self.__create_matrix_df(vec)
        return matrix

    def __tfidf_vectorize(self):
        print('\nCreating TfidfVectorizer...')
        vec = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_features=500)     
        matrix = self.__create_matrix_df(vec)
        return matrix

    def __create_matrix_df(self, vec):
        wordcount = vec.fit_transform(self.df['Review'].tolist())
        tokens = vec.get_feature_names_out()
        doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wordcount)]
        return pd.DataFrame(data=wordcount.toarray(), index=doc_names, columns=tokens)

    def grid_search(self, params):
        print('\nGetting best parameters...')
        start = time.time()

        sc = StandardScaler()
        pipe = Pipeline(steps=[('sc', sc),('model', self.model)])
        grid_search = GridSearchCV(pipe, params, cv=5)
        print('YHEAAAA')
        grid_search.fit(self.vec, self.df['Label'])

        end = time.time()

        print(f'Finished getting parameters in {end - start} seconds')
        print(f'Best score: {grid_search.best_score_}')
        print(f'Best parameters: {grid_search.best_estimator_.get_params()["model"]}')

        return grid_search

    def create_model(self):
        print(f'\nCreating model: {self.model}...')

        start = time.time()
        y = self.df['Label']
        X = self.vec

        if not self.k_fold:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)

            self.model.fit(X_train, y_train)

            end = time.time()
            print(f'Finished creating {self.model} in {end - start} seconds')

            print('\nPredicting test data...')
            y_pred = self.model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            print(f'Score:', score)
            print(f'Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
            print(f'Classification Report:\n', classification_report(y_test, y_pred))

        else:
            acc_score = []
            k_fold = KFold(n_splits=self.k_fold)

            for train_index, test_index in k_fold.split(X):
                start_time = time.time()
                X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
                y_train , y_test = y[train_index] , y[test_index]
                
                self.model.fit(X_train,y_train)
                y_pred = self.model.predict(X_test)
                
                acc = accuracy_score(y_pred , y_test)
                acc_score.append(acc)
                end_time = time.time()
                print(f'Finished creating {self.model} in {end_time - start_time} seconds')

            avg_acc_score = np.mean(acc_score)
            end = time.time()

            print(f'Finished creating all models in {end - start} seconds')

            print(f'All scores: {acc_score}')
            print(f'Avarage score: {avg_acc_score}')
            print(f'Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
            print(f'Classification Report:\n', classification_report(y_test, y_pred))


        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        print('ROC AUC:', auc_score)

        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title(f'ROC curve for reviews (AUC: {auc_score})')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)

        return self