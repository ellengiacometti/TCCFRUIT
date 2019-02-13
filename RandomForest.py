import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd

if __name__ == '__main__':
    train = pd.read_csv('Train.csv',index_col='Object')
    test = pd.read_csv('Test.csv', index_col='Object')
    Texture = train.columns[1:3]
    TextureLabel = train['TextureLabel']
    TextureLabelTest=test['TextureLabel']
    le = LabelEncoder()
    # # Fit the encoder to the pandas column
    le.fit(TextureLabelTest)
    TextureLabelTest = le.transform(TextureLabelTest)
    TextureLabel = le.transform(TextureLabel)
#class_weight={0:1,1:2} should do the job. Now, class 0 has weight 1 and class 1 has weight 2.
##Create a based model
# rf = RandomForestClassifier(random_state=42)
# # Create the parameter grid based on the results of random search
# param_grid = {
#     'n_estimators': [10, 20, 30, 40,50,60,70,80,90,100,150,200],
#     'class_weight': [{0: w1,1: w2} for w1 in [2, 4, 6, 10,12] for w2 in [2, 4, 6, 10,12]]
#
# }
# # Instantiate the grid search model
# CV_rfc = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 5, n_jobs = -1, verbose = 2)
# # Fit the grid search to the data
# CV_rfc.fit(train[Texture], TextureLabel)
# CV_rfc.best_params_
# pprint(CV_rfc.best_params_)


##class_weight={0:1,1:2} should do the job. Now, class 0 has weight 1 and class 1 has weight 2.



BEST_RFC=RandomForestClassifier(class_weight={0: 1, 1: 6}, n_estimators= 10)
BEST_RFC.fit(train[Texture], TextureLabel)
pred=BEST_RFC.predict(test[Texture])
print("Accuracy for Random Forest on CV data: ",accuracy_score(TextureLabelTest,pred))

# # #TODO https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74