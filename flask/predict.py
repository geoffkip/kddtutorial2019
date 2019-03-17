import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Transform features for training
def transform(features):
    categorical =  features.select_dtypes(include=[np.object])
    categorical.fillna('Unknown',inplace=True)
    features_cat = pd.get_dummies(categorical)

    # Numerical Variables
    numerical = features.select_dtypes(include=[np.float64,np.int64])
    numerical.fillna(method="ffill",inplace=True)

    #Create training features
    train_features= pd.concat([features_cat, numerical], axis=1)
    return train_features

def get_data():
    df = pd.read_csv("kidney_disease.csv")
    df = shuffle(df)
    df.head()


    #reformat outcome
    df["classification"] = np.where(df["classification"] == "ckd" , 1, 0)
    df.drop(['id','rbc','pc','pcc','pcv','wc','rc','ba','dm','cad','appet','pe',
             'ane'],axis=1 ,inplace=True)

    # Split labels and features for test set
    labels = df.iloc[:,-1]
    features = df.loc[:, df.columns != 'classification']
    features = transform(features)
    features.fillna(0,inplace=True)
    return features, labels


def train_model():
    features , labels = get_data()

    #Split data into training and test sets
    test_size = 0.30
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)

    #fit model
    model =  GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    model_columns=list(X_train.columns)
    return model, model_columns

def process_features(raw_data):
    features = pd.pivot_table(pd.DataFrame(raw_data).reset_index(),values="features",columns="index").infer_objects()
    return features

def predict(data):
    clean_data = process_features(data)
    model, model_columns = train_model()
    clean_data = clean_data[model_columns]
    return pd.DataFrame(model.predict(clean_data),columns = ['prediction'])
