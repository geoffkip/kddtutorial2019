import pandas as pd
import numpy as np
from os import chdir
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from clipper_admin import ClipperConnection, KubernetesContainerManager
from clipper_admin.deployers import python as python_deployer
import requests, json


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
    wd = "/Users/geoffrey.kip/Projects/kddtutorial2019"
    chdir(wd)
    
    df = pd.read_csv("data/kidney_disease.csv")
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
    return model.predict(clean_data)

#create test data
test_data= {
     "features": {
        'htn_Unknown': 0,
        'htn_no' : 0,
        'htn_yes': 1,
        'age' : 67,
        'bp': 70,
        'sg' : 1.01,
        'al': 1,
        'su': 0,
        'bgr': 102 ,
        'bu': 48,
        'sc': 3.20,
        'sod': 137,
        'pot': 5,
        'hemo':11.90}
    }

### CLIPPER PART

#Start clipper
cl = ClipperConnection(KubernetesContainerManager(useInternalIP=True, 
                              kubernetes_proxy_addr="127.0.0.1:8080"))
cl.start_clipper()

cl.register_application(name= 'kddtutorial', input_type= "doubles", default_output= "1.0", slo_micros= 100000)

python_deployer.deploy_python_closure(cl, name='gb-model', version=1,
     input_type= "doubles", func=predict, 
     pkgs_to_install=['numpy','pandas','scipy','scikit-learn'],
     registry= "gkip")



cl.link_model_to_app(app_name = 'kddtutorial', model_name = 'gb-model')

#post to endpoint
# Get Address
addr = cl.get_query_addr()
# Post Query
response = requests.post(
     "http://%s/%s/predict" % (addr, 'kddtutorial'),
     headers={"Content-type": "application/json"},
     data=json.dumps(test_data),verify=False)
result = response.json()
if response.status_code == requests.codes.ok and result["default"]:
     print('A default prediction was returned.')
elif response.status_code != requests.codes.ok:
    print(result)
else:
    print('Prediction Returned:', result)

cl.stop_all()


