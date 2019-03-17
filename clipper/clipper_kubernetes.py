from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import json
import requests

# load data
ckd = pd.read_csv("data/kidney_disease.csv")

#make binary variable of outcome
ckd["classification"] = np.where(ckd["classification"] == "ckd" , 1, 0)

#drop unnecessary columns
ckd.drop(['id','rbc','pc','pcc','pcv','wc','rc','ba','dm','cad','appet','pe',
             'ane','sg','sc','pot','hemo','htn'],axis=1 ,inplace=True)

#create train features and labels
X = ckd.iloc[:300,]
X = X.loc[:, X.columns != 'classification']
X.fillna(0,inplace=True)
y= ckd.iloc[:300,-1]

# train a classifier
model = GradientBoostingClassifier(random_state=2019)
model.fit(X, y)

# First we need to import Clipper
from clipper_admin import ClipperConnection, KubernetesContainerManager
from clipper_admin.deployers.python import deploy_python_closure

# Create a Clipper connection
clipper_conn = ClipperConnection(KubernetesContainerManager(useInternalIP=True,
                              kubernetes_proxy_addr="127.0.0.1:8080"))

# Start a Clipper cluster or connect to a running one
clipper_conn.start_clipper()

# Register an app called 'kddtutorial'. This would create a REST endpoint
clipper_conn.register_application(name="kddtutorial", input_type="doubles",
                                  default_output="-1.0", slo_micros=10000000)

# Access the trained model via closure capture
def predict(inputs):
    global model
    pred = model.predict(inputs)
    return [str(p) for p in pred]

# Point to the gradient boosting model
model = model

# Deploy the 'predict' function as a model
deploy_python_closure(clipper_conn, name="gb-model",
                      version=1, input_type="doubles", func=predict,
                      pkgs_to_install=['scikit-learn','pandas','numpy','scipy'],
                      registry= "gkip")

# Routes requests for the application 'kddtutorial' to the model 'gb-model'
clipper_conn.link_model_to_app(app_name="kddtutorial", model_name="gb-model")

inputs = X.loc[200, X.columns != 'classification'] # use random data point
headers = {"Content-type": "application/json"}
addr = clipper_conn.get_query_addr()
response =requests.post("http://%s/%s/predict" % (addr, 'kddtutorial'), headers=headers,
              data=json.dumps({"input": list(inputs)})).json()
print(response)

clipper_conn.stop_all()
