# Kdd Tutorial 2019
## Deploying Model Servers to Support Endpoints for Real Time Predictions

Chronic kidney dataset https://www.kaggle.com/mansoordaku/ckdisease

Clipper is a low-latency prediction serving system for machine learning. Clipper makes it simple to integrate machine learning into user-facing serving systems.

## Overview
This tutorial demonstrates how to train a simple Gradient Boosting Classifier model to predict whether a patient has Chronic Kidney Disease (CKD) or not. A predict function is then deployed using the clipper
DockerContainerManager in the ``clipper_docker.py`` script and the KubernetesContainerManager in the ``clipper_kubernetes.py`` script.  
Basically you can just run each script to see how clipper works with the DockerContainerManager and the KubernetesContainerManager.

Basically these Container Manager creates a docker container with the Gradient Boosting model. The KubernetesContainerManager deploys the docker container on Kubernetes.
Since the model is containerized in a Docker container one can post data to a API endpoint and this will return predictions for that data.

## Kubernetes Deployment
For the KubernetesContainerManager make sure you first run a proxy by running
```
kubectl proxy --port 8080
```

In addition you must make sure you authenticate Kubernetes to your private docker registry.
For example
```
kubectl create secret docker-registry myregistrykey \
--docker-server=yourregistrydomain.io \
--docker-username=yourusername \
--docker-password=yourpassword \
--docker-email=email@server.com
```
Then to patch the kubernetes service account with your key run
```
kubectl patch serviceaccount default -p ‘{“imagePullSecrets”: [{“name”: “myregistrykey”}]}’
```
Now you should be ready to deploy your kubernetes pod with the model.

To check that all the pods are up and running run
```
kubectl get pods
```

To find the clipper query front end you can run
```
kubectl describe service query-frontend
```
and look for the line saying ``Endpoints``

## Posting data to the endpoint
For example using Python to post to the api address endpoint looks like this
```
headers = {"Content-type": "application/json"}
addr = clipper_conn.get_query_addr()
response =requests.post("http://%s/%s/predict" % (addr, 'kddtutorial'), headers=headers,
              data=json.dumps({"input": list(inputs)})).json()
print(response)
```

We then get a prediction response in the response variable.
