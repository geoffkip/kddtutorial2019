import json
import requests

data= {
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

header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

res=requests.post('http://localhost:30502/predict', data=json.dumps(data) , headers=header, verify= False)
print(res.status_code) # Should return 200
print(res.json()) # returns predictions
