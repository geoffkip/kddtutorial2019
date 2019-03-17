import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from flask import jsonify
from predict import predict
import sys
import os

app = Flask(__name__)

@app.route('/')
def index():
    return 'Flask Machine learning API'

@app.route('/predict', methods=['POST'])
def apicall():
        try:
            data= request.get_json()
        except Exception as e:
            print(str(e))
            return(bad_request(500,str(e)))

        #Load the saved model
        prediction =  predict(data)
        response = jsonify(prediction=prediction.to_json(orient="records"))
        response.status_code = 200
        return (response)

@app.errorhandler(400)
def bad_request(error=400, message="bad request"):
	message = {
			'status': error,
			'message': message,
	}
	resp = jsonify(message)
	resp.status_code = error

	return resp


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)
