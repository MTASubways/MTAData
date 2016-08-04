# This script runs the application on a local server.
# It contains the definition of routes and views for the application.

import flask
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

#---------- MODEL IN MEMORY ----------------#

# # Read in the titanic data and build a model on it
# df = pd.read_csv('data/titanic.csv')
# include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
#
# # Create dummies and drop NaN
# df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
# df = df[include].dropna()
#
# X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
# y = df['Survived']
#
# PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)





#---------- CREATING AN API, METHOD 1 ----------------#

# Initialize the app
app = flask.Flask(__name__)


# When you navigate to the page 'server/predict', this will run
# the predict() function on the parameters in the url.
#
# Example URL:
# http://localhost:4000/predict?pclass=1&sex=1&age=18&fare=500&sibsp=0
@app.route('/predict', methods=["GET"])
def predict():
    '''Makes a prediction'''
    day = float(flask.request.args['day'])
    month = float(flask.request.args['month'])
    station = float(flask.request.args['station'])

    # item = np.array([pclass, sex, age, fare, sibsp])
    # score = PREDICTOR.predict_proba(item)
    # results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    # results = {'0i':1000,'0o':2000,'4i':1000,'4o':2500,'8i':1000,'8o':2000,'12i':1000,'12o':2000,'16i':1000,'16o':2300,'20i':1000,'20o':2000}

    results = {"in":{"0":1000,"4":1500,"8":1000,"12":1000,"16":1000,"20":1000,"24":1000},"out":{"0":1000,"4":1500,"8":1000,"12":1000,"16":1000,"20":1000,"24":1000}}
    return flask.jsonify(results)



#---------- CREATING AN API, METHOD 2 ----------------#


# This method takes input via an HTML page
# @app.route('/page')
# def page():
#    with open("page.html", 'r') as viz_file:
#        return viz_file.read()
#
# @app.route('/result', methods=['POST', 'GET'])
# def result():
#     '''Gets prediction using the HTML form'''
#     if flask.request.method == 'POST':
#
#        inputs = flask.request.form
#
#        pclass = inputs['pclass'][0]
#        sex = inputs['sex'][0]
#        age = inputs['age'][0]
#        fare = inputs['fare'][0]
#        sibsp = inputs['sibsp'][0]
#
#        item = np.array([pclass, sex, age, fare, sibsp])
#        score = PREDICTOR.predict_proba(item)
#        results = {'survival chances': score[0,1], 'death chances': score[0,0]}
#        return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '0.0.0.0'
    PORT = '4000'

    app.run(HOST, PORT)
