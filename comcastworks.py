from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

import scipy
import json
import requests

from absl import logging


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('autocomplete.html')

#reviewbody = []
#reviewresponse = []
#with open('reviewskaggle.csv', encoding='utf-8', errors='ignore') as csv_file:
#    lines = csv_file.readlines()
#    for row in lines:
#        reviewbody.append(row)
#        
#with open('reviewskaggle.csv', encoding='utf-8', errors='ignore') as csv_file:
#    resp = csv_file.readlines()
#    for row in resp:
#        reviewresponse.append(row)
   
comcast2 = []

with open('comcastfinalall.csv', encoding='utf-8', errors='ignore') as csv_file:
    resp = csv_file.readlines()
    for row in resp:
        comcast2.append(row)  

ko =[]
my_prediction= []
my_prediction2 = []
        
comcast2_embeddings = np.load('comcastall.npy')

#embed_fn = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")



@app.route('/',methods=['POST', 'GET'])
def predict(): 
    if request.method == 'POST':
        messages = request.form['message']
        messages = [messages]
        print(messages)   
    module_url = ("/Users/berhandiclepolat/bookingberhan/tensorflow3")
    embed = hub.Module(module_url)
    logging.set_verbosity(logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                    (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

#        sendtens = data.toarray()
    corr = np.inner(message_embeddings,comcast2_embeddings)
    corrsortindex = np.argsort(corr)
    corrsortindex = corrsortindex.tolist()
    flat_list = []
    for sublist in corrsortindex:
        for item in sublist:
            flat_list.append(item)
            ko = flat_list[-5:]
            ko.reverse()
    my_prediction = "First Prediction:" + comcast2[ko[0]]
    messages = (', '.join(messages))
    my_prediction = my_prediction.replace('"','')
    my_prediction2 = '\n' + "Second Prediction:" +comcast2[ko[1]]
    my_prediction2 = my_prediction2.replace('"','')
    print (my_prediction)
    return render_template('autocomplete.html',prediction = my_prediction+my_prediction2, message = messages)
    

#        data = [message]
#		vect = cv.transform(data).toarray()
#
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
