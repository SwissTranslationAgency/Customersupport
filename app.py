from flask import Flask,render_template,url_for,request,jsonify
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import csv
import pandas
from tempfile import TemporaryFile
from numpy.linalg import inv
from IPython.display import IFrame
from IPython.core.display import display, HTML
from collections import Counter
from tqdm import tqdm_notebook as tqdm  # cool progress bars

app = Flask(__name__, static_url_path='/public', static_folder='./public')
        
cust_embeddings = np.load('cust_embeddings.npy')
suppo_embeddings = np.load('suppo_embeddings.npy')
#@app.route('/')
#def entry():
#    return render_template("autocomplete.html")
messages = []
my_prediction = []

@app.route('/',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            pastedreview = request.form['pastedreview']
            print(pastedreview)
        except:
            pastedreview=None

        tqdm().pandas()  # Enable tracking of progress in dataframe `apply` calls
        tweets = pd.read_csv('twcs.csv',encoding='utf-8')
        print(tweets.shape)
        tweets.head()
        first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
        QnR = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                      right_on='in_response_to_tweet_id')
        
        # Filter to only outbound replies (from companies)
        QnR = QnR[QnR.inbound_y ^ True]
        print(f'Data shape: {QnR.shape}')
        QnR.head()
        
        QnR = QnR[["author_id_x","created_at_x","text_x","author_id_y","created_at_y","text_y"]]
        QnR.head(5)
        
        #count = QnR.groupby("author_id_y")["text_x"].count()
        #c = count[count>15000].plot(kind='barh',figsize=(10, 8), color='#619CFF', zorder=2, width=width,)
        #c.set_ylabel('')
        #plt.show()
        
        amazonQnR = QnR[QnR["author_id_y"]=="AmazonHelp"]
        
        amazonQnR = amazonQnR[:10000]
        
        module_url = "/Users/berhandiclepolat/bookingberhan/tensorflow3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
        embed = hub.Module(module_url)
        
        
        messages = amazonQnR
        custmessages = amazonQnR.loc[:,"text_x"]
        custmessages = custmessages.values.tolist()
        suppomessages = amazonQnR.loc[:,"text_y"]
        suppomessages = suppomessages.values.tolist()
         
        ##print(df[['name', 'score']])
        #
        # Reduce logging output.
        #tf.logging.set_verbosity(tf.logging.ERROR)
        #
        #with tf.Session() as session:
        #  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #  cust_embeddings = session.run(embed(custmessages))
        #
        #for i, message_embedding in enumerate(np.array(cust_embeddings).tolist()):
        #    print("Message: {}".format(custmessages[i]))
        #    print("Embedding size: {}".format(len(custmessages_embedding)))
        #    custmessages_embedding_snippet = ", ".join(
        #            (str(x) for x in message_embedding[:3]))
        #    print("Embedding: [{}, ...]\n".format(custmessages_embedding_snippet))
        
        #tf.logging.set_verbosity(tf.logging.ERROR)
        #
        #with tf.Session() as session:
        #  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #  suppo_embeddings = session.run(embed(suppomessages))
        
        #for i, message_embedding in enumerate(np.array(suppo_embeddings).tolist()):
        #    print("Message: {}".format(suppomessages[i]))
        #    print("Embedding size: {}".format(len(suppomessages_embedding)))
        #    suppomessages_embedding_snippet = ", ".join(
        #            (str(x) for x in message_embedding[:3]))
        #    print("Embedding: [{}, ...]\n".format(suppomessages_embedding_snippet))
        
#        customer = ["I did the payment but received a message that the payment was not made!"]
        
        
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        with tf.Session() as session:
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          customer_embeddings = session.run(embed(pastedreview))
          
        corr = np.inner(customer_embeddings,cust_embeddings)
        corrsortindex = np.argsort(-corr)
        corrsortindex = corrsortindex.tolist()
        
        flat_list = []
        for sublist in corrsortindex:
            for item in sublist:
                flat_list.append(item)
                
        custopred = flat_list[0]     
        firstpred=flat_list[0]
        
        my_prediction=suppomessages[firstpred]
        cust_prediction=custmessages[custopred]

    
    return render_template('autocomplete.html',prediction = my_prediction, pastedreview = pastedreview)
  

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    

