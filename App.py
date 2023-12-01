import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from gensim.models import Word2Vec
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle

nltk.download('punkt')
nltk.download('stopwords')

App = Flask(__name__)
api = Api(App)


tokenizer = pickle.load(open("token.pickle", 'rb'))
loaded_model = tf.keras.models.load_model("ai.h5")

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

def preprocessing_teks(dataset):
  dataset = dataset.lower() #CaseFolding
  dataset = dataset.replace(r'[^\w\s]+', '') # Penghapusan Tanda Baca
  dataset = nltk.word_tokenize(dataset)
  return dataset

class PredictHoax(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        teks = args['query']

        # vectorize the user's query and make a prediction
        teks_preprocess = preprocessing_teks(teks)
        prepad = tokenizer.texts_to_sequences([teks_preprocess])
        prepad2 = pad_sequences(prepad,maxlen=100,padding='pre')    

        # Output either 'Negative' or 'Positive' along with the score

        if(np.argmax(loaded_model.predict(prepad2)) == 1):
          pred_text = "Hoax"        
        else:
          pred_text = "Fakta"
            
        # round the predict proba value and set to new variable

        # create JSON object
        output = {'prediction': pred_text}
        
        return output
        
api.add_resource(PredictHoax, '/')
if __name__ == '__main__':
    App.run(debug=True)