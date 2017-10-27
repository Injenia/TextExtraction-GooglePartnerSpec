from __future__ import print_function
from __future__ import division
import os
from lib import predict_pdf as pp
from lib import extract_parts as ep
from lib import text_extraction as te
import json
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import base64

prediction_models = {'gensim_file':'../models/gensim_model_5000.d2v', 
                     'keras_model_file':'../models/keras_model.json',
                     'keras_weights_file':'../models/keras_weights_5000.h5',    
                     'permitted_words_file':'../dictionaries/first_5000_words.json'}

prediction_models_verb = {'gensim_file':'../models/gensim_5000_model_with_verb.d2v', 
                         'keras_model_file':'../models/keras_model.json',
                         'keras_weights_file':'../models/keras_new_weights_with_verb_es.h5',    
                         'permitted_words_file':'../dictionaries/first_5000_words_with_verb_cost.json'}

prediction_models_we = {'keras_model_file':'../models/keras_model_word_embedding.json',
            'keras_weights_file':'../models/keras_weights_word_embedding.h5',
            'reduced_dictionary_file':'../dictionaries/reduced_dictionary_cost.json'}

use_word_embedding = True

def load_predictor_extractor():
    if use_word_embedding:
        models = pp.load_models_we(**prediction_models_we)
    else:
        models = pp.load_models(**prediction_models)           
    name_extractor = ep.NotaioNameExtractor.load_from_file()
    pe = ep.PartsExtraction.load_from_files('../models/extraction_model_30_all.json',
                                     '../models/extraction_weights_30_all.h5',
                                     '../dictionaries/first_5000_words_extraction.json')
    return ep.PredictorExtractor(models, pe, name_extractor)

app = Flask(__name__)
CORS(app)
pred_extract = load_predictor_extractor()
upload_dir = '../temp_upload_flask'
allowed_extensions = set(['pdf'])
with open("../samples/5115047380001.json") as f:
    sample = json.load(f)
#print(sample)
    
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/sample', methods=['GET', 'POST', 'OPTIONS'])
def sample_response():
    if request.method == 'POST' or  request.method == 'OPTIONS':
        #print(sample)
        return jsonify(sample)

@app.route('/document', methods=['POST', 'OPTIONS'])
def predict_and_extract():
    if request.method == 'POST' or  request.method == 'OPTIONS':
        if request.is_json:
            json_req = request.get_json()
            filename = os.path.join(upload_dir, json_req['filename'])
            if filename.split('.')[-1].lower() not in allowed_extensions:
                return jsonify('Please upload a pdf'), 415
            
            content = base64.b64decode(json_req['content'])

            with open(filename, 'wb') as o:
                o.write(content)
            
            try:
                res = pred_extract.predict_extract_pdf_json(filename)
            except Exception as e:
                return jsonify(ep.build_json_response())
            finally:
                os.remove(filename)
            
            return jsonify(res)
        else:
            return jsonify('Not json')
    else:
        return jsonify('Send with POST please')
    