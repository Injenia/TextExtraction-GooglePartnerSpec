from __future__ import print_function
from __future__ import division
import os
from lib import predict_pdf as pp
from lib import extract_parts as ep
from lib import text_extraction as te
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from tempfile import NamedTemporaryFile
import base64

def load_predictor_extractor():
    models = pp.load_models()
    name_extractor = ep.NotaioNameExtractor.load_from_file()
    pe = ep.PartsExtraction.load_from_files('models/extraction_model_30_all.json',
                                     'models/extraction_weights_30_all.h5',
                                     'first_5000_words_extraction.json')
    return ep.PredictorExtractor(models, pe, name_extractor)

app = Flask(__name__)
CORS(app)
pred_extract = load_predictor_extractor()
upload_dir = '../test_upload'

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/document', methods=['GET', 'POST', 'OPTIONS'])
def predict_and_extract():
    if request.method == 'POST' or request.method == 'OPTIONS':
        if request.is_json:
            json_req = request.get_json()
            filename = os.path.join(upload_dir, json_req['filename'])
            content = base64.b64decode(json_req['content'])

            with open(filename, 'wb') as o:
                o.write(content)
            
            try:
                res = pred_extract.predict_extract_pdf_json(filename)
            except Exception as e:
                return jsonify({'exception':str(e), "extracted_text":te.extract_text(filename, do_ocr=False, pages=-1)})
            finally:
                os.remove(filename)
            
            return jsonify(res)
        else:
            return jsonify('Not json')
    else:
        return jsonify('Send with POST please')
    