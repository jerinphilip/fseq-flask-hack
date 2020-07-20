from flask import Flask, redirect
from flask import request
from flask import jsonify
from flask import render_template
from flask import send_file

# standard lib imports
import re
from io import BytesIO
import logging

from ilmulti.translator import from_pretrained
from ilmulti.utils.language_utils import detect_lang
from pprint import pprint
# tts related imports
# from tts import api as tts_engine

class SpecialLogger:
    def __init__(self, filename):
        self.filename = filename
        self._file = open(filename, 'a+')

    def log(self, payload):
        lines = []
        line = '{} {} {}'.format(payload['remote'], payload['src_lang'],  payload['tgt_lang'])
        lines.append(line)
        line = '{} ({})'.format('\t >', payload['content'])
        lines.append(line)
        line = '{} ({})'.format('\t <', payload['tgt'][0]['hypotheses'][0]['prediction_raw'])
        lines.append(line)
        lines.append('')
        print('\n'.join(lines), file=self._file, flush=True)
        
logger = SpecialLogger("service.log")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

class LazyLoad:
    def __init__(self):
        self._loaded = {}

    def __call__(self, key, *args, **kwargs):
        if key not in self._loaded:
            self._loaded[key] = from_pretrained(key, use_cuda=False)
        return self._loaded[key](*args, **kwargs)


hub = LazyLoad()
    
@app.route('/index-old')
def hello():
    return 'Hello, World!'

@app.route('/babel/api/', methods=['POST'])
def api_translate():
    content = request.form['content']
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    tag = request.form['system']

    sequences = []
    structured_output = hub(tag, content, tgt_lang=tgt_lang)
    forwarded_ip = request.environ.get('HTTP_X_FORWARDED_FOR', 'localhost')
    payload = {
        "remote": forwarded_ip,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "content": request.form['content'],
        "tgt": structured_output
    }

    # from pprint import pformat
    pprint(structured_output)
    # app.logger.info(pformat(payload))
    # logger.log(payload)
    return jsonify(structured_output)

paths = ['tts-data/checkpoint.v2.346k.pth', 'tts-data/checkpoint.v2.246k.pth', 'tts-data/checkpoint.v1.138k.pth']
paths = [
        'tts-data/checkpoint.v2.996k.pth'
        # , 'tts-data/checkpoint.v2.569k.pth'
        # , 'tts-data/checkpoint.v2.810k.pth'
        ]

# tts_models = {'m{}'.format(i) : tts_engine.get_model(p) for i, p  in enumerate(paths)}
disp_names = ['.'.join(p.split('.')[1:3]) for p in paths]

@app.route('/babel', methods=['POST', 'GET'])
@app.route('/babel/multi-gui', methods=['POST', 'GET'])
@app.route('/babel/gui/hi-en', methods=['GET'])
@app.route('/babel/gui', methods=['GET'])
@app.route('/babel/frontend', methods=['GET'])
def frontend():
    return render_template('dynamic_index.html', multi=True, audio_names=disp_names)

############# tts code ############


# tts_model = tts_engine.get_model('tts-data/checkpoint.pth')

@app.route('/babel/tts', methods=['GET'])
def tts_home():
    return render_template('tts_index.html', audio_src='', sentence=None)

@app.route('/babel/tts/api/m0', methods=['GET'], endpoint='m0')
# @app.route('/babel/tts/api/m1', methods=['GET'], endpoint='m1')
# @app.route('/babel/tts/api/m2', methods=['GET'], endpoint='m2')
def tts_speak():
    content = request.args['q'].splitlines()
    buf = BytesIO()

    for line in content:
        line = line.strip()
        tts_engine.tts(tts_models[request.endpoint], line, buf)

    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        attachment_filename='audio.wav',
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    app.run('0.0.0.0', port=1618, debug=True)

