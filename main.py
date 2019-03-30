from flask import Flask, redirect
from flask import request
from flask import jsonify
from flask import render_template
from flask import send_file

from interactive import build_instance
from tokenizer import SPTokenizer
from pf.sentencepiece import SentencePieceTokenizer

# standard lib imports
import re
from io import BytesIO

# tts related imports
# from tts import api as tts_engine
from untitled.engines import engines

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/index-old')
def hello():
    return 'Hello, World!'


@app.route('/babel/api/', methods=['POST'])
def api_translate():
    content = request.form['content']
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    tag = request.form['system']
    engine = engines.get(tag)
    results = engine(content, tgt_lang)
    return jsonify(results)

# paths = ['tts-data/checkpoint.v2.346k.pth', 'tts-data/checkpoint.v2.246k.pth', 'tts-data/checkpoint.v1.138k.pth']
# tts_models = {'m{}'.format(i) : tts_engine.get_model(p) for i, p  in enumerate(paths)}
# disp_names = ['.'.join(p.split('.')[1:3]) for p in paths]

@app.route('/babel/gui', methods=['GET'])
@app.route('/babel/gui/hi-en', methods=['GET'])
@app.route('/babel/multi-gui', methods=['GET'])
@app.route('/babel', methods=['GET'])
@app.route('/babel/frontend', methods=['GET'])
def frontend():
    disp_names = []
    return render_template('dynamic_index.html', multi=True, audio_names=disp_names)

############# tts code ############


# tts_model = tts_engine.get_model('tts-data/checkpoint.pth')

# @app.route('/babel/tts', methods=['GET'])
# def tts_home():
#     return render_template('tts_index.html', audio_src='', sentence=None)
# 
# @app.route('/babel/tts/api/m0', methods=['GET'], endpoint='m0')
# @app.route('/babel/tts/api/m1', methods=['GET'], endpoint='m1')
# @app.route('/babel/tts/api/m2', methods=['GET'], endpoint='m2')
# def tts_speak():
#     content = request.args['q'].splitlines()
#     buf = BytesIO()
# 
#     for line in content:
#         line = line.strip()
#         tts_engine.tts(tts_models[request.endpoint], line, buf)
# 
#     buf.seek(0)
#     return send_file(
#         buf,
#         as_attachment=True,
#         attachment_filename='audio.wav',
#         mimetype='audio/wav'
#     )

if __name__ == '__main__':
    app.run('0.0.0.0', port=2619, debug=True)

