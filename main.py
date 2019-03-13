from flask import Flask, redirect
from flask import request
from flask import jsonify
from flask import render_template
from flask import send_file

from interactive import build_instance
from tokenizer import SPTokenizer
from pf.sentencepiece import SentencePieceTokenizer
from args import args, multi_args, hi_en_args

# standard lib imports
import re
from io import BytesIO

# tts related imports
from tts import api as tts_engine
spt = SentencePieceTokenizer()

def process(ordered_results, tokenized, lines):
    out = []
    for i, result in enumerate(ordered_results):
        export = {
                "source": tokenized[i],
                "source_raw": lines[i],
                "hypotheses": [],
        }

        def detok(h):
            h = h[1:]
            h = h.replace(" ", "")
            h = h.replace("▁", " ")
            return h


        for h in result.hypos:
            tag, score, hyp = h.split('\t')
            h_exp = {
                "score": score,
                "prediction": hyp,
                "prediction_raw": detok(hyp)
            }
            export["hypotheses"].append(h_exp)

        out.append(export)
    return out 

def twrapped(f):
    def __inner(lines):
        hi_path = 'data/hi.8000.model'
        en_path = 'data/en.8000.model'

        from collections import namedtuple
        Pair = namedtuple('Pair', 'hi en')
        t = Pair(en=SPTokenizer(en_path), hi=SPTokenizer(hi_path))
        tokenized = []
        for i, src in enumerate(lines):
            enc = t.en.tok(src)
            tokenized.append(enc)
        ordered_results = f(tokenized)
        return process(ordered_results, tokenized, lines)
    return __inner

def ttwrapped(f):
    def __inner(lines):
        hi_path = 'data/hi.8000.model'
        en_path = 'data/en.8000.model'

        from collections import namedtuple
        Pair = namedtuple('Pair', 'hi en')
        t = Pair(en=SPTokenizer(en_path), hi=SPTokenizer(hi_path))
        tokenized = []
        for i, src in enumerate(lines):
            enc = t.hi.tok(src)
            tokenized.append(enc)
        ordered_results = f(tokenized)
        return process(ordered_results, tokenized, lines)
    return __inner

def agwrapped(f):
    def __inner(lines, translate_to, source_lang=None):
        tokenized = []
        for i, src in enumerate(lines):
            lang, enc = spt(src, source_lang)
            enc = ' '.join(enc)
            injected = '__t2{xx}__ {enc}'.format(xx=translate_to, enc=enc)
            tokenized.append(injected)
        print(tokenized)
        ordered_results = f(tokenized)
        return process(ordered_results, tokenized, lines)
    return __inner


agfish = build_instance(multi_args)
agfish = agwrapped(agfish)
babel_fish = build_instance(args)
babel_fish = twrapped(babel_fish)

# hi_en_i = build_instance(hi_en_args)
# hi_en_i = ttwrapped(hi_en_i)
hi_en_i = lambda x: []

models = {
    "wat-en-hi": babel_fish,
    "multi": agfish
}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/index-old')
def hello():
    return 'Hello, World!'

@app.route('/babel/gui', methods=['POST', 'GET'])
def guitranslate():
    sentence = {"id": '', "content": "", "translate_to": "", "source_lang": ""}
    if request.method == 'GET':
        return render_template('index.html', sentence=sentence, multi=False)
    if request.method == 'POST':
        print(dict(request.form))
        lines = request.form['src'].splitlines()
        results = babel_fish(lines)
        results = list(map(lambda x: x["hypotheses"][0]["prediction_raw"], results))
        serialized = '\n'.join(results)
        sentence = {"id": '', "content": request.form['src'], "tgt": serialized}
        return render_template('index.html', sentence=sentence, multi=False)

@app.route('/babel/gui/hi-en', methods=['POST', 'GET'])
def _guitranslate():
    sentence = {"id": '', "content": "", "translate_to": "", "source_lang": ""}
    if request.method == 'GET':
        return render_template('index.html', sentence=sentence, multi=False)
    if request.method == 'POST':
        print(dict(request.form))
        lines = request.form['src'].splitlines()
        results = hi_en_args(lines)
        results = list(map(lambda x: x["hypotheses"][0]["prediction_raw"], results))
        serialized = '\n'.join(results)
        sentence = {"id": '', "content": request.form['src'], "tgt": serialized}
        return render_template('index.html', sentence=sentence, multi=False)

@app.route('/babel/multi-gui', methods=['POST', 'GET'])
def multiguitranslate():
    # return redirect("/babel/frontend", code=302)

    sentence = {"id": '', "content": "", "translate_to": "", "source_lang": ""}
    if request.method == 'GET':
        return render_template('index.html', sentence=sentence, multi=True)
    if request.method == 'POST':
        lines = request.form['src'].splitlines()
        #lines = list(map(lambda x: x.encode().decode("utf-8"), lines))
        translate_to = request.form['translate_to']
        source_lang = request.form['source_lang']
        sentence["translate_to"] = translate_to
        sentence["source_lang"] = source_lang
        results = agfish(lines, translate_to, source_lang)
        results = list(map(lambda x: x["hypotheses"][0]["prediction_raw"], results))
        serialized = '\n'.join(results)
        sentence = {"id": '', "content": request.form['src'], "tgt": serialized}
        return render_template('index.html', sentence=sentence, multi=True)


@app.route('/babel', methods=['POST', 'GET'])
def translate():
    if request.method == 'GET':
        contents = request.args.get('q')

    elif request.method == 'POST':
        contents = request.form['q']

    lines = contents.splitlines()
    results = babel_fish(lines)
  
    return jsonify(results)

mtok_factory = {
        "mm-v1": (build_instance(multi_args), SentencePieceTokenizer()),
        "iitb-en-hi": (build_instance(args), None),
}

@app.route('/babel/api/', methods=['POST'])
def api_translate():
    content = request.form['content'].splitlines()
    # If required, do a sentence tokenization at language level.
    # More broken sentences => Better results
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    tag = request.form['system']
    model, tokenize = mtok_factory[tag]

    sequences = []
    for line in content:
        if src_lang == '-detect-':
            src_lang, tokens = tokenize(line)
        else:
            src_lang, tokens = tokenize(line, lang=src_lang)
        tokens_space_joined = ' '.join(tokens)
        injected = '__t2{xx}__ {enc}'.format(xx=tgt_lang, enc=tokens_space_joined)
        sequences.append(injected)

    ordered_results = model(sequences)
    structured_output = process(ordered_results, sequences, content)
    return jsonify(structured_output)

@app.route('/babel/frontend', methods=['GET'])
def frontend():
    return render_template('dynamic_index.html', multi=True)

############# tts code ############


tts_model = tts_engine.get_model('tts-data/checkpoint.pth')

@app.route('/babel/tts', methods=['GET'])
def tts_home():
    return render_template('tts_index.html', audio_src='', sentence=None)

@app.route('/babel/tts/api', methods=['GET'])
def tts_speak():
    content = request.args['q'].splitlines()
    buf = BytesIO()

    for line in content:
        line = line.strip()
        tts_engine.tts(tts_model, line, buf)

    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        attachment_filename='audio.wav',
        mimetype='audio/wav'
    )
    # lines = re.split(u'।|\n', contents)

if __name__ == '__main__':
    app.run('0.0.0.0', port=1618, debug=True)

