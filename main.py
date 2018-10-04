
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from interactive import build_instance
from tokenizer import SPTokenizer

babel_fish = build_instance()

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
        out = []
        for i, result in enumerate(ordered_results):
            export = {
                    "source": tokenized[i],
                    "source_raw": lines[i],
                    "hypotheses": [],
            }
            for h in result.hypos:
                tag, score, hyp = h.split('\t')
                h_exp = {
                    "score": score,
                    "prediction": hyp,
                    "prediction_raw": t.hi.detok(hyp)
                }
                export["hypotheses"].append(h_exp)

            out.append(export)
        return out 
    return __inner

babel_fish = twrapped(babel_fish)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/index-old')
def hello():
    return 'Hello, World!'

@app.route('/babel/gui', methods=['POST', 'GET'])
def guitranslate():
    sentence = {"id": '', "content": ""}
    if request.method == 'GET':
        return render_template('index.html', sentence=sentence)
    if request.method == 'POST':
        lines = request.form['src'].splitlines()
        results = babel_fish(lines)
        results = list(map(lambda x: x["hypotheses"][0]["prediction_raw"], results))
        serialized = '\n'.join(results)
        sentence = {"id": '', "content": request.form['src'], "tgt": serialized}
        return render_template('index.html', sentence=sentence)

@app.route('/babel', methods=['POST', 'GET'])
def translate():
    if request.method == 'GET':
        contents = request.args.get('q')

    elif request.method == 'POST':
        contents = request.form['q']

    lines = contents.splitlines()
    results = babel_fish(lines)
    
    return jsonify(results)


if __name__ == '__main__':
    app.run('0.0.0.0', port=1618)


