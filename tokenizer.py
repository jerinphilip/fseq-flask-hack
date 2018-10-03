
import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model):
        self.model = model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model)

    def tok(self, string):
        return ' '.join(self.sp.EncodeAsPieces(string))

    def detok(self, string):
        pieces = string.split()
        return self.sp.DecodePieces(pieces)


if __name__ == '__main__':
    import sys
    hi_path = 'data/hi.8000.model'
    en_path = 'data/en.8000.model'

    from collections import namedtuple
    Pair = namedtuple('Pair', 'hi en')
    t = Pair(en=SPTokenizer(en_path), hi=SPTokenizer(hi_path))

    for src in sys.stdin:
        spm_encoded = t.en.tok(src)
        spm_decoded = t.en.detok(spm_encoded)
        print("enc:", spm_encoded)
        print("dec:", spm_decoded)

