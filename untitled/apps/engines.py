from .translator import MTEngine, FairseqTranslator
import fairseq
from .args import multi_args as args
import pf


class LineSegmenter:
    def __call__(self, content):
        lang = None
        return lang, content.splitlines()


engines = {}

# Build fseq translator
parser = fairseq.options.get_generation_parser(interactive=True)
default_args = fairseq.options.parse_args_and_arch(parser)
kw = dict(default_args._get_kwargs())
args.enhance(print_alignment=True)
args.enhance(**kw)
fseq_translator = FairseqTranslator(args)
# segmenter = LineSegmenter()
segmenter = pf.segment.Segmenter()
tokenizer = pf.sentencepiece.SentencePieceTokenizer()

engines["mm-v1"] = MTEngine(fseq_translator, segmenter, tokenizer)

