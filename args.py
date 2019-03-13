
from collections import namedtuple

class Args:
    def __init__(self, **kwargs):
        self.custom_set = set()
        self.enhance(**kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, None)

    def enhance(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.__dict__:
                self.custom_set.add(key)
                self.__dict__[key] = val

    def __str__(self):
        lines = []
        for key in sorted(list(self.custom_set)):
            line = '{key} : {val}'.format(key=key, val=self.__dict__[key])
            lines.append(line)
        return '\n'.join(lines)


ckpt = 'checkpoints/iitb-best.ckpt'
args = Args(
                path=ckpt, 
                max_tokens=1000, 
                task='translation', 
                source_lang='en', 
                target_lang='hi', 
                buffer_size=2, 
                data=['data/']
            )

_ckpt = 'checkpoints/iitb-hi-en.ckpt'
hi_en_args = Args(
                path=_ckpt, 
                max_tokens=1000, 
                task='translation', 
                source_lang='hi', 
                target_lang='en', 
                buffer_size=2, 
                data=['data/']
            )
# ckpt = 'checkpoints/massive-multi.ckpt'
# multi_args = Args(
#                 path=ckpt, 
#                 max_tokens=1000, 
#                 task='translation', 
#                 source_lang='src', 
#                 target_lang='tgt', 
#                 buffer_size=2, 
#                 data=['data/']
#             )

ckpt = 'checkpoints/mm-new.ckpt'
multi_args = Args(
                path=ckpt, 
                max_tokens=1000, 
                task='translation', 
                source_lang='src', 
                target_lang='tgt', 
                buffer_size=2, 
                data=['data/mm-all/']
            )
