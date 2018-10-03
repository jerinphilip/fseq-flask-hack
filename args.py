
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


ckpt = '/scratch/jerin/nat+iitb/checkpoints/unigram/8000/en-hi/transformer/checkpoint_best.pt'
args = Args(path=ckpt, max_tokens=1000, task='translation', source_lang='en', target_lang='hi', buffer_size=2, data=['data/'])

