import sys
sys.path.append('..')
from bms.utils import Tokenizer
from bms.chemistry import RGROUP_SYMBOLS, get_substitutions


# https://github.com/deepchem/deepchem/blob/master/deepchem/feat/tests/data/vocab.txt
with open('vocab.txt') as f:
    lines = [line.strip() for line in f]
    data = lines[15:]

for symbol in RGROUP_SYMBOLS:
    data.append(f'[{symbol}]')

for sub in get_substitutions():
    for symbol in sub.abbrvs:
        data.append(f'[{symbol}]')

print(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
tokenizer.save('vocab_rf.json')
