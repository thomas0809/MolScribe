import sys
sys.path.append('..')
from bms.utils import Tokenizer

with open('vocab.txt') as f:
    lines = [line.strip() for line in f]
    data = lines[15:]

for symbol in ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'X', 'Ar']:
    data.append(f'[{symbol}]')
print(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
tokenizer.save('vocab.json')
