import json
import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class Tokenizer(object):

    def __init__(self, path=None):
        self.stoi = {}
        self.itos = {}
        if path:
            self.load(path)

    def __len__(self):
        return len(self.stoi)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = [PAD, SOS, EOS, UNK] + sorted(vocab)
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        if tokenized:
            tokens = text.split(' ')
        else:
            tokens = atomwise_tokenizer(text)
        for s in tokens:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


class NodeTokenizer(Tokenizer):

    def __init__(self, input_size=100, path=None, sep_xy=False):
        super().__init__(path)
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK]
        self.offset = len(self.stoi)

    def __len__(self):
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    def len_symbols(self):
        return len(self.stoi)

    def fit_atom_symbols(self, atoms):
        vocab = self.special_tokens + atoms
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        self.offset = len(self.stoi)

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.offset + self.maxx <= y
        return self.offset <= y

    def is_symbol(self, s):
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def x_to_id(self, x):
        return self.offset + int(x * (self.maxx - 1))

    def y_to_id(self, y):
        if self.sep_xy:
            return self.offset + self.maxx + int(y * (self.maxy - 1))
        return self.offset + int(y * (self.maxy - 1))

    def id_to_x(self, id):
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        if self.sep_xy:
            return (id - self.offset - self.maxx) / self.maxy
        return (id - self.offset) / (self.maxy - 1)

    def symbol_to_id(self, symbol):
        if symbol not in self.stoi:
            return UNK_ID
        return self.stoi[symbol]

    def symbols_to_labels(self, symbols):
        labels = []
        for symbol in symbols:
            labels.append(self.symbol_to_id(symbol))
        return labels

    def labels_to_symbols(self, labels):
        symbols = []
        for label in labels:
            symbols.append(self.itos[label])
        return symbols

    def nodes_to_grid(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        grid = np.zeros((self.maxx, self.maxy), dtype=int)
        for [x, y], symbol in zip(coords, symbols):
            x = int(x * (self.maxx - 1))
            y = int(y * (self.maxy - 1))
            grid[x][y] = self.symbol_to_id(symbol)
        return grid

    def grid_to_nodes(self, grid):
        coords, symbols, indices = [], [], []
        for i in range(self.maxx):
            for j in range(self.maxy):
                if grid[i][j] != 0:
                    x = i / (self.maxx - 1)
                    y = j / (self.maxy - 1)
                    coords.append([x, y])
                    symbols.append(self.itos[grid[i][j]])
                    indices.append([i, j])
        return {'coords': coords, 'symbols': symbols, 'indices': indices}

    def nodes_to_sequence(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            labels.append(self.symbol_to_id(symbol))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i + 2 < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_x(sequence[i+1])
                symbol = self.itos[sequence[i+2]]
                coords.append([x, y])
                symbols.append(symbol)
            i += 3
        return {'coords': coords, 'symbols': symbols}
