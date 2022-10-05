import json
import random
import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4


class Tokenizer(object):

    def __init__(self, path=None):
        self.stoi = {}
        self.itos = {}
        if path:
            self.load(path)

    def __len__(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return False

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
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
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

    def __init__(self, input_size=100, path=None, sep_xy=False, continuous_coords=False, debug=False):
        super().__init__(path)
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.continuous_coords = continuous_coords
        self.debug = debug

    def __len__(self):
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    @property
    def offset(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return not self.continuous_coords

    def len_symbols(self):
        return len(self.stoi)

    def fit_atom_symbols(self, atoms):
        vocab = self.special_tokens + list(set(atoms))
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.offset + self.maxx <= y
        return self.offset <= y

    def is_symbol(self, s):
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def is_atom(self, id):
        if self.is_symbol(id):
            return self.is_atom_token(self.itos[id])
        return False

    def is_atom_token(self, token):
        return token.isalpha() or token.startswith("[") or token == '*' or token == UNK

    def x_to_id(self, x):
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, id):
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)
    
    def get_output_mask(self, id):
        mask = [False] * len(self)
        if self.continuous_coords:
            return mask
        if self.is_atom(id):
            return [True] * self.offset + [False] * self.maxx + [True] * self.maxy
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

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
            x = round(x * (self.maxx - 1))
            y = round(y * (self.maxy - 1))
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
                y = self.id_to_y(sequence[i+1])
                symbol = self.itos[sequence[i+2]]
                coords.append([x, y])
                symbols.append(symbol)
            i += 3
        return {'coords': coords, 'symbols': symbols}

    def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False):
        tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            if atom_only and not self.is_atom_token(token):
                continue
            if token in self.stoi:
                labels.append(self.stoi[token])
            else:
                if self.debug:
                    print(f'{token} not in vocab')
                labels.append(UNK_ID)
            if self.is_atom_token(token):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                    elif coords is not None:
                        if atom_idx < len(coords):
                            x, y = coords[atom_idx]
                            assert 0 <= x <= 1
                            assert 0 <= y <= 1
                        else:
                            x = random.random()
                            y = random.random()
                        labels.append(self.x_to_id(x))
                        labels.append(self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        return labels, indices

    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        for i, label in enumerate(sequence):
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                continue
            token = self.itos[label]
            smiles += token
            if self.is_atom_token(token):
                if has_coords:
                    if i+3 < len(sequence) and self.is_x(sequence[i+1]) and self.is_y(sequence[i+2]):
                        x = self.id_to_x(sequence[i+1])
                        y = self.id_to_y(sequence[i+2])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(i+3)
                else:
                    if i+1 < len(sequence):
                        symbols.append(token)
                        indices.append(i+1)
        results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results


class CharTokenizer(NodeTokenizer):

    def __init__(self, input_size=100, path=None, sep_xy=False, continuous_coords=False, debug=False):
        super().__init__(input_size, path, sep_xy, continuous_coords, debug)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(list(text))
        if ' ' in vocab:
            vocab.remove(' ')
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
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
            assert all(len(s) == 1 for s in tokens)
        else:
            tokens = list(text)
        for s in tokens:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def fit_atom_symbols(self, atoms):
        atoms = list(set(atoms))
        chars = []
        for atom in atoms:
            chars.extend(list(atom))
        vocab = self.special_tokens + chars
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def get_output_mask(self, id):
        ''' TO FIX '''
        mask = [False] * len(self)
        if self.continuous_coords:
            return mask
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

    def nodes_to_sequence(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            for char in symbol:
                labels.append(self.symbol_to_id(char))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if i+2 < len(sequence) and self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_y(sequence[i+1])
                for j in range(i+2, len(sequence)):
                    if not self.is_symbol(sequence[j]):
                        break
                symbol = ''.join(self.itos(sequence[k]) for k in range(i+2, j))
                coords.append([x, y])
                symbols.append(symbol)
                i = j
            else:
                i += 1
        return {'coords': coords, 'symbols': symbols}

    def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False):
        tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            if atom_only and not self.is_atom_token(token):
                continue
            for c in token:
                if c in self.stoi:
                    labels.append(self.stoi[c])
                else:
                    if self.debug:
                        print(f'{c} not in vocab')
                    labels.append(UNK_ID)
            if self.is_atom_token(token):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                    elif coords is not None:
                        if atom_idx < len(coords):
                            x, y = coords[atom_idx]
                            assert 0 <= x <= 1
                            assert 0 <= y <= 1
                        else:
                            x = random.random()
                            y = random.random()
                        labels.append(self.x_to_id(x))
                        labels.append(self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        return labels, indices

    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                continue
            if not self.is_atom(label):
                smiles += self.itos[label]
                i += 1
                continue
            if self.itos[label] == '[':
                j = i + 1
                while j < len(sequence):
                    if not self.is_symbol(sequence[j]):
                        break
                    if self.itos[sequence[j]] == ']':
                        j += 1
                        break
                    j += 1
            else:
                if i+1 < len(sequence) and (self.itos[label] == 'C' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'l' \
                        or self.itos[label] == 'B' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'r'):
                    j = i+2
                else:
                    j = i+1
            token = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            if has_coords:
                if j+2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j+1]):
                    x = self.id_to_x(sequence[j])
                    y = self.id_to_y(sequence[j+1])
                    coords.append([x, y])
                    symbols.append(token)
                    indices.append(j+2)
                    i = j+2
                else:
                    i = j
            else:
                if j < len(sequence):
                    symbols.append(token)
                    indices.append(j)
                i = j
        results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results


if __name__ == "__main__":
    smiles_list = ["N#N", "CN=C=O", "[Cu+2].[O-]S(=O)(=O)[O-]", "O=Cc1ccc(O)c(OC)c1", "COc1cc(C=O)ccc1O",
            "CC(=O)NCCC1=CNc2c1cc(OC)cc2", "CC(=O)NCCc1c[nH]c2ccc(OC)cc12",
            "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4", "CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1",
            "CN1CCC[C@H]1c2cccnc2", r"CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\CO", "CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO",
            "CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H]", r"(C2(C)C)/C=C(\C)/C(=O)OC)C/C=C\C=C",
            "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5", "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1",
            "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2", "CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C",
            "CC[C@H](O1)CC[C@@]12CCCO2", "CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2", "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
            "CC(C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO"]
    for tokenizer in [NodeTokenizer(path="bms/vocab_uspto_new.json"), CharTokenizer(path="bms/vocab_chars.json")]:
        for smiles in smiles_list:
            # print("SMILES:", smiles)
            labels, indices = tokenizer.smiles_to_sequence(smiles, coords=[])
            results = tokenizer.sequence_to_smiles(labels[1:])
            # print("NEW SMILES:", results['smiles'])
            assert smiles == results['smiles']
            # print("SMILES TO SEQ INDICES:", indices)
            # print("SEQ TO SMILES INDICES:", results['indices'])
            assert indices == results['indices']
            for i, (symbol, idx) in enumerate(zip(results['symbols'], indices)):
                if isinstance(tokenizer, CharTokenizer):
                    assert labels[idx-1-len(symbol):idx-1] == list(tokenizer.stoi[char] for char in symbol)
                else:
                    assert labels[idx-2] == tokenizer.stoi[symbol]
