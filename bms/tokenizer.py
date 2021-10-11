import json
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


class NodeTokenizer(object):

    def __init__(self, input_size):
        self.width = input_size
        self.height = input_size
        self.special_tokens = [PAD, SOS, EOS, UNK]
        self.offset = len(self.special_tokens)

    def __len__(self):
        return self.offset + max(self.width, self.height)

    def nodes_to_sequence(self, nodes):
        labels = [SOS_ID]
        for x, y in nodes:
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.offset + round(x * (self.width - 1)))
            labels.append(self.offset + round(y * (self.height - 1)))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        nodes = []
        i = 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            x = (sequence[i] - self.offset) / (self.width - 1)
            y = (sequence[i+1] - self.offset) / (self.height - 1)
            nodes.append([x, y])
            i += 2
        return nodes
