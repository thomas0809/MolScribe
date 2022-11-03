import torch


class DecodeStrategy(object):
    def __init__(self, pad, bos, eos, batch_size, parallel_paths, min_length, max_length,
                 return_attention=False, return_hidden=False):
        self.pad = pad
        self.bos = bos
        self.eos = eos

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        # result catching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.token_scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.hidden = [[] for _ in range(batch_size)]

        self.alive_attn = None
        self.alive_hidden = None

        self.min_length = min_length
        self.max_length = max_length

        n_paths = batch_size * parallel_paths
        self.return_attention = return_attention
        self.return_hidden = return_hidden

        self.done = False

    def initialize(self, memory_bank, device=None):
        if device is None:
            device = torch.device('cpu')
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths],
            dtype=torch.uint8, device=device)
        self.alive_log_token_scores = torch.zeros(
            [self.batch_size * self.parallel_paths, 0],
            dtype=torch.float, device=device)

        return None, memory_bank

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20 # forced non-end

    def ensure_max_length(self):
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def advance(self, log_probs, attn):
        raise NotImplementedError()

    def update_finished(self):
        raise NotImplementedError

