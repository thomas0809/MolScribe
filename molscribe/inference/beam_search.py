import torch
from .decode_strategy import DecodeStrategy


class BeamSearch(DecodeStrategy):
    """Generation with beam search.
    """

    def __init__(self, pad, bos, eos, batch_size, beam_size, n_best, min_length,
                 return_attention, max_length):
        super(BeamSearch, self).__init__(
            pad, bos, eos, batch_size, beam_size, min_length, return_attention, max_length)
        self.beam_size = beam_size
        self.n_best = n_best

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.bool)

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False

    def initialize(self, memory_bank, device=None):
        """Repeat src objects `beam_size` times.
        """

        def fn_map_state(state, dim):
            return torch.repeat_interleave(state, self.beam_size, dim=dim)

        memory_bank = torch.repeat_interleave(memory_bank, self.beam_size, dim=0)
        if device is None:
            device = memory_bank.device

        self.memory_length = memory_bank.size(1)
        super().initialize(memory_bank, device)

        self.best_scores = torch.full([self.batch_size], -1e10, dtype=torch.float, device=device)
        self._beam_offset = torch.arange(
            0, self.batch_size * self.beam_size, step=self.beam_size, dtype=torch.long, device=device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1), device=device
        ).repeat(self.batch_size)
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((self.batch_size, self.beam_size), dtype=torch.float, device=device)
        self.topk_ids = torch.empty((self.batch_size, self.beam_size), dtype=torch.long, device=device)
        self._batch_index = torch.empty([self.batch_size, self.beam_size], dtype=torch.long, device=device)

        return fn_map_state, memory_bank

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs):
        """Return token decision for a step.

        Args:
            log_probs (FloatTensor): (B, vocab_size)

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)

        # Flatten probs into a list of probabilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        return topk_scores, topk_ids

    def advance(self, log_probs, attn):
        """
        Args:
            log_probs: (B * beam_size, vocab_size)
        """
        vocab_size = log_probs.size(-1)

        # (non-finished) batch_size
        _B = log_probs.shape[0] // self.beam_size

        step = len(self)  # alive_seq
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        curr_length = step + 1
        curr_scores = log_probs / curr_length  # avg log_prob
        self.topk_scores, self.topk_ids = self._pick(curr_scores)
        # topk_scores/topk_ids: (batch_size, beam_size)

        # Recover log probs
        torch.mul(self.topk_scores, curr_length, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids // vocab_size
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        if self.return_attention:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # len(self)
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)

        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            # Store finished hypothesis for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start token
                    attention[:, i, j, :self.memory_length]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score.item())
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)

        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step
        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)

        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)
