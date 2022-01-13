import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from bms.tokenizer import PAD_ID, MASK, MASK_ID


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        log_probs = F.log_softmax(output, dim=-1)

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # reduction mean or sum?
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        loss = self.criterion(output, target)
        return loss


class GraphLoss(nn.Module):

    def __init__(self):
        super(GraphLoss, self).__init__()
        weight = torch.ones(7) * 10
        weight[0] = 1
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)

    def forward(self, outputs, targets):
        results = {}
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        return results


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_coord: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coord: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "coords": Tensor of dim [batch_size, num_queries, 2] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "coords": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_coord = outputs["coords"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_coord = torch.cat([v["coords"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_coord = torch.cdist(out_coord, tgt_coord, p=1)

        # Final cost matrix
        C = self.cost_coord * cost_coord + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["coords"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes):
        """ Create the criterion. """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        # self.weight_dict = weight_dict
        # self.eos_coef = eos_coef
        self.losses = ['labels', 'coords', 'edges']
        labels_weight = torch.ones(self.num_classes)
        labels_weight[PAD_ID] = 0.1
        self.register_buffer('labels_weight', labels_weight)
        edges_weight = torch.ones(7)
        edges_weight[0] = 0.1
        self.register_buffer('edges_weight', edges_weight)

    def loss_labels(self, outputs, targets, indices, num_nodes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits' in outputs
        src_logits = outputs['logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], PAD_ID, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # print(src_logits.shape, target_classes.shape)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.labels_weight)
        losses = {'labels': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_nodes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_coords(self, outputs, targets, indices, num_nodes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'coords' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_coords = outputs['coords'][idx]
        target_coords = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_coords = F.l1_loss(src_coords, target_coords, reduction='none')
        losses = {'coords': loss_coords.sum() / num_nodes}
        return losses

    def loss_edges(self, outputs, targets, indices, num_nodes):
        """Compute the edge loss."""
        assert 'edges' in outputs
        src_edges = outputs['edges']
        tgt_edges = [t['edges'] for t in targets]
        loss = 0
        for b, (i, j) in enumerate(indices):
            n = len(i)
            src = src_edges[b][i][:, i].view(n*n, -1)
            tgt = tgt_edges[b][j][:, j].view(n*n)
            loss += F.cross_entropy(src, tgt, self.edges_weight)

        losses = {'edges': loss / len(indices)}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, nodes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            'coords': self.loss_coords,
            'edges': self.loss_edges
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, nodes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_nodes = sum(len(t["labels"]) for t in targets)
        num_nodes = torch.as_tensor([num_nodes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_nodes)
        num_nodes = torch.clamp(num_nodes / torch.distributed.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_nodes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux' in outputs:
            for i, aux_outputs in enumerate(outputs['aux']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'edges':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_nodes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        loss = sum(losses.values())
        return losses


class Criterion(nn.Module):

    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        for format_ in args.formats:
            if format_ == 'edges':
                criterion['edges'] = GraphLoss()
            elif format_ == 'graph':
                criterion['graph'] = SetLoss(tokenizer['graph'].len_symbols())
            elif format_ == 'grid':
                weight = torch.ones(tokenizer['grid'].len_symbols())
                weight[PAD_ID] = 1
                criterion['grid'] = nn.CrossEntropyLoss(weight)
            else:
                if MASK in tokenizer[format_].stoi:
                    ignore_indices = [PAD_ID, MASK_ID]
                else:
                    ignore_indices = []
                criterion[format_] = SequenceLoss(args.label_smoothing, len(tokenizer[format_]),
                                                  ignore_index=PAD_ID, ignore_indices=ignore_indices)
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, results, refs):
        losses = {}
        reweight_coef = refs.get('reweight_coef', None)
        for format_ in results:
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_](predictions, targets)
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                if loss_.numel() > 1:
                    if reweight_coef is not None:
                        loss_ = (reweight_coef.to(loss_.device) * loss_).mean()
                    else:
                        loss_ = loss_.mean()
                losses[format_] = loss_
        return losses
