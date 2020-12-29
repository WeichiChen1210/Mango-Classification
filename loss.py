import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(args, criterion_type='CE'):
    criterion = None
    # CrossEntropy Loss
    if criterion_type == 'CE':
        weights = None
        if args.ce_weight is not None:
            weights = torch.tensor(args.ce_weight).cuda()
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    # Focal Loss
    elif criterion_type == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    # KL div loss
    elif criterion_type == 'KL':
        criterion = KL_div(temperature=args.temperature, size_average=None, reduce=None, reduction='batchmean')
    # triplet-like loss
    elif criterion_type == 'tri':
        criterion = TripletLikeLoss(triplet_margin=args.tri_margin, ordinal_margin=args.ord_margin)
    
    return criterion

class CE_weighted(nn.Module):
    def __init__(self, weighted_mat=None):
        super(CE_weighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.w_mat = weighted_mat
    
    def forward(self, outputs, targets):
        # CE loss of the batch
        loss = self.CE(outputs, targets)
        
        # find the indexes of predicted labels
        _, idx = torch.max(outputs, dim=1)
        
        # stack the true and predicted labels
        indices = torch.stack((targets, idx), dim=0)
        indices = indices.tolist()
        
        # get the corresponding weights
        mask = self.w_mat[indices].cuda()

        # final loss
        loss = (loss * mask).mean()
        return loss

class KL_div(nn.Module):
    def __init__(self, temperature=1, size_average=None, reduce=None, reduction='mean'):
        super(KL_div, self).__init__()
        self.temp = temperature
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs / self.temp, dim=1)
        softmax_targets = F.softmax(targets / self.temp, dim=1)
        # loss = -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
        loss = F.kl_div(log_softmax_outputs, softmax_targets, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction) * (self.temp ** 2)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # C is class num, N is batch size, H and W are the size of feature map
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        
        target = target.view(-1, 1)
        
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# online triplet loss from https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
# include triplet loss and ordinal loss
class TripletLikeLoss(nn.Module):
    def __init__(self, triplet_margin=1.0, ordinal_margin=1.0):
        super(TripletLikeLoss, self).__init__()
        self.tri_margin = triplet_margin
        self.ord_margin = ordinal_margin
        # reuse some repeating computation        
        self.pair_dist = None   # pair distance is the same with same features(triplet and ordinal use the same features)
        self.distinct_indices = None    # the indices are the same with same batch of labels(during the same iteration)
        self.triplet_mask = None    # the same with same batch of labels
        self.ordinal_mask = None    # the same with same batch of labels
    
    def reset(self):
        self.pair_dist = None
        self.distinct_indices = None
        self.triplet_mask = None
        self.ordinal_mask = None        

    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 -mask) * torch.sqrt(distances)
        
        self.pair_dist = distances
        return distances

    def _compute_distinct_indices(self, labels):
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0)).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        # indices that i != j != k (i, j, k are distinct)
        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
        distinct_indices = distinct_indices.cuda()
        self.distinct_indices = distinct_indices

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: torch.int32 `Tensor` with shape [batch_size]
        """
        # compute distinct indices
        self._compute_distinct_indices(labels)

        # labels that are equal
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        # labels[i] == labels[j] and labels[i] != labels[k]
        valid_labels = ~i_equal_k & i_equal_j

        mask = valid_labels & self.distinct_indices
        self.triplet_mask = mask
        return mask
   
    def _get_anchor_positive_triplet_mask(self, labels, device):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0)).bool().to(device)
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal
    
    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
    
    def batch_all_triplet_loss(self, labels, embeddings, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.tri_margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        if self.triplet_mask is not None:   # for the same batch of labels, triplet mask is the same
            mask = self.triplet_mask
        else:
            mask = self._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        # num_valid_triplets = mask.sum()

        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        # print(triplet_loss, fraction_positive_triplets)

        # return triplet_loss, fraction_positive_triplets
        return triplet_loss

    def batch_hard_triplet_loss(self, labels, embeddings, squared=False, device='cuda'):
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels, device).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        
        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True) # hardest negative is the close one to anchor

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.tri_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    def _get_ordinal_mask(self, labels):
        """Return a 3D mask where mask[a, b, c] is True iff the order (0, 1, 2) or (2, 1, 0) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] != labels[j] and labels[i] != labels[k] and labels[j] != labels[k]
            - labels[i] == 0 and labels[j] == 1 and labels[k] == 2
              or 
              labels[i] == 2 and labels[j] == 1 and labels[k] == 0
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        if self.distinct_indices is None:
            # Compute distinct indices
            self._compute_distinct_indices(labels)

        # labels that are equal
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        j_equal_k = label_equal.unsqueeze(0)

        #  labels[i] != labels[j] and labels[i] != labels[k] and label[j] != label[k]
        valid_labels = (~i_equal_k) & (~i_equal_j) & (~j_equal_k)

        mask = (valid_labels & self.distinct_indices)
        
        # mask the labels of the triplets that are not in the order of (0, 1, 2) or (2, 1, 0)
        # if the middle label is not 1, then not valid
        not_b = (labels != 1).nonzero(as_tuple=True)[0] # idx of labels that are not 1
        mask[:, not_b] = False
        
        self.ordinal_mask = mask
        
        return mask
    
    def batch_all_ordinal_loss(self, labels, embeddings, squared=False):
        """Build the ordinal loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            ordinal_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        if self.pair_dist is not None:  # if triplet loss has computed the pdist of the features
            pairwise_dist = self.pair_dist
        else:
            pairwise_dist = self._pairwise_distances(embeddings, squared=squared)
        
        a_b_dist = pairwise_dist.unsqueeze(2)
        a_c_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # ordinal_loss[i, j, k] will contain the ordinal loss of A=i, B=j, C=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        ordinal_loss = a_b_dist - a_c_dist + self.ord_margin

        # Put to zero the invalid triplets
        # (valid: disinct indices and label order == (0, 1, 2) or (2, 1, 0))
        if self.ordinal_mask is not None:   # for the same batch of labels, ordinal mask is the same
            mask = self.ordinal_mask
        else:
            mask = self._get_ordinal_mask(labels)
        
        ordinal_loss = mask.float() * ordinal_loss

        # Remove negative losses (i.e. the easy triplets)
        ordinal_loss[ordinal_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = ordinal_loss[ordinal_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        # num_valid_triplets = mask.sum()

        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        ordinal_loss = ordinal_loss.sum() / (num_positive_triplets + 1e-16)
        # print(triplet_loss, fraction_positive_triplets)

        # return triplet_loss, fraction_positive_triplets
        return ordinal_loss