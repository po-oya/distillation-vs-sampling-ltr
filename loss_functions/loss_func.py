import torch as _torch


def mask_padding(x: _torch.Tensor, n: _torch.Tensor, fill: float = 0.0):
    n_batch, n_results = x.shape
    n = n.unsqueeze(-1)
    mask = _torch.arange(n_results).repeat(n_batch, 1).type_as(x)
    x = x.float()
    x[mask >= n] = fill
    return x


class PointWiseRegLoss(_torch.nn.Module):
    def __init__(self):
        r""""""
        super().__init__()
        self._loss = _torch.nn.MSELoss(reduction='none')

    def forward(self, scores: _torch.FloatTensor, relevance: _torch.LongTensor,
                n: _torch.LongTensor) -> _torch.FloatTensor:
        """Computes the loss for given batch of samples.

        Args:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        relevance = relevance.to(_torch.float32)
        #####

        raw_loss = self._loss(scores, relevance)

        if n is not None:
            mask = _torch.arange(1, raw_loss.shape[1] + 1).repeat(raw_loss.shape[0], 1).unsqueeze(2)
            mask = mask.to(n.device)
            n = n.unsqueeze(1)
            nprime = n.repeat(1, raw_loss.shape[1]).unsqueeze(2)


            raw_loss[mask > nprime] = 0

        loss = raw_loss.view(raw_loss.shape[0], -1).sum(1)

        ######

        return loss.mean()


class MatchLoss(_torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, grad_syn, grad_real):
        pass


class MSEGradientMatchLoss(MatchLoss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, grad_syn, grad_real):
        """
        The code is from here: https://github.com/amazon-research/doscond/blob/ef98e6c0013f3e47768c3154dfd6d43c45483d2b/utils.py#L162
        """
        dis = _torch.tensor(0.0, requires_grad=True)
        
        
        gw_real_vec = []
        gw_syn_vec = []
        
        for ig in range(len(grad_real)):
            gw_real_vec.append(grad_real[ig].reshape((-1)))
            gw_syn_vec.append(grad_syn[ig].reshape((-1)))
        gw_real_vec = _torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = _torch.cat(gw_syn_vec, dim=0)
        dis = _torch.sum((gw_syn_vec - gw_real_vec)**2) / _torch.sum((gw_real_vec)**2) 

        return dis


class CosGradientMatchLoss(MatchLoss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, grad_syn, grad_real):
        """
        The code is from here: https://github.com/amazon-research/doscond/blob/ef98e6c0013f3e47768c3154dfd6d43c45483d2b/utils.py#L162
        """
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_real)):
            gw_real_vec.append(grad_real[ig].reshape((-1)))
            gw_syn_vec.append(grad_syn[ig].reshape((-1)))
        gw_real_vec = _torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = _torch.cat(gw_syn_vec, dim=0)
        dis = 1 - _torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (_torch.norm(gw_real_vec, dim=-1) * _torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        return dis
