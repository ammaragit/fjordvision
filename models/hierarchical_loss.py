import torch.nn.functional as F

def hierarchical_loss(outputs, targets, weights=None):
    """
    Paper-style joint hierarchical loss
    """
    if weights is None:
        weights = {
            "binary": 1.0,
            "class": 1.0,
            "genus": 1.0,
            "species": 1.0
        }

    loss = 0.0
    for level in outputs:
        loss += weights[level] * F.cross_entropy(
            outputs[level],
            targets[level]
        )

    return loss
