import torch
import torch.nn.functional as F

from livelossplot import PlotLosses

from models.gcn import GCN

def eval_gcn_model(model, features, adj, labels, idx_val):

    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()

    preds = output[idx_val].detach().max(1)[1].type_as(labels[idx_val])
    correct = preds.eq(labels[idx_val]).double()
    correct = correct.sum().cpu().item()
    acc_val = correct / len(labels[idx_val])

    return loss_val, acc_val