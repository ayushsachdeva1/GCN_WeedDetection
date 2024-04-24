import torch
import torch.nn.functional as F

from livelossplot import PlotLosses

from models.gcn import GCN
from utils.eval_gcn import eval_gcn_model

def train_gcn(features, adj, labels, idx_train, idx_val, device, epochs=100, lr=0.0001, weight_decay=0, save_name=""):

    model = GCN(feat_dim=features.shape[1], num_class=int(labels.max().item() + 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)

    liveloss = PlotLosses()
    best_accuracy = 0
    logs = {}

    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.type(torch.LongTensor).to(device)

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        preds = output[idx_train].detach().max(1)[1].type_as(labels[idx_train])
        correct = preds.eq(labels[idx_train]).double()
        correct = correct.sum().cpu().item()
        acc_train = correct / len(labels[idx_train])

        loss_train.backward()
        optimizer.step()
        # scheduler.step()

        loss_val, acc_val = eval_gcn_model(model, features, adj, labels, idx_val)

        logs['loss'] = loss_train.item()
        logs['accuracy'] = acc_train

        logs['val_loss'] = loss_val
        logs['val_accuracy'] = acc_val

        # Save the parameters for the best accuracy on the validation set so far.
        if logs['val_accuracy'] > best_accuracy:
            best_accuracy = logs['val_accuracy']
            torch.save(model.state_dict(), 'checkpoints/best_' + save_name + 'gcn_model_so_far.pth')

        # Update the plot with new logging information.
        liveloss.update(logs)
        liveloss.send()