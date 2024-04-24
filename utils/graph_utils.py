import torch
import numpy as np
from tqdm import tqdm

def generate_feature_relationship_matrix(trainset, testset, cnn_model, device):

    features = []
    labels = []
    files = []

    train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size = 1,
                                           shuffle = False,
                                           pin_memory = True,
                                           num_workers = 2)
    
    test_loader = torch.utils.data.DataLoader(testset,
                                           batch_size = 1,
                                           shuffle = False,
                                           pin_memory = True,
                                           num_workers = 2)
  
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    print("Generating train features:")
    for i, (img, label) in enumerate(tqdm(train_loader)):
        img = img.to(device)
        features.append(cnn_model(img)[0].detach().cpu())
        labels.append(label)

    idx_train = range(len(labels))

    print("Generating test features:")
    for i, (img, label) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        features.append(cnn_model(img)[0].detach().cpu())
        labels.append(label)

    idx_val = range(len(idx_train), len(labels))
    
    features = torch.stack(features)
    labels = torch.Tensor(labels)
    
    dist = np.ones((features.shape[0], features.shape[0]))
    
    print("Calculating pairwise distances:")
    for i in tqdm(range(features.shape[0])):
        curr_feature = features[i] 

        for j in range(i+1, features.shape[0]):
            other_feature = features[j] 
                    
            dist[i, j] = torch.sum((curr_feature - other_feature)**2).item()
            dist[j, i] = dist[i, j] 
    
    return dist, features, labels, idx_train, idx_val

def get_deg(adj):
    deg = np.zeros(adj.shape)
    
    for i in range(adj.shape[0]):
        deg[i, i]=np.sum(adj[i])
        
    return deg
    
def get_laplacian(adj):
    I = np.eye(adj.shape[0])
    D = get_deg(adj)
    D_= np.linalg.inv(np.mat(D**(0.5))) #np.mat(D**(0.5)).I
    L = np.dot(np.dot(D_, adj),D_)
    return L