from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torchvision
import torch 
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt 



def data_imbalance_check(train_path, val_path, test_path):

    train_dataset = torchvision.datasets.ImageFolder(train_path)
    val_dataset = torchvision.datasets.ImageFolder(val_path)
    test_dataset = torchvision.datasets.ImageFolder(test_path)

    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    labels = torch.tensor([label for _, label in combined_dataset])

    len_glaucoma = (labels == 1).sum().item()
    len_normal = (labels == 0).sum().item()

    plt.figure(figsize=(8, 5))
    plt.bar(['Glaucoma', 'Normal'], [len_glaucoma, len_normal], color=['orange', 'green'])
    plt.xlabel('Categories')
    plt.ylabel('Number of Samples')
    plt.title('Number of Glaucoma vs. Normal Samples')
    plt.savefig("data_imbalance_check.jpg")

