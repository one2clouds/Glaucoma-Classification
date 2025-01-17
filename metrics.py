from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torchvision
import torch 
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt 
from torchvision.transforms import transforms



transform = transforms.Compose([
    transforms.RandomCrop(size=(512,512), pad_if_needed=True), 
    transforms.ToTensor()
])


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], fontsize=15)


def data_imbalance_check(train_path):
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    # val_dataset = torchvision.datasets.ImageFolder(val_path)
    # test_dataset = torchvision.datasets.ImageFolder(test_path)

    # combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    with open('shape_of_images.txt', 'w') as file:
        file.write('Shape of different image files \n')
        for data in train_dataset:
            if data[0].shape[1] > 512:
                file.write(str(data[0].shape) + '\n')
                # print(data[0].shape)

    labels = torch.tensor([label for _, label in train_dataset])

    len_glaucoma = (labels == 1).sum().item()
    len_normal = (labels == 0).sum().item()

    plt.figure(figsize=(8, 5))
    plt.bar(['Glaucoma', 'Normal'], [len_glaucoma, len_normal], color=['orange', 'green'])
    addlabels(['Glaucoma', 'Normal'], [len_glaucoma, len_normal])

    plt.xlabel('Categories')
    plt.ylabel('Number of Samples')
    plt.title('Number of Glaucoma vs. Normal Samples')
    plt.savefig("data_imbalance_check.jpg")

