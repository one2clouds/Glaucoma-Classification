from efficientnet_pytorch import EfficientNet
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import torch 
from tqdm import tqdm 
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pathlib
import matplotlib.pyplot as plt
from glob import glob 
import cv2
import pandas as pd 
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from get_data import get_data
from metrics import data_imbalance_check
import argparse
from PIL import Image


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_needed_metrics(labels, predicted_test): 
    assert isinstance(labels, list)
    assert isinstance(predicted_test, list)
    
    accuracy = accuracy_score(labels, predicted_test)
    precision = precision_score(labels, predicted_test, zero_division=0.0)
    recall = recall_score(labels, predicted_test, zero_division=0.0)
    f1 = f1_score(labels, predicted_test, zero_division=0.0)
    
    return accuracy, precision, recall, f1



def test(test_loader, loaded_model, ):
    image_paths = []
    x_image, y_labels, y_pred = torch.tensor(0), torch.tensor(0), torch.tensor(0)
    loaded_model.eval()
    acc_test_epoch, precision_test_epoch, recall_test_epoch, f1_test_epoch  = [], [], [], []
    with torch.no_grad():
        for inputs, labels, image_path in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = nn.Sigmoid()(loaded_model(inputs)) 
            _, predicted_test = torch.max(outputs, 1)

            # print(outputs)

            # print(labels)
            # print(predicted_test)
            
            acc_batch_test, precision_batch_test, recall_batch_test, f1_batch_test = get_needed_metrics(labels.cpu().detach().tolist(), predicted_test.cpu().detach().tolist())
        
            acc_test_epoch.append(acc_batch_test)
            precision_test_epoch.append(precision_batch_test)
            recall_test_epoch.append(recall_batch_test)
            f1_test_epoch.append(f1_batch_test)

            x_image = inputs.cpu().detach() if x_image.equal(torch.tensor(0)) else torch.cat((x_image, inputs.cpu().detach()), dim=0)
            y_labels = labels.cpu().detach() if y_labels.equal(torch.tensor(0)) else torch.cat((y_labels, labels.cpu().detach()), dim=0)
            y_pred = predicted_test.cpu().detach() if y_pred.equal(torch.tensor(0)) else torch.cat((y_pred, predicted_test.cpu().detach()), dim=0)
            image_paths.extend(image_path)
        
        print(
                f'Test Accuracy : {torch.tensor(acc_test_epoch).mean()} , '
                f'Test Precision : {torch.tensor(precision_test_epoch).mean()} , '
                f'Test Recall : {torch.tensor(recall_test_epoch).mean()} , '
                f'Test F1  : {torch.tensor(f1_test_epoch).mean()} , '
                )
        
        return x_image, y_labels, y_pred, image_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    if args.data == 'eyepacs_airogs':
        test_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/test/'
    elif args.data == 'dristi_gs1':
        test_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_Drishti-GS/Drishti-GS1_processed_train_test_overall/'
    else: 
        print("data is not given")
        exit()
    
    test_dataloader = get_data(test_path, args.data, get_path=True, shuffle=True)
    root=pathlib.Path(test_path)
    classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

    
    if args.model == 'efficient_net':
        loaded_model = EfficientNet.from_pretrained('efficientnet-b4')
        loaded_model._fc = nn.Linear(1792, len(classes))
        checkpoint = torch.load('/home/shirshak/Glaucoma_Efficientnet_simple/model_least_val_loss.pth', weights_only=True)
    elif args.model == 'resnet':
        loaded_model = torchvision.models.resnet50(pretrained=True)
        loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(classes))
        print("More code needed !! checkpoint need to be added for resnet model")
        exit()
    
    loaded_model = loaded_model.to(device)
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=1e-4)

    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']


    x_image, y_labels, y_pred, image_paths = test(test_dataloader, loaded_model)

    confusion_matrix_chart = confusion_matrix(y_labels.tolist(), y_pred.tolist())
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_chart, display_labels = ['Not Glaucoma', 'Glaucoma'])
    
    cm_display.plot()
    plt.savefig('confusion_matrix_my_my.png', dpi=500)
    plt.close()

    x_image = list(torch.unbind(x_image, dim=0))
    y_labels = y_labels.tolist()
    y_pred = y_pred.tolist()

    right_cases = [(x_w, y_w, yp_w, img_p) for x_w, y_w, yp_w, img_p in zip(x_image, y_labels, y_pred, image_paths) if y_w == yp_w]
    wrong_cases = [(x_w, y_w, yp_w, img_p) for x_w, y_w, yp_w, img_p in zip(x_image, y_labels, y_pred, image_paths) if y_w != yp_w]
    labels = ['Glaucoma', 'Not Glaucoma']


    for count, right_case in enumerate(right_cases[:20]):
        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        
        ax[0].imshow(Image.open(right_case[3]))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[0].text(0.5, -0.1, f'Real : {labels[right_case[1]]}', ha='center', va='center', transform=ax[0].transAxes, fontsize=10)

        ax[1].imshow(torchvision.transforms.ToPILImage()(right_case[0]))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')
        ax[1].text(0.5, -0.1, f'Predicted : {labels[right_case[2]]}', ha='center', va='center', transform=ax[1].transAxes, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"/home/shirshak/Glaucoma_Efficientnet_simple/glaucoma_test_images/correct{count}.jpg")
        plt.close()

    for count,wrong_case in enumerate(wrong_cases[:20]):
        fig, ax = plt.subplots(1,2, figsize=(5,5))

        ax[0].imshow(Image.open(wrong_case[3]))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[0].text(0.5, -0.1, f'Real : {labels[wrong_case[1]]}', ha='center', va='center', transform=ax[0].transAxes, fontsize=10)

        ax[1].imshow(torchvision.transforms.ToPILImage()(wrong_case[0]))
        ax[1].set_title('Transformed Image')
        ax[1].axis('off')
        ax[1].text(0.5, -0.1, f'Predicted : {labels[wrong_case[2]]}', ha='center', va='center', transform=ax[1].transAxes, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"/home/shirshak/Glaucoma_Efficientnet_simple/glaucoma_test_images/wrong{count}.jpg")
        plt.close()





# python3 test.py --data eyepacs_airogs --model efficient_net
# python3 test.py --data dristi_gs1