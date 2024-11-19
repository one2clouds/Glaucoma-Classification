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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from get_data import get_data
from metrics import data_imbalance_check

import argparse


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_needed_metrics(labels, predicted): 
    assert isinstance(labels, list)
    assert isinstance(predicted, list)
    
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, zero_division=0.0)
    recall = recall_score(labels, predicted, zero_division=0.0)
    f1 = f1_score(labels, predicted, zero_division=0.0)
    
    return accuracy, precision, recall, f1



def train(model, train_loader, val_loader, loss_func, optimizer, num_epochs):
    best_val_recall = -100
    for epoch in range(num_epochs):
        model.train()
        acc_train_epoch, precision_train_epoch, recall_train_epoch, f1_train_epoch  = [], [], [], []
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = nn.Sigmoid()(model(inputs))     
            
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
        
            _, predicted_train = torch.max(outputs, 1)
            

            acc_batch_train, precision_batch_train, recall_batch_train, f1_batch_train = get_needed_metrics(labels.cpu().detach().tolist(), predicted_train.cpu().detach().tolist())
        
            acc_train_epoch.append(acc_batch_train)
            precision_train_epoch.append(precision_batch_train)
            recall_train_epoch.append(recall_batch_train)
            f1_train_epoch.append(f1_batch_train)
        
        # Validating the model
        model.eval()
        acc_val_epoch, precision_val_epoch, recall_val_epoch, f1_val_epoch  = [], [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Testing {epoch}/{num_epochs}', unit='batch'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nn.Sigmoid()(model(inputs))     
                val_loss = loss_func(outputs, labels)
                _, predicted_val = torch.max(outputs, 1)

                acc_batch_val, precision_batch_val, recall_batch_val, f1_batch_val = get_needed_metrics(labels.cpu().detach().tolist(), predicted_val.cpu().detach().tolist())

                acc_val_epoch.append(acc_batch_val)
                precision_val_epoch.append(precision_batch_val)
                recall_val_epoch.append(recall_batch_val)
                f1_val_epoch.append(f1_batch_val)
                
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss.item():.4f}, '
            f'Train Accuracy: {torch.tensor(acc_train_epoch).mean() * 100:.2f}%, '
            f'Train Precision: {torch.tensor(precision_train_epoch).mean() * 100:.2f}%, '
            f'Train Recall: {torch.tensor(recall_train_epoch).mean() * 100:.2f}%, '
            f'Train F1: {torch.tensor(f1_train_epoch).mean() * 100:.2f}%, '

            f'Val Loss: {val_loss.item():.4f}, '
            f'Val Accuracy: {torch.tensor(acc_val_epoch).mean() * 100:.2f}%, '
            f'Val Precision: {torch.tensor(precision_val_epoch).mean() * 100:.2f}%, '
            f'Val Recall: {torch.tensor(recall_val_epoch).mean() * 100:.2f}%, '
            f'Val F1: {torch.tensor(f1_val_epoch).mean() * 100:.2f}%')
        
        if torch.tensor(recall_val_epoch).mean() > best_val_recall:
            best_val_recall = torch.tensor(recall_val_epoch).mean()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                }, 'model_best.pth')
            print(f"Best model saved, Recall - {torch.tensor(recall_val_epoch).mean()} ")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            }, 'model_recent.pth')
        
    return model, epoch, optimizer, train_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    args = parser.parse_args()

    # train_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/train/'
    # val_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/validation/'
    # test_path = '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/test/'

    data_imbalance_check(args.train_path, args.val_path, args.test_path)


    train_dataloader = get_data(args.train_path, get_path=False, shuffle=True)
    val_dataloader = get_data(args.val_path, get_path=False, shuffle=True)

    root=pathlib.Path(args.train_path)
    classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])


    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(1792, len(classes)) 
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100

    model, optimizer, epoch, train_loss = train(model, train_dataloader, val_dataloader, loss_func, optimizer, num_epochs)

    


# python3 train.py --train_path '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/train/' --val_path '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/validation/' --test_path '/mnt/Enterprise2/shirshak/Glaucoma_Dataset_eyepacs_airogs_lightv2/eyepac-light-v2-512-jpg/test/'










