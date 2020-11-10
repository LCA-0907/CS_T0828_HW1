from __future__ import print_function
from __future__ import division
import torch
import torchvision
from torchvision import models, datasets, transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from pandas import Series, DataFrame
import time
import os
import copy
from PIL import Image
from torch.utils.data import Dataset
import math
import sys
import csv

# functions
def train_model(model, dataloaders, loss_fun, optimizer, num_epochs = 25, is_inception = False, mode= 'default'):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_acc_history = []
    print('training mode:', mode)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        scheduler.step(epoch)
        print(optimizer.param_groups[0]['lr'])
        #for phase in ['training_data', 'val_data']:
        for phase in ['training_data']:
            print("phase:{}".format(phase))
            model.train()
            running_loss = 0.0
            running_corrects = 0.0 

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if is_inception and phase =='training_data':
                    outputs = model(inputs)
                    loss = loss_fun(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = loss_fun(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                if phase == 'training_data':
                    loss.backward()
                    optimizer.step()

                running_loss = running_loss + (loss.item() * inputs.size(0))
                running_corrects = running_corrects + torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            loss_acc_history.append([epoch_loss, epoch_acc])

            print('Loss: {:.4f}'.format(epoch_loss))
            if mode == '-loadmodel':
                torch.save(model.state_dict(), './models/Epoch+' +str(epoch)+'.pkl')
            else: #default
                torch.save(model.state_dict(), './models/Epoch' +str(epoch)+'.pkl')

        print("--phase loop--")

    f = open('loss_acc.csv', 'w')
    wcsv = csv.writer(f)
    wcsv.writerows(loss_acc_history)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        '''
         resnet50
        '''
        model_ft = models.resnet50(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_features=2048, out_features=196)
        nn.init.kaiming_normal_(model_ft.fc.weight, mode='fan_in')
        #input_size = 448
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# Dataloader
train_preprocess = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomCrop((448,448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.0126, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def default_loader(path):
    img_pil =  Image.open(path).convert('RGB')
    #img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

def train_loader(path):
    img_pil =  Image.open(path).convert('RGB')
    #img_pil = img_pil.resize((224,224))
    img_tensor = train_preprocess(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, images_file, number_target, loader=default_loader):
        self.images = images_file
        self.target = number_target
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

# main start
if __name__ == "__main__":
    data_path = "./"
    num_class = 196 #196 labels
    batch_size = 32
    num_epochs = 30
    feature_extract = True
    #input_size = 224 #resnet
    model_name = "resnet"
    training_mode = 'default'
    
    # initialize model
    Res50 = initialize_model("resnet", num_class, feature_extract, use_pretrained=True)

    # Detect if GPU available
    device = torch.device("cuda:0")
    Res50 = nn.DataParallel(Res50, device_ids=[0,1,2])
    Res50.to(device)
    params_to_update = Res50.parameters()

    if len(sys.argv) == 3:
        if sys.argv[1] == 'loadmodel':
            Res50.load_state_dict(torch.load(sys.argv[2]))
            print(Res50)
            training_mode = '-loadmodel'
        elif sys.argv[1] == 'test':
            print("mode: testing")
            Res50.eval()
            # load label
            ddf = pd.read_csv("label_id.csv", header=None, index_col=1, squeeze=True)
            dic = ddf.to_dict()
            print(dic) 
            # load model
            Res50.load_state_dict(torch.load(sys.argv[2]))
            f = open('output.csv', 'w')
            wcsv = csv.writer(f)
            wcsv.writerow(['id', 'label'])
            # testing
            for test_path in os.listdir('./testing_data/testing_data'):
                img = default_loader(os.path.join('./testing_data/testing_data', test_path))
                img =  img.unsqueeze(0)
                output = Res50(img)
                output = output[0].tolist()
                pred = output.index(max(output))

                # write to csv
                img_str = str(os.path.splitext(test_path)[0])
                prediction_str = str(dic[pred])

                #print(img_str, prediction_str)
                wcsv.writerow([img_str, prediction_str])
            
            exit()
        else:
            print("Wrong mode.")
            exit()
         
    # load train.csv
    df = pd.read_csv('training_labels.csv', dtype=str)
    label = df['label']
    # load dictionary
    if os.path.isfile('label_id.csv'):
        ddf = pd.read_csv("label_id.csv", header=None, index_col=1, squeeze=True)
        invdic = ddf.to_dict()
        dic = {v: k for k, v in invdic.items()}
    else:
        ## for labels, make a dictionary
        label_set = set(Series.to_numpy(label))
        print(len(label_set)) #label num = 196
        label_196_list = list(label_set)
        dic = {}
        for i in range(len(label_set)):
            dic[label_196_list[i]] = i
        label_df = DataFrame.from_dict(dic, orient="index")
        label_df.to_csv("label_id.csv")
        ddf = pd.read_csv("label_id.csv")
    print(dic)                
    ## for ids
    file = Series.to_numpy(df['id'])
    file = [str(i) + '.jpg' for i in file]
    file = [os.path.join("./training_data/training_data", i) for i in file]
    file_train = file[:]
    print(len(file_train))
    # np.save("file_train.npy", file_train)

    ## for labels
    label_series = Series.to_numpy(label)
    number = []
    for i in range(len(label_series)):
        number.append(dic[label_series[i]])
    number = np.array(number)
    number_train = number[:]
    print(number_train.shape)
    # np.save("number_train.npy", number_train)

    # load data
    print("Initializing Datasets and Dataloaders...")

    training_data = trainset(file_train, number_train, loader=train_loader)

    trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size ,shuffle=True, num_workers=3)
    dataloaders = {'training_data': trainloader}
    optimizer_ft = torch.optim.SGD(params_to_update, lr = 0.005, momentum =0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size = 1, gamma = 0.99)
    
    # loss function
    loss_fun = nn.CrossEntropyLoss()

    #Train
    Trained_model, hist = train_model(Res50, dataloaders, loss_fun, optimizer_ft, num_epochs = num_epochs, mode= training_mode, is_inception=(model_name == "inception"))