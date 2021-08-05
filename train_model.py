import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt


class StyleClassifier(nn.Module):
        def __init__(self):
                super(StyleClassifier, self).__init__()
                self.fc1 = nn.Linear(2137, 512)
                self.bn1  = nn.BatchNorm1d(512)              
                self.fc2 = nn.Linear(512, 128)
                self.dp2 = nn.Dropout(p=0.25) 
                self.fc3 = nn.Linear(128, 27)
                
        def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)  
                x = self.fc2(x)
                x = self.dp2(x)
                x = self.fc3(x)
                return F.log_softmax(x, -1)

class StyleClassifier_HOG_LBP_VGG(nn.Module):
        def __init__(self):
                super(StyleClassifier_HOG_LBP_VGG, self).__init__()
                self.fc1 = nn.Linear(3545, 512)
                self.bn1  = nn.BatchNorm1d(512)              
                self.fc2 = nn.Linear(512, 128)
                self.dp2 = nn.Dropout(p=0.25) 
                self.fc3 = nn.Linear(128, 27)
                
        def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)  
                x = self.fc2(x)
                x = self.dp2(x)
                x = self.fc3(x)
                return F.log_softmax(x, -1)
    
    
def train(Net, X_train, y_train, optimizer, epoch):
        Net.train()
        for i in range(0, len(y_train), batch_size):
                X = X_train[i:i + batch_size].cuda()
                y = y_train[i:i + batch_size].cuda()

                def closure():
                        optimizer.zero_grad()
                        output = Net.forward(X)
                        loss = loss_fun(output, y)
                        loss.backward()   
                        if i % 1000 == 0:
                                print('Train epoch: %s, {%s}, Loss: %s' % (epoch, i, loss.item()))
                        return loss
                optimizer.step(closure)

CLASSES = [
        'Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern',
        'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism',
        'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance',
        'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism',
        'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
        'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism',
        'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e'
]
class_dict = {cls: i for i, cls in enumerate(CLASSES)}

print('Load data:', end='')
X = np.load('Models/database_HOG_LBP_VGG_CONV.npy')
target = np.array([class_dict[i] for i in pd.read_csv('Models/target.csv')['target']])
print('OK')

X = torch.tensor(X, dtype=torch.float).cuda()
target = torch.tensor(target, dtype=torch.long).cuda()

n_epoch = 25
batch_size = 2500

print('Create models:', end='')
model_1 = StyleClassifier_HOG_LBP_VGG().cuda()
optimizer_1 = optim.Adam(model_1.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss().cuda()
print('OK')

indices = np.arange(len(X))
np.random.shuffle(indices)
X_train, y_train = X[indices], target[indices]

print('Start train HOG_LBP_VGG model')
for epoch in range(n_epoch):
        train(model_1, X_train[:,:3545], y_train, optimizer_1, epoch)

torch.save(model_1, 'output/model_HOG_LBP_VGG.pt')