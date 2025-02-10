# Import the modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import os

import copy

import torch.nn.functional as F
import re
import yaml


class WSC_1det_struct(nn.Module):
    def __init__(self, 
                 encoder_struct):
        super(WSC_1det_struct, self).__init__()
        
        self.dep = len(encoder_struct)
        self.encoder_struct = torch.IntTensor(encoder_struct)

        self.relu = nn.ReLU()  # 激活函数
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()


        for i in range(self.dep-1):
            layer = nn.Linear(self.encoder_struct[i], self.encoder_struct[i+1])
            nn.init.kaiming_normal_(layer.weight)
            self.layers.append(layer)
            self.norm_layers.append(nn.BatchNorm1d(self.encoder_struct[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.dep-2:
                x = self.relu(x)
                # x = nn.BatchNorm1d(self.encoder_struct[i+1])(x)
                x = self.norm_layers[i](x)

        return x


def return_model_with_least_valloss(model_list):
    # This function returns the model with the least valloss in the model_list
    # valloss_list = np.empty((0))
    # epoch_list = np.empty((0))
    # extracting the epochs the model is using
    # print(model_list.keys())
    epoch_list = [int(re.search(r'(\d+)valloss', item).group(1)) for item in model_list.keys() if isinstance(item, str) and 'valloss' in item]
    valloss_list = [model_list[key] for key in [item for item in model_list.keys() if isinstance(item, str) and 'valloss' in item]]
    # print(epoch_list[valloss_list.index(min(valloss_list))])
    return(model_list[epoch_list[valloss_list.index(min(valloss_list))]])


# device here need a method to do an overall definition
def trainSeriesSupC_struct(datasets, config_dict):
# datasets: multiple datasets, datasets[0, 1, ...] are for the 1st, 2nd, ... class
# datasets should have the keys to be the integres 0, 1, 2, ...
# config dict is the dictionary containing the configuration of the training. The structure is like the one under 'Weakly_Supervised' tag.
    
    assert config_dict['Training_scheme']['Early_stopping_applied']
    # Haven't implemented the method without early stopping yet
    
    epochs_wsc = config_dict['Training_scheme']['Total_epochs']
    batch_size_wsc = config_dict['Training_scheme']['Batch_size']
    lr_wsc = config_dict['Training_scheme']['Learning_rate']
    
    least_epochs = config_dict['Training_scheme']['Least_epochs']
    epochs_interval = config_dict['Training_scheme']['Epochs_interval']
    # device = config_dict['Training_scheme']['Epochs_interval']
    
    rTrain = config_dict['Training_scheme']['Ratio_train']
    rTest = config_dict['Training_scheme']['Ratio_test']
    struct = config_dict['Model_params']['Model_struct_before_final_layer'] + [len(config_dict['Model_params']['Class_type'])]
    
    fig_save_path = os.path.join(config_dict['Training_scheme']['Output_dir'], config_dict['Training_scheme']['Output_file_infix']+config_dict['Training_scheme']['Output_file_suffix']+'.png')
    model_save_path = os.path.join(config_dict['Training_scheme']['Output_dir'], config_dict['Training_scheme']['Output_file_infix']+config_dict['Training_scheme']['Output_file_suffix']+'.pt')
    
    
    model_list = {}
    
    wsc = WSC_1det_struct(struct).to(device)
    nparam = sum(p.numel() for p in wsc.parameters() if p.requires_grad)
    
    Nclass = len(datasets)
    wclass = torch.FloatTensor([len(datasets[0])/len(datasets[i]) for i in range(Nclass)]).to(device)
    nTotal = {}
    nTrain = {}
    nTest = {}
    nbkg_train = 0
    nsig_train = 0
    for i in np.arange(Nclass):
        nTotal[i] = datasets[i].shape[0]
        nTrain[i] = int(rTrain*nTotal[i])
        nTest[i] = int(rTest*nTotal[i])
        if i < 2:
            nbkg_train += nTrain[i]
        else:
            nsig_train += nTrain[i]

    X_train = np.concatenate([datasets[i][:nTrain[i]] for i in range(Nclass)])
    X_test = np.concatenate([datasets[i][-nTest[i]:] for i in range(Nclass)])
    X_validation = np.concatenate([datasets[i][nTrain[i]:-nTest[i]] for i in range(Nclass)])

    Y_train = np.concatenate([i*np.ones(nTrain[i], dtype=int) for i in np.arange(Nclass)])
    Y_validation = np.concatenate([i*np.ones(nTotal[i]-nTrain[i]-nTest[i], dtype=int) for i in np.arange(Nclass)])

    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(Y_train).to(device))
    validation_dataset = TensorDataset(torch.FloatTensor(X_validation).to(device), torch.LongTensor(Y_validation).to(device))
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size_wsc, shuffle=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validation_dataset, batch_size=batch_size_wsc, shuffle=True, drop_last=True)

    # associated with a direct sum of probability, see below
    optimizer = optim.Adam(wsc.parameters(), lr=lr_wsc)
    loss_func = nn.CrossEntropyLoss(weight=wclass).to(device)
    loss_train = np.empty(epochs_wsc)
    loss_validation = np.empty(epochs_wsc)

    for epoch in range(epochs_wsc):
        wsc.train()
        for batchidx, (x, y) in enumerate(trainDataLoader):
            yprime = wsc(x)
            loss = loss_func(yprime, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        wsc.eval()
        with torch.no_grad():
            val_loss = 0
            for batchidx, (x, y) in enumerate(validationDataLoader):
                yprime = wsc(x)
                lossVal = loss_func(yprime, y)
                val_loss += lossVal.item()

            val_loss /= len(validationDataLoader)
            
        loss_train[epoch] = loss.item()
        loss_validation[epoch] = val_loss
        
        if ((epoch+1) > least_epochs) and ((epoch+1) % epochs_interval == 0):
            model_list[(epoch+1)] = copy.deepcopy(wsc.cpu().eval())
            model_list[str(epoch+1)+'valloss'] = val_loss
            wsc.to(device)
        
    wsc.to(device).eval()
    
    _, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(loss_train)
    ax[0].plot(loss_validation)
    foo = ax[1].hist(nn.Softmax(dim=1)(wsc(torch.FloatTensor(X_train).to(device))).cpu().detach().numpy().dot([0., 0.]+[1.]*(Nclass-2)), range=(0, 1), bins=20, density=True, histtype="step")
    foo = ax[1].hist(nn.Softmax(dim=1)(wsc(torch.FloatTensor(X_test ).to(device))).cpu().detach().numpy().dot([0., 0.]+[1.]*(Nclass-2)), range=(0, 1), bins=20, density=True, histtype="step")

    # plt.show()
    plt.savefig(fig_save_path)
    plt.close()
    
    model = return_model_with_least_valloss(model_list)
    torch.save(model, model_save_path)
    
    return 0


# Debugging part
if __name__ == "__main__":
    device = 'cpu'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    dataset_trial = torch.load('/home/app/test_data/trial.json', weights_only=False)
    
    
    print(dataset_trial.keys())
    
    config['Weakly_Supervised']['Training_scheme']['Output_dir'] = config['Full_pipeline']['Training_scheme']['Output_dir']
    
    trainSeriesSupC_struct(dataset_trial, config_dict=config['Weakly_Supervised'])
    