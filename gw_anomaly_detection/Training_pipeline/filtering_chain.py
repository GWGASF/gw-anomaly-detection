import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io

import torch.nn.functional as F
import os
import yaml

from gw_anomaly_detection.event_preprocessing.dataset_preprocess import read_from_directory


class AE_1det_struct(nn.Module):
    def __init__(self, 
                 encoder_struct):
        super(AE_1det_struct, self).__init__()
        
        self.dep = len(encoder_struct)
        self.encoder_struct = torch.IntTensor(encoder_struct)

        self.relu = nn.ReLU()  # 激活函数
        self.encoder_layers = nn.ModuleList()
        self.norm_encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.norm_decoder_layers = nn.ModuleList()

        # print(self.encoder_struct.devide)
        

        for i in range(self.dep-1):
            layer = nn.Linear(self.encoder_struct[i], self.encoder_struct[i+1])
            nn.init.kaiming_normal_(layer.weight)
            self.encoder_layers.append(layer)
            self.norm_encoder_layers.append(nn.BatchNorm1d(self.encoder_struct[i+1]))

            layer = nn.Linear(self.encoder_struct[self.dep-1-i], self.encoder_struct[self.dep-2-i])
            nn.init.kaiming_normal_(layer.weight)
            self.decoder_layers.append(layer)
            self.norm_decoder_layers.append(nn.BatchNorm1d(self.encoder_struct[self.dep-2-i]))

    def forward(self, x):
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.relu(x)
            # x = nn.BatchNorm1d(self.encoder_struct[i+1])(x)
            x = self.norm_encoder_layers[i](x)
            
        encoded = x;

        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if(i < self.dep-2):
                x = self.relu(x)
                # x = nn.BatchNorm1d(self.encoder_struct[self.dep-2-i])(x)
                x = self.norm_decoder_layers[i](x)

        decoded = nn.Sigmoid()(x)

        return encoded, decoded

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

def trainAE_struct(dataset, struct, config_dict):
    
    rTrain = config_dict['Ratio_train']
    rTest = config_dict['Ratio_test']
    epochs = config_dict['Total_epochs']
    batch_size = config_dict['Batch_size']
    learning_rate = config_dict['Learning_rate']
    save_path = os.path.join(config_dict['Output_dir'], config_dict['Output_file_infix']+config_dict['Output_file_suffix']+'.png')
    
    nTotal = len(dataset)
    nTrain = int(rTrain * nTotal)
    nTest = int(rTest * nTotal)
    # logger.info("{} events into AE training. ".format(nTotal))

    X_train = dataset[:nTrain]
    X_test = dataset[-nTest:]
    X_validation = dataset[nTrain:-nTest]

    trainData = torch.FloatTensor(X_train)
    testData = torch.FloatTensor(X_test)
    validationData = torch.FloatTensor(X_validation)

    train_dataset = TensorDataset(trainData)
    test_dataset = TensorDataset(testData)
    validation_dataset = TensorDataset(validationData)

    print(trainData.shape)
    print(validationData.shape)

    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    autoencoder = AE_1det_struct(struct).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss().to(device)
    
    loss_train = np.empty(epochs)
    loss_validation = np.empty(epochs)

    for epoch in range(epochs):

        autoencoder.train()
        for batchidx, x in enumerate(trainDataLoader):
            x = x[0].to(device)
            encoded, decoded = autoencoder(x)
            loss_overall = loss_func(decoded, x)
            weighted_lossTrain = loss_overall

            optimizer.zero_grad()
            weighted_lossTrain.backward()
            optimizer.step()
            
        autoencoder.eval()
        with torch.no_grad():
            val_loss = 0
            for batchidx, x in enumerate(validationDataLoader):
                x = x[0].to(device)
                encoded, decoded = autoencoder(x)
                lossVal = loss_func(decoded, x)
                val_loss += lossVal.item()

            val_loss /= len(validationDataLoader)

        loss_train[epoch] = weighted_lossTrain.item()
        loss_validation[epoch] = val_loss
    
    autoencoder.to(device).eval()
    _, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(loss_train)
    ax[0].plot(loss_validation)
    
    dcd_train = autoencoder(torch.FloatTensor(X_train).to(device))[1].cpu().detach().numpy()
    err_train = np.mean((X_train-dcd_train)**2, axis=1)
    dcd_test = autoencoder(torch.FloatTensor(X_test).to(device))[1].cpu().detach().numpy()
    err_test = np.mean((X_test-dcd_test)**2, axis=1)
    foo = ax[1].hist(err_train, range=(0, max(err_train)), bins=50, density=True, histtype="step")
    foo = ax[1].hist(err_test, range=(0, max(err_train)), bins=50, density=True, histtype="step")

    plt.savefig(save_path)
    plt.close()
    # logger.info("training figure saved to "+save_path+". ")
            
    return autoencoder.cpu().eval()


# For WSC_2class, need maintenance
# def trainWSC_2class(dataset0_to_be_examined, dataset1_to_be_examined, struct, save_path):
# # dataset0: bkg set from AE
# # dataset1: identified signal from AE

#     wsc = WSC_1det_struct(struct).to(device)
#     trainable_params_WSC = sum(p.numel() for p in wsc.parameters() if p.requires_grad)
#     Nsize = 20
#     if len(dataset0_to_be_examined) > Nsize * trainable_params_WSC:
#         dataset0 = dataset0_to_be_examined[np.random.choice(len(dataset0_to_be_examined), Nsize * trainable_params_WSC, replace = False)]
#     else:
#         dataset0 = dataset0_to_be_examined
    
#     if len(dataset1_to_be_examined) > Nsize * trainable_params_WSC:
#         dataset1 = dataset1_to_be_examined[np.random.choice(len(dataset1_to_be_examined), Nsize * trainable_params_WSC, replace = False)]
#     else:
#         dataset1 = dataset1_to_be_examined

#     logger.info('{}, bkg events and {} signal events passed to WSC for training. '.format(len(dataset0), len(dataset1)))

#     nTotal0, nTotal1 = len(dataset0), len(dataset1);
#     nTrain0, nTrain1 = int(rTrain * nTotal0), int(rTrain * nTotal1)
#     nTest0 , nTest1  = int(rTest * nTotal0) , int(rTest * nTotal1)

#     X_train = np.concatenate((dataset0[:nTrain0], dataset1[:nTrain1]))
#     X_test = np.concatenate((dataset0[-nTest0:], dataset1[-nTest1:]))
#     X_validation = np.concatenate((dataset0[nTrain0:-nTest0], dataset1[nTrain1:-nTest1]))
    
#     Y_train = np.concatenate((np.zeros((nTrain0, 1)), np.ones((nTrain1, 1)))).flatten().astype(int)
#     Y_test = np.concatenate((np.zeros((nTest0, 1)), np.ones((nTest1, 1)))).flatten().astype(int)
#     Y_validation = np.concatenate((np.zeros((dataset0[nTrain0:-nTest0].shape[0], 1)), np.ones((dataset1[nTrain1:-nTest1].shape[0], 1)))).flatten().astype(int)

#     train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(Y_train).to(device))
#     validation_dataset = TensorDataset(torch.FloatTensor(X_validation).to(device), torch.FloatTensor(Y_validation).to(device))

#     trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     validationDataLoader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle = True, drop_last=True)

#     # wsc = WSClassifier_3class().to(device)
#     optimizer = optim.Adam(wsc.parameters(), lr=0.00005)
#     loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([nTrain0/nTrain1])).to(device)
    
#     loss_train = np.empty(epochs)
#     loss_validation = np.empty(epochs)

#     for epoch in range(epochs):
#         wsc.train()
#         for batchidx, (x, y) in enumerate(trainDataLoader):
#             yprime = wsc(x).flatten()
#             loss = loss_func(yprime, y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#         wsc.eval()
#         with torch.no_grad():
#             val_loss = 0
#             for batchidx, (x, y) in enumerate(validationDataLoader):
#                 yprime = wsc(x).flatten()
#                 lossVal = loss_func(yprime, y)
#                 val_loss += lossVal.item()

#             val_loss /= len(validationDataLoader)

#         loss_train[epoch] = loss.item()
#         loss_validation[epoch] = val_loss
        
#     wsc.cpu().eval()
    
#     _, ax = plt.subplots(1, 2, figsize=(14, 5))
#     ax[0].plot(loss_train)
#     ax[0].plot(loss_validation)
#     foo = ax[1].hist(nn.Sigmoid()(wsc(torch.FloatTensor(X_train))).detach().numpy().flatten(), range=(0, 1), bins=20, density=True, histtype="step")
#     foo = ax[1].hist(nn.Sigmoid()(wsc(torch.FloatTensor(X_test ))).detach().numpy().flatten(), range=(0, 1), bins=20, density=True, histtype="step")
    
#     plt.savefig(save_path)
#     plt.close()
#     logger.info("training figure saved to "+save_path)
    
#     return wsc.cpu().eval()

# SupC is already implemented in the WSC section. The WSC channel need further maintenance

# def trainSeriesSupC_struct(datasets, struct, config_dict):
# # datasets: multiple datasets, datasets[0, 1, ...] are for the 1st, 2nd, ... class
# # datasets should have the keys to be the integres 0, 1, 2, ...

#     epochs_wsc = 500
#     batch_size_wsc = 384
#     lr_wsc = 2e-5
    
#     least_epochs = 100
#     epochs_interval = 20
#     model_list = {}
    
#     wsc = WSC_1det_struct(struct).to(device)
#     nparam = sum(p.numel() for p in wsc.parameters() if p.requires_grad)
    
#     Nclass = len(datasets)
#     wclass = torch.FloatTensor([len(datasets[0])/len(datasets[i]) for i in range(Nclass)]).to(device)
#     nTotal = {}
#     nTrain = {}
#     nTest = {}
#     nbkg_train = 0
#     nsig_train = 0
#     for i in np.arange(Nclass):
#         nTotal[i] = datasets[i].shape[0]
#         nTrain[i] = int(rTrain*nTotal[i])
#         nTest[i] = int(rTest*nTotal[i])
#         if i < 2:
#             nbkg_train += nTrain[i]
#         else:
#             nsig_train += nTrain[i]

#     X_train = np.concatenate([datasets[i][:nTrain[i]] for i in range(Nclass)])
#     X_test = np.concatenate([datasets[i][-nTest[i]:] for i in range(Nclass)])
#     X_validation = np.concatenate([datasets[i][nTrain[i]:-nTest[i]] for i in range(Nclass)])

#     Y_train = np.concatenate([i*np.ones(nTrain[i], dtype=int) for i in np.arange(Nclass)])
#     Y_validation = np.concatenate([i*np.ones(nTotal[i]-nTrain[i]-nTest[i], dtype=int) for i in np.arange(Nclass)])

#     train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(Y_train).to(device))
#     validation_dataset = TensorDataset(torch.FloatTensor(X_validation).to(device), torch.LongTensor(Y_validation).to(device))
#     trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size_wsc, shuffle=True, drop_last=True)
#     validationDataLoader = DataLoader(dataset=validation_dataset, batch_size=batch_size_wsc, shuffle=True, drop_last=True)

#     # associated with a direct sum of probability, see below
#     optimizer = optim.Adam(wsc.parameters(), lr=lr_wsc)
#     loss_func = nn.CrossEntropyLoss(weight=wclass).to(device)
#     loss_train = np.empty(epochs_wsc)
#     loss_validation = np.empty(epochs_wsc)

#     for epoch in range(epochs_wsc):
#         wsc.train()
#         for batchidx, (x, y) in enumerate(trainDataLoader):
#             yprime = wsc(x)
#             loss = loss_func(yprime, y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#         wsc.eval()
#         with torch.no_grad():
#             val_loss = 0
#             for batchidx, (x, y) in enumerate(validationDataLoader):
#                 yprime = wsc(x)
#                 lossVal = loss_func(yprime, y)
#                 val_loss += lossVal.item()

#             val_loss /= len(validationDataLoader)
            
#         loss_train[epoch] = loss.item()
#         loss_validation[epoch] = val_loss
        
#         if ((epoch+1) > least_epochs) and ((epoch+1) % epochs_interval == 0):
#             model_list[(epoch+1)] = copy.deepcopy(wsc.cpu().eval())
#             model_list[str(epoch+1)+'valloss'] = val_loss
#             wsc.to(device)
        
#     wsc.to(device).eval()
    
#     _, ax = plt.subplots(1, 2, figsize=(14, 5))
#     ax[0].plot(loss_train)
#     ax[0].plot(loss_validation)
#     foo = ax[1].hist(nn.Softmax(dim=1)(wsc(torch.FloatTensor(X_train).to(device))).cpu().detach().numpy().dot([0., 0.]+[1.]*(Nclass-2)), range=(0, 1), bins=20, density=True, histtype="step")
#     foo = ax[1].hist(nn.Softmax(dim=1)(wsc(torch.FloatTensor(X_test ).to(device))).cpu().detach().numpy().dot([0., 0.]+[1.]*(Nclass-2)), range=(0, 1), bins=20, density=True, histtype="step")

#     plt.show()
#     plt.savefig(save_path)
#     plt.close()
    
#     return model_list

def Series_training(training_set_ae, config_dict):
    
    # Remember the training_set_ae is a dictionary with keys 0, 1, 2, ..., n, the 0 are pure H glitches and 1 are pure L glitches. all in 202 shape. 
    
    classnum = config_dict['Class_to_use']
    classkey = ['GlitchH_class', 'GlitchL_class'] + ['Class'+str(i+1) for i in range(classnum - 1)]
    classname = [config_dict[key]['Model_params']['Class_name'] for key in classkey]
    
    # Right now the cache part can only working for the AE
    use_cache = [config_dict[key]['Training_scheme']['Use_cache'] for key in classkey]
    generate_cache = [config_dict[key]['Training_scheme']['Generate_cache'] for key in classkey]
    train_wsc = [config_dict[key]['Training_scheme']['Train_wsc'] for key in classkey]
    ae_struct = [config_dict[key]['Model_params']['Model_struct_half'] for key in classkey]
    cutAE_ratio = [config_dict[key]['Training_scheme']['Cut_ratio'] for key in classkey]
    
    for key in classkey:
        config_dict[key]['Training_scheme']['Output_dir'] = config_dict['Output_dir']
        config_dict[key]['Training_scheme']['Output_file_infix'] = config_dict['Output_file_infix'] + config_dict[key]['Model_params']['Class_name'] + '_'
        config_dict[key]['Training_scheme']['Output_file_suffix'] = config_dict['Output_file_suffix']
    
    # fig_save_path_list = [os.path.join(config_dict[key]['Training_scheme']['Output_dir'], config_dict[key]['Training_scheme']['Output_file_infix']+config_dict[key]['Training_scheme']['Output_file_suffix']+'.png') for key in classkey]
    model_chain_save_path = os.path.join(config_dict['Output_dir'], config_dict['Output_file_infix']+config_dict['Output_file_suffix']+'.json')
    
    # Modification has to be made on the param of the AE training
    
    
    aes = {}
    cutAE = {}
    # models = {}
    # foo = torch.load(modelDir+"/glitch_AE_freq_new.json")
    # models['glitch_H'] = foo["H_101-10-20-10"].cpu().eval()
    # models['glitch_L'] = foo["L_101-10-10"].cpu().eval()
    # foo = torch.load(modelDir+"/noise_AE_freq_new.json")
    # models['noise'] = foo["2det_202-40-20"].cpu().eval()
    # foo = torch.load(modelDir+"/BBH_AE_freq.json")
    # models['BBH'] = foo["mixed_202-20-20"].cpu().eval()
    # foo = torch.load(modelDir+"/SG_AE_freq_new.json")
    # models['SGHF'] = foo["SGHF_2det_48-96_202-40"].cpu().eval()
    
    # If we have cached filtering chain, load it. 
    # Better have a cached dir, not the output one to avoid confusion
    if os.path.exists(model_chain_save_path):
        models = torch.load(model_chain_save_path, weights_only=False)
    
    # Glitch part comes first
    
    if use_cache[0]:
        aes['glitch_H'] = models['glitch_H']
    else:
        aes['glitch_H'] = trainAE_struct(training_set_ae[0][:,:101], ae_struct[0], config_dict[classkey[0]]['Training_scheme'])
    
    if use_cache[1]:
        aes['glitch_L'] = models['glitch_L']
    else:
        aes['glitch_L'] = trainAE_struct(training_set_ae[1][:,101:], ae_struct[1], config_dict[classkey[1]]['Training_scheme'])
        
    
    if train_wsc[0]:
        
        exit('Training WSC for glitchH is not implemented yet.')
        
        
        dcd = models['glitch_H'](torch.FloatTensor(training_set_ae[5][:, 101:]))[1].detach().numpy()
        err_score_H = np.mean((training_set_ae[5][:, 101:]-dcd)**2, axis=1)
        passidx = err_score_H > cutAE[0]
        dataset1 = training_set_ae[5][passidx].copy()
        wscs['glitch_H'] = trainWSC_2class(training_set_ae[0], dataset1[:, 101:], [101, 16, 1], outputDir+"/WSC_training_figures/wsc_glitchH_"+stric+".png")

        dcd = models['glitch_L'](torch.FloatTensor(training_set_ae[5][:, :101]))[1].detach().numpy()
        err_score_L = np.mean((training_set_ae[5][:, :101]-dcd)**2, axis=1)
        passidx = err_score_L > cutAE[1]
        dataset1 = training_set_ae[5][passidx].copy()
        wscs['glitch_L'] = trainWSC_2class(training_set_ae[1], dataset1[:, :101], [101, 16, 1], outputDir+"/WSC_training_figures/wsc_glitchL_"+stric+".png")

        for iStep in range(2, 6):
            passidxH = nn.Sigmoid()(wscs['glitch_H'](torch.FloatTensor(training_set_ae[iStep][:, 101:]))).detach().numpy().flatten()>=0.5
            passidxL = nn.Sigmoid()(wscs['glitch_L'](torch.FloatTensor(training_set_ae[iStep][:, :101]))).detach().numpy().flatten()>=0.5
            passidx = np.logical_and(passidxH, passidxL)
            training_set_ae[iStep] = training_set_ae[iStep][passidx]
    else:    
        dcd_train = aes['glitch_H'](torch.FloatTensor(training_set_ae[0][:, :101]))[1].detach().numpy()
        err_score_train = np.mean((training_set_ae[0][:, :101]-dcd_train)**2, axis=1)
        cutAE[0] = np.sort(err_score_train)[-int(cutAE_ratio[0] * len(err_score_train))]
        
        dcd_train = aes['glitch_L'](torch.FloatTensor(training_set_ae[1][:, 101:]))[1].detach().numpy()
        err_score_train = np.mean((training_set_ae[1][:, 101:]-dcd_train)**2, axis=1)
        cutAE[1] = np.sort(err_score_train)[-int(cutAE_ratio[1] * len(err_score_train))]
            
        for iStep in range(2, classnum+1):
            dcd = aes['glitch_H'](torch.FloatTensor(training_set_ae[iStep][:, :101]))[1].detach().numpy()
            err_score_H = np.mean((training_set_ae[iStep][:, :101]-dcd)**2, axis=1)
            passH = err_score_H > cutAE[0]


            dcd = aes['glitch_L'](torch.FloatTensor(training_set_ae[iStep][:, 101:]))[1].detach().numpy()
            err_score_L = np.mean((training_set_ae[iStep][:, 101:]-dcd)**2, axis=1)
            passL = err_score_L > cutAE[1]

            passidx = np.logical_and(passH, passL)
            training_set_ae[iStep] = training_set_ae[iStep][passidx]

    # logger(time.time()-t0)
    # t0 = time.time()

    for iCS in np.arange(2, classnum+1):
        if use_cache[iCS]:
            aes[classname[iCS]] = models[classname[iCS]]
        else:
            aes[classname[iCS]] = trainAE_struct(training_set_ae[iCS], ae_struct[iCS], config_dict[classkey[iCS]]['Training_scheme'])
        
        if train_wsc[iCS]:
            exit('Training WSC for {} is not implemented yet.'.format(classname[iCS]))
            # train AE with passed data
            
            # logger.info("Start training the {}-th node in the AE series. ".format(iCS))
            # if iCS==2: # if noise, find the cut value according to the distribution
            #     dcd = aes[indCS](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
            #     err_score = np.var(training_set_ae[iCS] - dcd, axis=1)
            #     err_score.sort()
            #     cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]
            
            # for all dataset, find the cut value according to the distribution
            dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
            err_score = np.mean((training_set_ae[iCS] - dcd)**2, axis=1)
            err_score.sort()
            cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]

            # dcd = aes[indCS](torch.FloatTensor(training_set_ae[5]))[1].detach().numpy()
            # err_score = np.var(training_set_ae[5] - dcd, axis=1)
            # passidx = err_score > cutAE[iCS]

            # dataset1 = training_set_ae[5][passidx].copy()
            # wscs[indCS] = trainWSC_2class(training_set_ae[iCS], dataset1, [202, 32, 1], outputDir+"/WSC_training_figures/wsc_"+ind2dt[iCS]+"_"+stric+".png")

            # logger.info("The node in the AE series training completed. ")
            
            # filter both the AE training set data and the test set
            # the AE training set
            for iStep in np.arange(iCS+1, classnum+1):
                # passidx = nn.Sigmoid()(wscs[indCS](torch.FloatTensor(training_set_ae[iStep]))).detach().numpy().flatten()>=0.5
                dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iStep]))[1].detach().numpy()
                err_score = np.mean((training_set_ae[iStep] - dcd)**2, axis=1)
                passidx = err_score > cutAE[iCS]
                training_set_ae[iStep] = training_set_ae[iStep][passidx]
        else:
            
            # logger.info("Skip training the {}-th node in the AE series. ".format(iCS))
            
            # if iCS==2:
            #     dcd = models[indCS](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
            #     err_score = np.var(training_set_ae[iCS] - dcd, axis=1)
            #     err_score.sort()
            #     cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]
            
            dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
            err_score = np.mean((training_set_ae[iCS] - dcd)**2, axis=1)
            err_score.sort()
            cutAE[iCS] = err_score[-int(cutAE_ratio[iCS]*len(err_score))]

            for iStep in np.arange(iCS+1, classnum+1):
                dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iStep]))[1].detach().numpy()
                err_score = np.mean((training_set_ae[iStep]-dcd)**2, axis=1)
                passidx = err_score > cutAE[iCS]
                training_set_ae[iStep] = training_set_ae[iStep][passidx]  
           
    aes['cut_vals'] = cutAE.copy()      
         
    torch.save(aes, model_chain_save_path) 
    
    return aes
    
def cut_events_from_waveforms_single_fullscan(cutting_config, waveforms):
    
    EVENT_PER_SEGMENT = cutting_config['General_config']['Event_per_segment']
    segment_freq = cutting_config['General_config']['Segment_frequency']
    segment_length = cutting_config['General_config']['Segment_length']
    file_path = cutting_config['Dataset_path']
    cutting_type = cutting_config['Cutting_type']
    
    # waveforms, _, _ = read_from_directory(file_path)
    # I have no ideas of how to use the SNR right now
    
    cutted_events = np.empty((segment_length-199,2,200))
    # midp = (waveforms[0].shape)[-1]//2
    
    
    for j in range(segment_length-199):
        starting_time = j
        ending_time = starting_time + 200
        
        cache = waveforms[:,starting_time:ending_time]
    
        cutted_events[j] = cache.copy()
    
    cutted_events /= np.linalg.norm(cutted_events, axis = -1).reshape(-1,2,1)
    cutted_events_fft = abs(np.fft.rfft(cutted_events))
    cutted_events_fft /= np.linalg.norm(cutted_events_fft, axis = -1).reshape(-1,2,1)
    
    return cutted_events_fft.reshape(-1,202)

def Series_passing(aes, config_dict, config_dict_for_passing):
    
    cutAE = aes['cut_vals']
    
    dataset = np.empty((0,202))
    
    # Remember the training_set_ae is a dictionary with keys 0, 1, 2, ..., n, the 0 are pure H glitches and 1 are pure L glitches. all in 202 shape. 
    
    classnum = config_dict['Class_to_use']
    classkey = ['GlitchH_class', 'GlitchL_class'] + ['Class'+str(i+1) for i in range(classnum - 1)]
    classname = [config_dict[key]['Model_params']['Class_name'] for key in classkey]
    
    # Right now the cache part can only work for the AE
    train_wsc = [config_dict[key]['Training_scheme']['Train_wsc'] for key in classkey]
    
    for key in classkey:
        config_dict[key]['Training_scheme']['Output_dir'] = config_dict['Output_dir']
        config_dict[key]['Training_scheme']['Output_file_infix'] = config_dict['Output_file_infix'] + config_dict[key]['Model_params']['Class_name'] + '_'
        config_dict[key]['Training_scheme']['Output_file_suffix'] = config_dict['Output_file_suffix']
    
    
    # Modification has to be made on the param of the AE training
    
    # Loading and making the testing waveforms
    
    file_path = config_dict_for_passing['Dataset_path']
    cache_path = config_dict_for_passing['Cache_path']
    waveforms, _, _ = read_from_directory(file_path)
    
    for waveform in waveforms:
        dataset_filtered = cut_events_from_waveforms_single_fullscan(config_dict_for_passing, waveform)
        # Glitch part comes first    
        if train_wsc[0]:
            
            exit('Training WSC for glitchH is not implemented yet.')
            
            
            dcd = models['glitch_H'](torch.FloatTensor(training_set_ae[5][:, 101:]))[1].detach().numpy()
            err_score_H = np.mean((training_set_ae[5][:, 101:]-dcd)**2, axis=1)
            passidx = err_score_H > cutAE[0]
            dataset1 = training_set_ae[5][passidx].copy()
            wscs['glitch_H'] = trainWSC_2class(training_set_ae[0], dataset1[:, 101:], [101, 16, 1], outputDir+"/WSC_training_figures/wsc_glitchH_"+stric+".png")

            dcd = models['glitch_L'](torch.FloatTensor(training_set_ae[5][:, :101]))[1].detach().numpy()
            err_score_L = np.mean((training_set_ae[5][:, :101]-dcd)**2, axis=1)
            passidx = err_score_L > cutAE[1]
            dataset1 = training_set_ae[5][passidx].copy()
            wscs['glitch_L'] = trainWSC_2class(training_set_ae[1], dataset1[:, :101], [101, 16, 1], outputDir+"/WSC_training_figures/wsc_glitchL_"+stric+".png")

            for iStep in range(2, 6):
                passidxH = nn.Sigmoid()(wscs['glitch_H'](torch.FloatTensor(training_set_ae[iStep][:, 101:]))).detach().numpy().flatten()>=0.5
                passidxL = nn.Sigmoid()(wscs['glitch_L'](torch.FloatTensor(training_set_ae[iStep][:, :101]))).detach().numpy().flatten()>=0.5
                passidx = np.logical_and(passidxH, passidxL)
                training_set_ae[iStep] = training_set_ae[iStep][passidx]
        else:    
            
            dcd = aes['glitch_H'](torch.FloatTensor(dataset_filtered[:, :101]))[1].detach().numpy()
            err_score_H = np.mean((dataset_filtered[:, :101]-dcd)**2, axis=1)
            passH = err_score_H > cutAE[0]


            dcd = aes['glitch_L'](torch.FloatTensor(dataset_filtered[:, 101:]))[1].detach().numpy()
            err_score_L = np.mean((dataset_filtered[:, 101:]-dcd)**2, axis=1)
            passL = err_score_L > cutAE[1]

            passidx = np.logical_and(passH, passL)
            dataset_filtered = dataset_filtered[passidx]

        # logger(time.time()-t0)
        # t0 = time.time()

        for iCS in np.arange(2, classnum+1):
            
            if train_wsc[iCS]:
                exit('Training WSC for {} is not implemented yet.'.format(classname[iCS]))
                # train AE with passed data
                
                # logger.info("Start training the {}-th node in the AE series. ".format(iCS))
                # if iCS==2: # if noise, find the cut value according to the distribution
                #     dcd = aes[indCS](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
                #     err_score = np.var(training_set_ae[iCS] - dcd, axis=1)
                #     err_score.sort()
                #     cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]
                
                # for all dataset, find the cut value according to the distribution
                dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
                err_score = np.mean((training_set_ae[iCS] - dcd)**2, axis=1)
                err_score.sort()
                cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]

                # dcd = aes[indCS](torch.FloatTensor(training_set_ae[5]))[1].detach().numpy()
                # err_score = np.var(training_set_ae[5] - dcd, axis=1)
                # passidx = err_score > cutAE[iCS]

                # dataset1 = training_set_ae[5][passidx].copy()
                # wscs[indCS] = trainWSC_2class(training_set_ae[iCS], dataset1, [202, 32, 1], outputDir+"/WSC_training_figures/wsc_"+ind2dt[iCS]+"_"+stric+".png")

                # logger.info("The node in the AE series training completed. ")
                
                # filter both the AE training set data and the test set
                # the AE training set
                for iStep in np.arange(iCS+1, classnum+1):
                    # passidx = nn.Sigmoid()(wscs[indCS](torch.FloatTensor(training_set_ae[iStep]))).detach().numpy().flatten()>=0.5
                    dcd = aes[classname[iCS]](torch.FloatTensor(training_set_ae[iStep]))[1].detach().numpy()
                    err_score = np.mean((training_set_ae[iStep] - dcd)**2, axis=1)
                    passidx = err_score > cutAE[iCS]
                    training_set_ae[iStep] = training_set_ae[iStep][passidx]
            else:
                
                # logger.info("Skip training the {}-th node in the AE series. ".format(iCS))
                
                # if iCS==2:
                #     dcd = models[indCS](torch.FloatTensor(training_set_ae[iCS]))[1].detach().numpy()
                #     err_score = np.var(training_set_ae[iCS] - dcd, axis=1)
                #     err_score.sort()
                #     cutAE[iCS] = err_score[-int(cutAE[iCS]*len(err_score))]
                
                dcd = aes[classname[iCS]](torch.FloatTensor(dataset_filtered))[1].detach().numpy()
                err_score = np.mean((dataset_filtered-dcd)**2, axis=1)
                passidx = err_score > cutAE[iCS]
                dataset_filtered = dataset_filtered[passidx]   

        dataset = np.append(dataset, dataset_filtered, axis = 0)
        
        np.save(cache_path, dataset)
    
    return dataset


if __name__ == "__main__":
    device = 'cpu'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    dataset_trial = torch.load('/home/app/test_data/trial.json', weights_only=False)
    dataset_trial[6] = dataset_trial[4].copy()
    
    dataset_trial[5] = dataset_trial[4]
    dataset_trial[4] = dataset_trial[3]
    dataset_trial[3] = dataset_trial[2]
    dataset_trial[2] = dataset_trial[1]
    dataset_trial[1] = dataset_trial[0][10000:]
    dataset_trial[0] = dataset_trial[0][:10000]
    
    print(dataset_trial.keys())
    
    for key in dataset_trial.keys():
        np.random.shuffle(dataset_trial[key])
    
    config['Filtering_Chain']['Output_dir'] = config['Full_pipeline']['Training_scheme']['Output_dir']
    
    
    Series_training(dataset_trial, config_dict=config['Filtering_Chain'])
   