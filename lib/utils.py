import time

import numpy as np
from lib.LISTA_class import LISTA
from lib.FC_Conv import FC_Conv
from lib.vgg_loss import VGG_Loss
from lib.unet_flexible import UNet_wrapper
from lib.DOTDataset_class import DOTDataset

import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

import torch
from torch.utils.data import DataLoader
from kornia.losses import ssim_loss

import os

def dssim_loss(T1, T2):
    imSize = T1.shape[0]
    imLen = int(imSize**0.5)

    if len(T1.shape) < 2:
        T1 = torch.reshape(T1, (imSize, 1))
        T2 = torch.reshape(T2, (imSize, 1))

    nTrain = T1.shape[1]

    T1 = torch.reshape(torch.permute(T1, (1,0)), (nTrain, 1, imLen, imLen))
    T2 = torch.reshape(torch.permute(T2, (1,0)), (nTrain, 1, imLen, imLen))

    return ssim_loss(T1, T2, 1, reduction="mean")

def showIms(ims):
    nIms = ims.shape[2]
    plt.figure(figsize=(3*nIms,3))
    for i in range(nIms):
        plt.subplot(1,nIms,i+1)
        plt.imshow(ims[:,:,i])
        plt.gca().set_axis_off()
    plt.show()


def getDatasetMat(matpath, nTest, measNormalization=1.0, datVarName="allDiffDat", truthVarName="truthIms",
                 Jname=None):
    mat_dict = loadmat(matpath)
    all_meas = mat_dict[datVarName] * (measNormalization / np.amax(mat_dict[datVarName]))
    all_truth = mat_dict[truthVarName]

    splitInds = np.random.permutation(all_meas.shape[2])
    testInds = splitInds[:nTest]
    trainInds = splitInds[nTest:]
    train_meas, test_meas = all_meas[:,:,trainInds], all_meas[:,:,testInds]
    train_truth, test_truth = all_truth[:,:,trainInds], all_truth[:,:,testInds]

    if Jname == None:
        return DOTDataset(trainMeas=train_meas, trainTruth=train_truth, testMeas=test_meas, testTruth=test_truth), trainInds, testInds
    else:
        return DOTDataset(trainMeas=train_meas, trainTruth=train_truth, testMeas=test_meas, testTruth=test_truth), trainInds, testInds, mat_dict[Jname]

def train_model(dataset_in, train_d, dev, A=None, visInds=[],
               model=None, optim=None):
    
    nLayers = train_d["nLayers"]
    scale_mag = train_d["scale_mag"]
    lam1 = train_d["lam1"]
    untied = train_d["untied"]
    lossFunc = train_d["lossFunc"]
    LR = train_d["LR"]
    nEpochs = train_d["nEpochs"]
    batch_sz = train_d["batch_sz"]
    showEvery = train_d["showEvery"]
    if "actfunc" in train_d:
        actfunc = train_d["actfunc"]
    else:
        actfunc = "shrink"
    if "vgg_weight" in train_d:
        vgg_weight = train_d["vgg_weight"]
    else:
        vgg_weight = 0
    if "unet_nfilts" in train_d:
        unet_nfilts = train_d["unet_nfilts"]
    else:
        unet_nfilts = 0

    szA, _, _, imY, imX = dataset_in.getDims()
    
    # Initialize training variables: model, loss(es), optimizer
    if lossFunc == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif lossFunc == "MAE":
        loss_fn = torch.nn.L1Loss()
    elif lossFunc == "SSIM":
        loss_fn = dssim_loss
    else:
        raise AssertionError("Invalid loss function")
        
    if model == None:
        model = LISTA(numLayers=nLayers, szA=szA, scale_mag=scale_mag, actfunc=actfunc, untrained_lamL1=lam1, untied=untied, A=A)
        model.to(dev)
        
    if vgg_weight > 0:
        vgg_loss_fn = VGG_Loss(imSz=imY, loss_fn=loss_fn)
        vgg_loss_fn.send2dev(dev)
    if unet_nfilts > 0:
        unet = UNet_wrapper(imSz=imY, nfilts=unet_nfilts, input_channels=1, bn_input=True)
        unet.send2dev(dev)
        optim_internal = torch.optim.Adam(list(model.parameters())+list(unet.get_params()), lr=LR)
    else:
        optim_internal = torch.optim.Adam(model.parameters(), lr=LR)
        
    if optim == None:
        optimizer = optim_internal
    else:
        optimizer = optim

    epoch_arr = np.array([])
    train_losses = np.array([])
    test_losses = np.array([])
    grad_arr = np.array([])
    runtime_arr = np.array([])
    misc = {}

    train_dataloader = DataLoader(dataset_in, batch_size=batch_sz, shuffle=True)
    trainMeas, trainTruth = dataset_in.getFullTrainSet()
    testMeas, testTruth = dataset_in.getFullTestSet()
        
    if len(visInds) > 0:
        truthIms = torch.cat((trainTruth[:,visInds], testTruth[:,visInds]), dim=1) # get true images for training and test sets
        truthIms = np.reshape(truthIms.cpu().detach().numpy(), (imY, imX, -1))
        showIms(truthIms)
    
    itr_start = time.perf_counter()
    for n in range(nEpochs):
        for i_batch, samples_batch in enumerate(train_dataloader):
            
            Y_torch, X_torch = samples_batch
            Y_torch = Y_torch.T.to(dev)
            X_torch = X_torch.T.to(dev)
        
            # Zero the gradient from the previous iteration
            optimizer.zero_grad()

            # Apply the forward model to training data
            X_pred = model(Y_torch)
            if unet_nfilts > 0:
                X_pred = unet(X_pred)

            # Calculate loss
            if vgg_weight > 0:
                vgg_loss_curr = vgg_weight * vgg_loss_fn.get_loss(X_torch, X_pred)
            else:
                vgg_loss_curr = 0
            loss = loss_fn(X_torch, X_pred) + vgg_loss_curr

            # Test current model on test data
            if (n % showEvery == 0 or n == nEpochs - 1) and (i_batch == len(train_dataloader) - 1):
                with torch.no_grad():
                    Y_test_torch = testMeas.to(dev)
                    X_test_torch = testTruth.to(dev)
                    X_pred_test = model(Y_test_torch)
                    if unet_nfilts > 0:
                        X_pred_test = unet(X_pred_test)
                    loss_test = loss_fn(X_test_torch, X_pred_test)

            # Backpropagate
            loss.backward()

            # Gradient step
            optimizer.step()
        
        # Print training info
        if (n % showEvery == 0 or n == nEpochs - 1):
            with torch.no_grad():
                itr_end = time.perf_counter()
                itr_time = itr_end - itr_start
                print("Epoch %d; Training Loss: %g; Test Loss: %g; ran %.2f sec" % (n, loss.item(), loss_test.item(), itr_time))
                epoch_arr = np.append(epoch_arr, n)
                train_losses = np.append(train_losses, loss.item())
                test_losses = np.append(test_losses, loss_test.item())
                runtime_arr = np.append(runtime_arr, itr_time)

                if len(visInds) > 0:
                    predIms_train = np.reshape(X_pred.cpu().detach().numpy()[:,visInds], (imY, imX, -1))
                    predIms_test = np.reshape(X_pred_test.cpu().detach().numpy()[:,visInds], (imY, imX, -1))
                    predIms = np.concatenate((predIms_train, predIms_test), axis=2)
                    showIms(predIms)

                itr_start = time.perf_counter()
                
                if "intermed_path" in train_d:
                    save_d = {
                        'n': n,
                        'epoch_arr': epoch_arr,
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'recon_ims_test': X_pred_test.cpu().detach().numpy(),
                        'truth_ims_test': testTruth.cpu().detach().numpy(),
                        'WT': model.W.T.cpu().detach().numpy(),
                    }
                    full_savestr = os.path.join(train_d["intermed_path"], 'itr=%d.mat' % (n))
                    savemat(full_savestr, save_d)
    
    if unet_nfilts > 0:
        misc["unet"] = unet
        
    misc["runtime_arr"] = runtime_arr
    
    return model, epoch_arr, train_losses, test_losses, misc