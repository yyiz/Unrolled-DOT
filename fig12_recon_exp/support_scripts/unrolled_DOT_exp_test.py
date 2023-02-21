import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat

import torch

import sys
import time

setpaths_dir = "../setpaths"
sys.path.append(setpaths_dir)
from setpaths import setpaths
libpath, datpath, resultpath, basepath = setpaths(setpaths_dir)

sys.path.append(basepath)
from lib.DOTDataset_class import DOTDataset
from lib.utils import train_model

#------------------------------------------------------------------------------

rand_seed = 0
np.random.seed(rand_seed)
torch_seed = torch.manual_seed(rand_seed)

GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

#------------------------------------------------------------------------------
# Load in model

modelname = sys.argv[1]

modelpath = os.path.join(resultpath, "exp", "%s.pt" % modelname)
saved_model = torch.load(modelpath)

model = saved_model["model"]
full_dataset = saved_model["full_dataset"]
train_dict = saved_model["train_dict"]
unet_nfilts = train_dict["unet_nfilts"]
if unet_nfilts > 0:
    unet = saved_model["unet"].to('cpu')

datVarName = "diff_L"

#------------------------------------------------------------------------------
# Image reconstruction

savename = "unrolled_dot_exp_eml_test"
savename = "%s_%s.mat" % (savename, modelname)
savepath = os.path.join(resultpath, "exp", savename)


Y_test_torch, _ = full_dataset.getFullTestSet()
_, _, _, imY, imX = full_dataset.getDims()
nIms = Y_test_torch.shape[1]

model = model.to('cpu')
Y_test_torch = Y_test_torch.cpu()

starttime = time.perf_counter()
reconIms = model(Y_test_torch)
if unet_nfilts > 0:
    reconIms = unet(reconIms)
finishtime = time.perf_counter()
reconTime = finishtime - starttime

reconIms_np = np.reshape(reconIms.detach().numpy(), (imY, imX, nIms))

print(savename)

matdict = {
    "reconTime_unrolled": reconTime,
    "reconIms_unrolled": reconIms_np, 
    "modelname_unrolled": modelname,
}

savemat(savepath, matdict)