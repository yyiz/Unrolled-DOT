import torch
import numpy as np

from scipy.io import savemat

import sys

setpaths_dir = "../setpaths"
sys.path.append(setpaths_dir)
from setpaths import setpaths
libpath, datpath, resultpath, basepath = setpaths(setpaths_dir)

sys.path.append(basepath)


savename = "%s/jacobian_compare/W_mat.mat" % (resultpath)

modelpath = sys.argv[1]
saved_model = torch.load(modelpath)
model = saved_model["model"]

save_dict = {
    "WT": model.W.cpu().detach().numpy().T,
}

savemat(savename, save_dict)
print(savename)