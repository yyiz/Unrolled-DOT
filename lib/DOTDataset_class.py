import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np

class DOTDataset(Dataset):
    def __init__(self, trainMeas, trainTruth, testMeas, testTruth):
        self.trainMeas = trainMeas
        self.trainTruth = trainTruth
        self.testMeas = testMeas
        self.testTruth = testTruth
        
        self.NBINS, self.nSrcDet, self.nTrain = self.trainMeas.shape
        self.nTest = self.testMeas.shape[2]
        self.imY, self.imX, _ = self.trainTruth.shape
        self.szA = (self.NBINS * self.nSrcDet, self.imY * self.imX)
        
        assert(self.testMeas.shape[0] == self.NBINS)
        assert(self.testMeas.shape[1] == self.nSrcDet)
        assert(self.testTruth.shape[0] == self.imY)
        assert(self.testTruth.shape[1] == self.imX)
        assert(self.trainTruth.shape[2] == self.nTrain)
        assert(self.testTruth.shape[2] == self.nTest)

        self.trainMeas = torch.tensor(np.reshape(self.trainMeas, (self.NBINS*self.nSrcDet, self.nTrain)), dtype=torch.double)
        self.trainTruth = torch.tensor(np.reshape(self.trainTruth, (self.imY*self.imX, self.nTrain)), dtype=torch.double)
        self.testMeas = torch.tensor(np.reshape(self.testMeas, (self.NBINS*self.nSrcDet, self.nTest)), dtype=torch.double)
        self.testTruth = torch.tensor(np.reshape(self.testTruth, (self.imY*self.imX, self.nTest)), dtype=torch.double)
        
    def __len__(self):
        return self.nTrain
    
    def __getitem__(self, idx):
        return self.trainMeas[:,idx], self.trainTruth[:,idx]
    
    def getFullTrainSet(self):
        return self.trainMeas, self.trainTruth
    
    def getFullTestSet(self):
        return self.testMeas, self.testTruth
    
    def getDims(self):
        return self.szA, self.NBINS, self.nSrcDet, self.imY, self.imX