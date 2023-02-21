import torch
import torch.nn as nn
import numpy as np

     
class LISTA(nn.Module):
    def __init__(self, numLayers, szA, scale_mag, actfunc="shrink", untrained_lamL1=None, untied=False, A=None):
        super().__init__()
        
        if A is None:
            self.szA_0, self.szA_1 = szA
            self.predefined_A = False
        else:
            self.szA_0, self.szA_1 = A.shape
            AT = A.T
            ATA = AT@A
            stp = scale_mag
            self.preset_W = (1/stp) * AT
            self.preset_S = np.eye(self.szA_1) - ((1/stp) * (AT@A))
            self.predefined_A = True
            
        self.NL = numLayers
        self.untied = untied
        self.actfunc = actfunc
        self.scale_mag = scale_mag
        self.untrained_lamL1 = untrained_lamL1

        if (untied):
            W, _, lam = self.initVars()
            self.W = W # W is always tied
            self.S = [] # If 0 layers, do not need S
            
            self.register_parameter("lam0", lam)
            self.lam = [lam]
            
            for L in range(self.NL):
                _, S, lam = self.initVars()
                self.register_parameter("S%d" % L, S)
                self.S.append(S)
                
                self.register_parameter("lam%d" % (L + 1), lam)
                self.lam.append(lam)

        else:
            W, S, lam = self.initVars()
            self.W = W
            self.S = S
            self.lam = lam

    def initVars(self):
        if self.predefined_A:
            W_np = self.preset_W
            S_np = self.preset_S
        else:
            W_np = self.scale_mag * np.random.rand(self.szA_1, self.szA_0)
            S_np = self.scale_mag * np.random.rand(self.szA_1, self.szA_1)
        lam_np = self.scale_mag * np.random.rand(1,1)
        
        W = nn.Parameter(torch.tensor(W_np, dtype=torch.double))
        S = nn.Parameter(torch.tensor(S_np, dtype=torch.double))

        if (self.untrained_lamL1 == None):
            lam = nn.Parameter(torch.tensor(lam_np, dtype=torch.double, requires_grad=True))
        else:
            lam = torch.tensor(self.untrained_lamL1, dtype=torch.double, requires_grad=False)

        return W, S, lam 
        
    def applyActFunc(self, x, L=None):
        if self.actfunc == "shrink":
            if self.untied:
                return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam[L], min=0)
            else:
                return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam, min=0)
        elif self.actfunc == "swish":
            return torch.nn.SiLU()(x)
        elif self.actfunc == "tanh":
            return torch.nn.Tanh()(x)
        else:
            raise ValueError('Incorrect activation function set')
        
    def forward(self, Y):
        if (self.untied):
            Wy = self.W @ Y 
            Z = Wy 
            for L in range(self.NL):
                Z = Wy + self.S[L] @ self.applyActFunc(Z, L)
            return self.applyActFunc(Z, self.NL)

        else:
            Wy = self.W@Y
            Z = Wy
            for L in range(self.NL):
                Z = Wy + self.S@self.applyActFunc(Z)
            return self.applyActFunc(Z)