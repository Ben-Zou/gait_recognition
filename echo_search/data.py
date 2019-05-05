import torch
import torch.nn as nn
import pdb
import numpy as np
import torchvision.transforms as transforms
import torchvision as T
import torchvision.transforms.functional as F
import torch.utils.data as data

from utils import target_win, target_loss

from PIL import Image


class gait_frame(data.Dataset):
    # gait_28x28.npy
    # 8x50x50x28x28
    def __init__(self,datax,
                 T = 50,
                 train=True):
        self.data = datax # [8,n,50,28,28]
        c,n,_,w,h = datax.shape
        nt = int(50/T)
        # self.data = self.data.reshape(c,n,10,5,w*h).transpose(0,1,3,2,4).reshape(c*n*5,10,w*h)
        self.data = self.data.reshape(c,n,T,nt,w*h).transpose(0,1,3,2,4).reshape(c*n*nt,T,w*h)
        self.label = np.arange(c).repeat(n*nt)
        self.c = c

    def __getitem__(self, index):
        x = self.data[index]
        x = x/255.
        y = self.label[index]
        return torch.FloatTensor(x),torch.FloatTensor([y]).long()

    def __len__(self):
        return len(self.data)


class gait_frame_val(data.Dataset):
    # gait_28x28.npy
    # 8x50x50x28x28
    def __init__(self,datax,
                 Tstim = 100,
                 scale = 60.,
                 label = 0,
                 T = 50):
        self.data = datax # [8,n,50,28,28]
        c,n,_,w,h = datax.shape
        n_Tstim_repeat = int(Tstim/T)
        # self.data = self.data.reshape(c,n,10,5,w*h).transpose(0,1,3,2,4).reshape(c*n*5,10,w*h)
        self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h).repeat(n_Tstim_repeat,axis=1)
        self.c = 10  # for 10 way
        self.label = np.array([label]).repeat(n)
        self.Tstim =  Tstim
        self.label_i = (np.tanh(np.linspace(-2.,2.,Tstim)) + 1.0)*scale/2
    def __getitem__(self, index):
        x = self.data[index]
        x = x/255.
        y_index = int(self.label[index])
        y = np.zeros((self.Tstim,self.c)) + 0.01
        y[:,y_index] = self.label_i
        return torch.FloatTensor(x),torch.FloatTensor(y)

    def __len__(self):
        return len(self.data)



class gait_frame_dm_xtarget(data.Dataset):
    # gait_28x28.npy
    # 8x50x50x28x28
    def __init__(self,datax,
                 Tstim = 100,
                 T = 50,
                 scale = 60.,
                 input_strength=1.,
                 normalize=False):
        self.data = datax # [8,n,50,28,28]
        c,n,_,w,h = datax.shape
        n_Tstim_repeat = int(Tstim/T)
        # self.data = self.data.reshape(c,n,10,5,w*h).transpose(0,1,3,2,4).reshape(c*n*5,10,w*h)
        self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h).repeat(n_Tstim_repeat,axis=1)

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.normalize = normalize

        self.label = np.arange(c).repeat(n)
        self.c = c
        self.input_strength = input_strength
        self.Tstim =  Tstim
        xx = np.tanh(np.linspace(-2.,2.,Tstim))
        self.label_win_i = target_win(xx,self.c)
        self.label_loss_i = target_loss(xx,self.c)

    def __getitem__(self, index):
        x = self.data[index]
        if self.normalize:
            x = (x-self.mean)/self.std*self.input_strength
        else:
            x = x/255.*self.input_strength
        y_index = int(self.label[index])
        # y = np.zeros((self.Tstim,self.c)) + 0.01
        y = self.label_loss_i.repeat(self.c).reshape(self.Tstim,self.c)
        y[:,y_index] = self.label_win_i
        return torch.FloatTensor(x),torch.FloatTensor(y)

    def __len__(self):
        return len(self.data)

class gait_frame_dm(data.Dataset):
    # gait_28x28.npy
    # 8x50x50x28x28
    def __init__(self,datax,
                 Tstim = 100,
                 scale = 60.,
                 T = 50,
                 input_strength=1.,
                 normalize=True):
        self.data = datax # [8,n,50,28,28]
        c,n,_,w,h = datax.shape
        n_Tstim_repeat = int(Tstim/T)
        # self.data = self.data.reshape(c,n,10,5,w*h).transpose(0,1,3,2,4).reshape(c*n*5,10,w*h)
        self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h).repeat(n_Tstim_repeat,axis=1)

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

        self.normalize = normalize

        self.label = np.arange(c).repeat(n)
        self.c = c
        self.input_strength = input_strength
        self.Tstim =  Tstim
        self.label_i = (np.tanh(np.linspace(-2.,2.,Tstim)) + 1.0)*scale/2 + 0.01
        # self.label_i = np.ones((Tstim,))*scale
    def __getitem__(self, index):
        x = self.data[index]
        if self.normalize:
            x = (x-self.mean)/self.std*self.input_strength
        else:
            x = x/255.*self.input_strength
        y_index = int(self.label[index])
        y = np.zeros((self.Tstim,self.c)) + 0.01
        y[:,y_index] = self.label_i
        return torch.FloatTensor(x),torch.FloatTensor(y)

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    data_array = np.load("../data/gait_28x28.npy")
    d = gait_frame_dm_xtarget(data_array[0:5])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x,y = d[1]


    # plt.figure(figsize=(4,50))
    # for n in range(100):
    #     plt.subplot(20,5,n+1)
    #     plt.imshow(x[n].reshape(28,28))
    #     plt.gray()
    #     plt.axis("off")
    # plt.savefig("xxx.png")
