import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
import numpy as np
import pdb
import sys
sys.path.append("../")
from rnn import SimpleEcho
from rnn import DMCell


class dm_echo1(nn.Module):

    def __init__(self,n_inp,
                      n_echo,
                      n_dm,
                      echo_dict,
                      dm_dict):
        super().__init__()
        self.n_inp = n_inp
        self.n_echo = n_echo
        self.n_dm = n_dm

        self.simple_echo = SimpleEcho(inp_num=n_inp,
                                      hid_num=n_echo,
                                      **echo_dict)
        self.dm = DMCell(inp_num=n_echo,
                         hid_num=n_dm,
                         **dm_dict)

    def forward(self,x,hid,y=None):
        if y is None:
            u_echo,h_echo,s_dm = hid
            h_echo_new, (u_echo_new,_) = self.simple_echo(x,(u_echo,h_echo))
            r_dm_new,(s_dm_new,) = self.dm(h_echo_new,(s_dm,))
            return r_dm_new, (u_echo_new,h_echo_new,s_dm_new)
        else:
            u_echo,h_echo,s_dm,P_dm = hid
            h_echo_new, (u_echo_new,_) = self.simple_echo(x,(u_echo,h_echo))
            err,r_dm_new ,(s_dm_new,P_dm_new) = self.dm(h_echo_new,(s_dm,P_dm),y)
            return err,r_dm_new ,(u_echo_new,h_echo_new,s_dm_new,P_dm_new)


class dm_echo3(nn.Module):

    def __init__(self,n_inp,
                      n_echo1,
                      n_echo2,
                      n_echo3,
                      n_dm,
                      echo_dict1,
                      echo_dict2,
                      echo_dict3,
                      dm_dict):
        super().__init__()
        self.n_inp = n_inp
        self.n_echo1 = n_echo1
        self.n_echo2 = n_echo2
        self.n_echo3 = n_echo3
        self.n_dm = n_dm

        self.simple_echo1 = SimpleEcho(inp_num=n_inp,
                                      hid_num=n_echo1,
                                      tau = echo_dict1["tau"],
                                      dt = echo_dict1["dt"],
                                      scale = echo_dict1["scale"],
                                      spars_p = echo_dict1["spars_p"],)

        self.simple_echo2 = SimpleEcho(inp_num=n_echo1,
                                      hid_num=n_echo2,
                                      tau = echo_dict2["tau"],
                                      dt = echo_dict2["dt"],
                                      scale = echo_dict2["scale"],
                                      spars_p = echo_dict2["spars_p"],)

        self.simple_echo3 = SimpleEcho(inp_num=n_echo2,
                                      hid_num=n_echo3,
                                      tau = echo_dict3["tau"],
                                      dt = echo_dict3["dt"],
                                      scale = echo_dict3["scale"],
                                      spars_p = echo_dict3["spars_p"],)

        self.dm = DMCell(inp_num=n_echo1+n_echo2+n_echo3,
                         hid_num=n_dm,
                         **dm_dict
                         )
                         # dt = dm_dict["dt"],
                         # taus = dm_dict["taus"])

    def forward(self,x,hid,y=None):
        if y is None:
            u_echo1,h_echo1,u_echo2,h_echo2,u_echo3,h_echo3,s_dm = hid
            h_echo_new1, (u_echo_new1,_) = self.simple_echo1(x,(u_echo1,h_echo1))
            h_echo_new2, (u_echo_new2,_) = self.simple_echo2(h_echo_new1,(u_echo2,h_echo2))
            h_echo_new3, (u_echo_new3,_) = self.simple_echo3(h_echo_new2,(u_echo3,h_echo3))
            h_echo_new = torch.cat([h_echo_new1,h_echo_new2,h_echo_new3],dim=1)
            r_dm_new,(s_dm_new,) = self.dm(h_echo_new,(s_dm,))
            return r_dm_new, \
                  (u_echo_new1,h_echo_new1,
                   u_echo_new2,h_echo_new2,
                   u_echo_new3,h_echo_new3,
                   s_dm_new)
        else:
            u_echo1,h_echo1,u_echo2,h_echo2,u_echo3,h_echo3,s_dm,P_dm = hid
            h_echo_new1, (u_echo_new1,_) = self.simple_echo1(x,(u_echo1,h_echo1))
            h_echo_new2, (u_echo_new2,_) = self.simple_echo2(h_echo_new1,(u_echo2,h_echo2))
            h_echo_new3, (u_echo_new3,_) = self.simple_echo3(h_echo_new2,(u_echo3,h_echo3))
            h_echo_new = torch.cat([h_echo_new1,h_echo_new2,h_echo_new3],dim=1)
            err, r_dm_new,(s_dm_new,P_dm_new) = self.dm(h_echo_new,(s_dm,P_dm),y)
            return err, r_dm_new,\
                  (u_echo_new1,h_echo_new1,
                   u_echo_new2,h_echo_new2,
                   u_echo_new3,h_echo_new3,
                   s_dm_new,P_dm_new)
