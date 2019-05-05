import torch
import math
import sys
sys.path.append("../")
import pdb
import numpy as np
import torch.nn as nn
from rnn import DMCell
from rnn import LogAct
from rnn import RecLogAct

import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision as T
import argparse
import os
import glob
import tqdm
# from data import gait_frame_dm
from data import gait_frame_dm_xtarget as gait_frame_dm

from api import train_model_bp
from api import test_model

from utils import plot_r

parser = argparse.ArgumentParser(description='Decision Making')
parser.add_argument('--save-dir', type=str, default='default', help='save directory')
parser.add_argument('--logdir', type=str, default='default', help='save directory')
parser.add_argument('--seed', type=int, default=10000, help='seed value')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--tau_dm', type=float, default=60, help='time constant of decision making')
parser.add_argument('--num_dm', type=int, default=10, help='number of dm cells')
parser.add_argument('--test', action='store_true', default=False, help='plot figs or not')
parser.add_argument('--valid', action='store_true', default=False, help='plot figs or not')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=0.5, help='gradient clipping value')
parser.add_argument('--target_scale', type=float, default=10., help='target scale')
parser.add_argument('--T', type=int, default=50, help='length of input sequence')
parser.add_argument('--plot', action='store_true', default=False, help='plot figs or not')
parser.add_argument('--target_mode', type=str, default="x_target", help='save directory')

args = parser.parse_args()
args.log_dir = './res/{}/'.format(args.logdir)
args.save_dir = "{},{}".format(args.tau_dm,args.seed)

os.makedirs(args.log_dir,exist_ok=True)
cuda = torch.cuda.is_available()
os.makedirs(args.log_dir+"test/",exist_ok=True)
os.makedirs(args.log_dir+"train/",exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if cuda:
	torch.cuda.manual_seed(args.seed)
tensor = torch.FloatTensor


# 2 class
if args.num_dm == 2:
	dm_dict = dict(dt  = 1.,
				   Je = 8.,
				   Jm = -2.,
				   gamma = 0.1,
				   learning_rule = "bp",
				   activation = LogAct(alpha=1.5,beta=4.,gamma=0.1,thr=6.),
				   rec_activation = RecLogAct(alpha=1.5,beta=4.,gamma=0.1,thr=6.),
				   taus = args.tau_dm)



# 5 class
if args.num_dm == 5:
	dm_dict = dict(dt  = 1.,
				   Je = 9.,
				   Jm = -5,
				   gamma = 0.1,
				   learning_rule = "bp",
				   activation = LogAct(alpha=1.5,beta=3.2,gamma=0.1,thr=3.),
				   rec_activation = RecLogAct(alpha=1.5,beta=3.2,gamma=0.1,thr=3.),
				   taus = args.tau_dm)

# # 10 class
if args.num_dm == 10:
	dm_dict = dict(dt  = 1.,
				   Je = 18.,
				   Jm = -11.,
				   gamma = 0.1,
				   learning_rule = "bp",
				   activation = LogAct(alpha=1.5,beta=1.,gamma=0.1,thr=1.),
				   rec_activation = RecLogAct(alpha=1.5,beta=1.,gamma=0.1,thr=1.),
				   taus = args.tau_dm)

if args.num_dm == 15:
	dm_dict = dict(dt  = 1.,
				   Je = 20.,
				   Jm = -20.,
				   I0 = 0.4,
				   gamma = 0.1,
				   learning_rule = "bp",
				   activation = LogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-5.),
				   rec_activation = RecLogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-5.),
				   taus = args.tau_dm)


if args.num_dm == 20:
	dm_dict = dict(dt  = 1.,
				   Je = 27.,
				   Jm = -27.,
				   I0 = 0.3,
				   gamma = 0.1,
				   learning_rule = "bp",
				   activation = LogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-9.),
				   rec_activation = RecLogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-9.),
				   taus = args.tau_dm)

gait_data = np.load("../data/gait_28x28_17.npy")
inp_size = 28*28

# gait_data = np.load("../data/gait_30x48_17.npy")
# inp_size = 30*48

# gait_data = np.load("../data/raw_30x48_17.npy")
# inp_size = 30*48

gait_data = gait_data[:args.num_dm]
tot_trails = 50
n_trails = 5

index = np.arange(0,tot_trails)
np.random.shuffle(index)
train_index = index[:n_trails]
val_index = index[5:20]
test_index = index[20:]

train_size = n_trails*args.num_dm
val_size = 15*args.num_dm
test_size = 30*args.num_dm

# index = np.arange(0,tot_trails)
# train_index = index[::int(tot_trails/n_trails)]
# val_index = [i for i in index if i not in train_index]
# test_index = val_index
# train_size = n_trails*args.num_dm
# val_size = 45*args.num_dm
# test_size = 45*args.num_dm

trainset = gait_frame_dm(gait_data[:,train_index],Tstim=args.T,scale=args.target_scale)
validset = gait_frame_dm(gait_data[:,val_index],Tstim=args.T,scale=args.target_scale)
testset = gait_frame_dm(gait_data[:,test_index],Tstim=args.T,scale=args.target_scale)

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_size))
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(val_size))
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(test_size))

# 20*5*8 = 800
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, sampler=valid_sampler, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,sampler=test_sampler ,num_workers=2)


if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

net = DMCell(inp_size, args.num_dm, **dm_dict).to(device)
criterion = nn.MSELoss()
optimizer = optimizer = optim.Adam(net.parameters(), lr=args.lr)

if not args.test:
	train_model_bp(net,criterion,optimizer,trainloader,validloader,args,layer_type="dm")
	if not args.valid:
		torch.save(net.state_dict(),args.log_dir+args.save_dir+"_net.pth")
	net_state_dict = torch.load(args.log_dir+args.save_dir+"_net.pth")
	net.load_state_dict(net_state_dict)
	acc = test_model(net,testloader,args,layer_type="dm")
	print("final test acc is ", acc)
	np.savetxt(args.log_dir+args.save_dir+"_acc.txt",[acc])
else:
	net_state_dict = torch.load(args.log_dir+args.save_dir+"_net.pth")
	net.load_state_dict(net_state_dict)
	print("model has been loaded!")
	acc=test_model(net,testloader,args,layer_type="dm")
	print("final test acc is ", acc)

# writer.close()
