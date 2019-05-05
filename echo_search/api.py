import torch
import math
import sys
sys.path.append("../")
import pdb
import numpy as np
import torch.nn as nn
from layers import dm_echo3
from rnn import LogAct
from rnn import RecLogAct
import torch.nn.functional as F
import torch.optim as optim
import torchvision as T
import os
from utils import plot_r

cuda = torch.cuda.is_available()

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


def train_model_bp(model,criterion,optimizer,trainloader,validloader,args,layer_type="echo1_dm"):
	"""
	layer_type: "echo1_dm", "echo3_dm" , "dm"
	"""
	epochs = args.n_epochs
	lr = args.lr
	T = args.T
	batch_size = args.batch_size
	num_dm = args.num_dm

	best_acc = 0.0
	for epoch in range(epochs):
		print('epoch ' + str(epoch + 1))
		iter_ctr = 0.
		train_acc = 0
		for data_i in trainloader:
			iter_ctr+=1.
			inp_x, inp_y = data_i
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)

			# train
			r_list = []
			loss=0
			rr = torch.zeros(batch_size,num_dm).to(device)


			## input--output

			if layer_type == "echo3_dm":
				u_echo1 = torch.zeros((batch_size,args.num_echo1)).to(device)
				h_echo1 = torch.zeros((batch_size,args.num_echo1)).to(device)
				u_echo2 = torch.zeros((batch_size,args.num_echo2)).to(device)
				h_echo2 = torch.zeros((batch_size,args.num_echo2)).to(device)
				u_echo3 = torch.zeros((batch_size,args.num_echo3)).to(device)
				h_echo3 = torch.zeros((batch_size,args.num_echo3)).to(device)
				s_dm = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (u_echo1,h_echo1,
							u_echo2,h_echo2,
							u_echo3,h_echo3,
							s_dm) = model(inp_x[:,i], (u_echo1,h_echo1,
															u_echo2,h_echo2,
															u_echo3,h_echo3,
															s_dm))    #inp_y[:,i]



					rr.copy_(r)
					r_list.append(rr.cpu().detach().numpy())
					loss += criterion(r, inp_y[:,i])

			if layer_type == "echo1_dm":
				u_echo = torch.zeros((batch_size,args.num_echo)).to(device)
				h_echo = torch.zeros((batch_size,args.num_echo)).to(device)
				s_dm = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (u_echo,h_echo,
						s_dm) =  model(inp_x[:,i], (u_echo,h_echo,s_dm))

					rr.copy_(r)
					r_list.append(rr.cpu().detach().numpy())
					loss += criterion(r, inp_y[:,i])

			if layer_type == "dm":
				s = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (s_dm,) =  model(inp_x[:,i], (s,))

					rr.copy_(r)
					r_list.append(rr.cpu().detach().numpy())
					loss += criterion(r, inp_y[:,i])


			model.zero_grad()
			loss.backward()
			norms = nn.utils.clip_grad_norm_(model.parameters(), args.clipval)
			optimizer.step()
			loss_val = loss.item()

			labels = inp_y[:,-1].argmax(dim=1)
			r_list = np.array(r_list).squeeze()
			preds = r_list[-30:].mean(axis=0).argmax()
			corrects = torch.sum(preds == labels).cpu().item()
			train_acc += corrects

			if args.plot and epoch == epochs-1:
				if iter_ctr<=10:
					lab = [0]*args.num_dm
					lab[labels[0]]=1
					plot_r(r_list,args.log_dir+"train/trail_{}".format(iter_ctr),label=lab)

		if args.valid:
			valid_acc = test_model(model,validloader,args,layer_type=layer_type)

			if best_acc < valid_acc:
				best_acc = valid_acc
				torch.save(model.state_dict(),args.log_dir+args.save_dir+"_net.pth")

		print("tot test number is {}, \n trail loss is {},\n train acc is {}".format(iter_ctr,
											loss_val,np.array(train_acc)/iter_ctr))
		print("best acc is {}".format(best_acc))



def test_model(model,testloader,args,layer_type="echo1_dm"):
	batch_size = args.batch_size
	T = args.T
	ctr = 0
	test_acc = 0
	with torch.no_grad():
		for data_i in testloader:
			inp_x, inp_y = data_i
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)


			r_list = []
			## input--output

			if layer_type == "echo3_dm":
				u_echo1 = torch.zeros((batch_size,args.num_echo1)).to(device)
				h_echo1 = torch.zeros((batch_size,args.num_echo1)).to(device)
				u_echo2 = torch.zeros((batch_size,args.num_echo2)).to(device)
				h_echo2 = torch.zeros((batch_size,args.num_echo2)).to(device)
				u_echo3 = torch.zeros((batch_size,args.num_echo3)).to(device)
				h_echo3 = torch.zeros((batch_size,args.num_echo3)).to(device)
				s_dm = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (u_echo1,h_echo1,
							u_echo2,h_echo2,
							u_echo3,h_echo3,
							s_dm) = model(inp_x[:,i], (u_echo1,h_echo1,
															u_echo2,h_echo2,
															u_echo3,h_echo3,
															s_dm))    #inp_y[:,i]


					r_list.append(r.cpu().detach().numpy())

			if layer_type == "echo1_dm":
				u_echo = torch.zeros((batch_size,args.num_echo)).to(device)
				h_echo = torch.zeros((batch_size,args.num_echo)).to(device)
				s_dm = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (u_echo,h_echo,
						s_dm) =  model(inp_x[:,i], (u_echo,h_echo,s_dm))
					r_list.append(r.cpu().detach().numpy())

			if layer_type == "dm":
				s = torch.zeros(batch_size,args.num_dm).to(device)

				for i in np.arange(T):
					r, (s_dm,) =  model(inp_x[:,i], (s,))
					r_list.append(r.cpu().detach().numpy())

			labels = inp_y[:,-1].argmax(dim=1)
			r_list = np.array(r_list).squeeze()
			preds = r_list[-30:].mean(axis=0).argmax()
			corrects = torch.sum(preds == labels).cpu().item()
			test_acc += corrects

			if args.plot:
				lab = [0]*args.num_dm
				lab[labels[0]]=1
				plot_r(r_list,args.log_dir+"test/trail_{}".format(ctr),label=lab)

			ctr += 1
		return test_acc/ctr
