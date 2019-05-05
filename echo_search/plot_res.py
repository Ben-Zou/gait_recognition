import numpy as np
import glob
import pdb
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def stat_mean_std(path):
    files_sort = glob.glob(path)
    if files_sort == []:
        return None
    acc_list = []
    for fi in files_sort:
        with open(fi,'r') as f:
            acc_ = f.readline().split("\n")[0]
            acc_list.append(float(acc_))

    mean = np.mean(acc_list)
    std = np.std(acc_list)
    return (mean,std)

# path = "./res_raw/echo_bp_5way_a/*.txt"
# files_sort = glob.glob(path)
# acc_list = []
# for fi in files_sort:
#     with open(fi,'r') as f:
#         acc_ = f.readline().split("\n")[0]
#         acc_list.append(float(acc_))
#
# mean = np.mean(acc_list)
# std = np.std(acc_list)
#
# print("mean is ", mean)
# print("std is ", std)


# def stat(path):
#     files = glob.glob(path)
#     x = [5,20,50,80,100,200,500,1000]
#     files_sort = []
#
#     for i in range(len(x)):
#         j=0
#         for jj,fj in enumerate(files):
#
#             if str(x[i])+"_" in fj:
#                 files_sort.append(fj)
#                 print(fj)
#                 j+=1
#             if j>=4:
#                 break
#
#
#     trials_num = 4
#
#     acc_list = []
#     for fi in files_sort:
#         with open(fi,'r') as f:
#             acc_ = f.readline().split("\n")[0]
#             acc_list.append(float(acc_))
#
#     y = np.array(acc_list).reshape(len(x),trials_num)
#     mean = np.mean(y,axis=1)
#     std_deviation = np.std(y,axis=1)
#
#     return (x, mean, std_deviation)
#
#
# x, mean, std_deviation = stat("./res_rnn_frezon_oth/*/*acc.txt")
# x1, mean1, std_deviation1 = stat("./res_rnn/*/*acc.txt")


# plt.figure()
# plt.errorbar(x, mean, yerr=std_deviation, fmt="k-o",ecolor='r',label="T=5")
# plt.errorbar(x, mean1, yerr=std_deviation1, fmt="b-o",ecolor='r',label="T=10")
# plt.title("RNN_compare")
# plt.xlabel("Hidden Units Number")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("rnn_acc_compare.png")


# plt.figure()
# plt.errorbar(x, mean, yerr=std_deviation, fmt="k-o",ecolor='r',label="frozen othogonal")
# plt.errorbar(x, mean1, yerr=std_deviation1, fmt="b-o",ecolor='r',label="not frozen")
# plt.title("RNN_compare")
# plt.xlabel("Hidden Units Number")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("rnn_acc_frozen_com1.png")
#







rho=0.9

mean_09=[]
std_09=[]

for i in range(10,200,10):
    path = "./res/echo_bp_15way_a/10.0,{}.0,{}*.txt".format(i,rho)
    xy = stat_mean_std(path)
    if xy is None:
        continue

    mean_09.append(xy[0])
    std_09.append(xy[1])

rho=1.1

mean_11=[]
std_11=[]

for i in range(10,200,10):
    path = "./res/echo_bp_15way_a/10.0,{}.0,{}*.txt".format(i,rho)
    xy = stat_mean_std(path)
    if xy is None:
        continue

    mean_11.append(xy[0])
    std_11.append(xy[1])


rho=1.5

mean_15=[]
std_15=[]

for i in range(10,200,10):
    path = "./res/echo_bp_15way_a/10.0,{}.0,{}*.txt".format(i,rho)
    xy = stat_mean_std(path)
    if xy is None:
        continue

    mean_15.append(xy[0])
    std_15.append(xy[1])

rho=2.0

mean_20=[]
std_20=[]

for i in range(10,200,10):
    path = "./res/echo_bp_15way_a/10.0,{}.0,{}*.txt".format(i,rho)
    xy = stat_mean_std(path)
    if xy is None:
        continue

    mean_20.append(xy[0])
    std_20.append(xy[1])


pdb.set_trace()
print(len(mean_20))
tau = range(10,100,10)
plt.figure()
plt.errorbar(tau, mean_09, yerr=std_09, fmt="-o",label="rho=0.9,tau_res=10")
plt.errorbar(tau, mean_11, yerr=std_11, fmt="-o",label="rho=1.1,tau_res=10")
plt.errorbar(tau, mean_15, yerr=std_15, fmt="-o",label="rho=1.5,tau_res=10")
plt.errorbar(tau, mean_20, yerr=std_20, fmt="-o",label="rho=2.0,tau_res=10")
plt.legend()
plt.xlabel("Tau of DM")
plt.ylabel("Accuracy")
plt.savefig("bp_15way_stat.png")





#


















#
