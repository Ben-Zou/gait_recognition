import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_r(x,name,label):
    # x ()
    T,_ = x.shape
    plt.figure()
    for i, xi in enumerate(x.T):
        plt.plot(xi,label=label[i],lw=2)
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Neuron Activity")
    plt.savefig(name+".png")




def target_win(x,num_dm):
    b = 0.8
    small_val = 0.1
    if num_dm == 5:
        Je = 9.
        I0 = 0.

    if num_dm == 10:
        Je = 18.
        I0 = 1.1

    if num_dm == 15:
        Je = 20.
        I0 = 0.4

    if num_dm == 20:
        Je = 27.
        I0 = 0.3

    return Je*(np.tanh(b*x)+1)/2. + I0 + small_val

def target_loss(x,num_dm):
    b = 0.8
    small_val = 0.1
    if num_dm == 5:
        Jm = -5.
        I0 = 0.

    if num_dm == 10:
        Jm = -11.
        I0 = 1.1

    if num_dm == 15:
        Jm = -20.
        I0 = 0.4

    if num_dm == 20:
        Jm = -27.
        I0 = 0.3

    return (Jm + I0 - small_val)*np.ones_like(x)
