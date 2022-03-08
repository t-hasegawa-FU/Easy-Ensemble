# -*- coding: utf-8 -*-
'''
@Time          : 22/03/08 17:07
@Author        : t-hasegawa-FU
@File          : main.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import datasets
from models import vgg

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def count_params(net):
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

def create_random_data(n, in_channels=3, labels=6):
    # Define the data shape (channels, win-size)
    shape = (n, in_channels, 256)
    x = np.random.random(size = shape).astype(np.float32)
    y = np.random.randint(0, labels, n)
    return x, y

def train(m, dl, epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for e in range(epochs):
        loss = 0
        for x, y in dl:
            x, y = x.to(device), y.type(torch.long).to(device)
            opt.zero_grad()
            l = criterion(m(x), y)
            l.backward()
            opt.step()
            loss += l
            losses.append(l.to("cpu").detach().numpy())
        print("\r epoch {} \t train loss {}".format(e, loss), end="")
    return losses

if __name__ == '__main__':
    ch = 12   # the number of channels of input (axes of sensor values)
    d = create_random_data(1000, in_channels=ch)
    ds = datasets.create_datasets(d, d, d, input_repeat=1)  # input_repeat can be set both in dataset and model
    N=4  # the number of ensembles

    # BL: original vgg model
    m_bl = vgg.create_vgg8(6, ch, nb_fils=16, c_groups=1, n_groups=1, repeat_input=1, lmd=1)
    # BL: original vgg model (scaled up by fileter num)
    m_bl2 = vgg.create_vgg8(6, ch, nb_fils=16*2, c_groups=1, n_groups=1, repeat_input=1, lmd=1)
    # EE: EE-style vgg
    m_ee = vgg.create_vgg8(6, ch*N, nb_fils=16*N, c_groups=N, n_groups=N, repeat_input=1, lmd=1)
    # EE: EE-stype vgg (modality ensemble)
    m_ee_mod = vgg.create_vgg8(6, ch, nb_fils=16*N, c_groups=N, n_groups=N, repeat_input=1, lmd=1)

    # parameters
    print(f"BL: {count_params(m_bl)}")
    print(f"BL(x2): {count_params(m_bl2)}")
    print(f"EE(IR4): {count_params(m_ee)}")
    print(f"EE(mod): {count_params(m_ee_mod)}")

    # training
    dl = DataLoader(ds[0], batch_size=200, shuffle=True)
    losses = train(m_bl, dl)
    losses = train(m_bl2, dl)
    losses = train(m_ee_mod, dl)

    # training (Input Repeat)
    ds_ir = datasets.create_datasets(d, d, d, input_repeat=N)
    dl = DataLoader(ds_ir[0], batch_size=200, shuffle=True)
    losses = train(m_ee, dl)

    # training (Input Masking)
    ds_im = datasets.create_masked_datasets(d, d, d, N, type="random")
    dl = DataLoader(ds_im[0], batch_size=200, shuffle=True)
    losses = train(m_ee, dl)

    # training (Augmented)
    ds_aug = datasets.create_augmented_datasets(d, d, d, type="fixed")
    dl = DataLoader(ds_aug[0], batch_size=200, shuffle=True)
    losses = train(m_ee, dl)
