# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:08:11 2019

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt

dir_his = "exp/history/"
idx=5
emo=0
lamb_cyc = 10
title_name = '[val'+str(idx)+'emo'+str(emo)+']  lamb_cyc ='+str(lamb_cyc)+' lamb_id = '+str(lamb_cyc*0.5)
#idx_name = dir_his+"val"+str(idx)+"emo"+str(emo)
idx_name  = dir_his+"test1D"+str(idx)+"emo"+str(emo)
history_d = np.load('%s_history_d.npy' %idx_name)
history_g = np.load('%s_history_g.npy' %idx_name)


xd = history_d[:,[0]]
xg = history_g[:,[0]]
xa = (history_g[:,[1]] + history_g[:,[2]])/2
xr = (history_g[:,[3]] + history_g[:,[4]])/2
xi = (history_g[:,[5]] + history_g[:,[6]])/2
xsd = []
xsg = []
xsa = []
xsr = []
xsi = []
window = 1000
for i in range(len(xd)-window+1):
    xsd.append(np.mean(xd[i:i+window]))
    xsg.append(np.mean(xg[i:i+window]))
    xsa.append(np.mean(xa[i:i+window]))
    xsr.append(np.mean(xr[i:i+window]))
    xsi.append(np.mean(xi[i:i+window]))
fig, loss_ax = plt.subplots()
#acc_ax = loss_ax.twinx()

loss_ax.plot(xsd, 'b', label='Discriminator')
loss_ax.plot(xsa, 'r', label='G_adv')
loss_ax.plot(xsr, 'g', label='G_rec')
loss_ax.plot(xsi, 'y', label='G_id')
loss_ax.plot(xsg/max(xsg), 'k', label='G_sum')
loss_ax.set_xlabel('mini-batchs')
loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
loss_ax.set_title(title_name)
#acc_ax.legend(loc='lower left')
plt.show()