import os
import random
import sys
import numpy as np
import datetime

#write here or command is both fine
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"0,1" if you want to use both
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # 1 means make every thing synchronized, only use for debuging

import torch
import torch.nn functional as F
import pretty_errors

from models.Net import Net
from models.utils import get_large_gradient

torch.backends.cudnn.enabled = False #allow to use cudnn
torch.backends.cudnn.deterministic = False #true will make program slower
torch.backends.cudnn.benchmark = True #true will analises network structures, could run with true for once and false for once, choose the better one

loss_log_every = 1
eval_every = 100
ckpoint_every = 1000
resume_ckpt= True
resume_checkpoint_path = ''
net = Net().cuda()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)

if resume_ckpt:
    checkpoint = torch.load(resume_checkpoint_path,map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    counter = checkpoint['iter_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 



train_loader = None
eval_loader = None

for data in train_loader:
    # train loop

    loss,acc,reg = net(data)
    optimizer.zero_grad()
    loss.backward()
    max_grad,min_grad = get_large_gradient(net)
    torch.nn.utils.clip_grad_value_(net.parameters(),10) #avoid large gradient
    optimizer.step()

    if counter %loss_log_every:
        #get current time
        ps = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ps += 'train'
        ps +=f'loss:{float(loss):.3f}'
        ps +=f'acc:{float(acc):.3f}'
        ps +=f'reg:{float(reg):.3f}'
        print(ps)

    if counter %eval_every:
        with torch.no_grad():# do't compute gradient
            net.eval()
            for eval_data in eval_loader:
                loss,acc,reg = net(eval_data)
            net.train()
    if counter%ckpoint_every:
        ckpt_file = 'ckpt' + str(counter)
        state_dict = net.state_dict() #all parameters
        ckpt_dict = {'iter_idx':counter,'model_state_dict':state_dict,'optimizer_state_dict':optimizer.state_dict()}
        torch.save(ckpt_dict,ckpt_file)
    
    counter+=1

