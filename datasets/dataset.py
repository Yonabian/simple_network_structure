import torch
from torch.utils.data import Dataset,DataLoader

class NetData():
    pass

# need to know how dataloader work here -> bottleneck
# bottleneck occer when you look at the GPU utility the percentage fluctuated widely

#in order to overcome that, using tfrecord 1e4->1e3*10