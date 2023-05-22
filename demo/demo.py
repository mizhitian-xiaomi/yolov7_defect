import torch
x1 = torch.rand(1, 256, 23, 40,dtype=torch.float32)
x2 = torch.rand(1, 256, 22, 40,dtype=torch.float32)
y = torch.cat([x1,x2],1)

