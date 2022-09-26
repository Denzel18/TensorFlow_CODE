import torch

'''MMD METHOD DEFINITION
- Works correctly only with 2D tensor inputs! (so, for color images in batches)
'''


KERNEL_TYPE = "rbf"
def custom_MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)


'''
#ESEMPIO DI MMD


%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet 
from torch.distributions.multivariate_normal import MultivariateNormal 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = 20 # sample size
x_mean = torch.zeros(2)+1
y_mean = torch.zeros(2)
x_cov = 2*torch.eye(2) # IMPORTANT: Covariance matrices must be positive definite
y_cov = 3*torch.eye(2) - 1

px = MultivariateNormal(x_mean, x_cov)
qy = MultivariateNormal(y_mean, y_cov)
x = px.sample([m]).to(device)
y = qy.sample([m]).to(device)

result = MMD(x, y, kernel="multiscale")
print(result)
'''

'''
#ESEMPIO 2 - DENIS

mmd_average = 0
mmd_sum = 0
for kk in range( real_imgs.shape[0]):
  #--- serve a (62,1, 256, 256) a (256,256) real_imgs[0].permute(1, 2, 0)[:, :, -1]
  transform = transforms.Grayscale()
  _real_imgs_ = transform(real_imgs[k])
  _gen_imgs_ = transform(gen_imgs[k])
  print(_real_imgs_.shape)
  print(_gen_imgs_.shape)
  _real_imgs_test2 = _real_imgs_.permute(1, 2, 0)
  print(_real_imgs_test2.shape)
  _real_imgs_test = _real_imgs_.permute(1, 2, 0)[:, :, -1]      #coi : mantengo la colonna, con il -1 la rimuovo
  print(_real_imgs_test.shape)


  mmd = MMD(_real_imgs_.permute(1, 2, 0)[:, :, -1], _gen_imgs_.permute(1, 2, 0)[:, :, -1], KERNEL_TYPE)     #permute serve per riordinare le dimensioni del tensore, a noi serve un reshape!
  #mmd = MMD(_real_imgs_, _gen_imgs_, KERNEL_TYPE)
  print("ok mmd")

  real_imgs_ = torch.from_numpy(real_combined)

  mmd = MMD(real_imgs_, gen_imgs_, KERNEL_TYPE)
  mmd_sum = mmd_sum + mmd
  #print('MMD : {}'.format(mmd))
  mmd_average = mmd_sum/real_imgs.shape[0]

'''