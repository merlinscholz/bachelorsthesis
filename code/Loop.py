import os
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.io as sio
import torch
from IPython.display import display
from joblib import Parallel, delayed
from scipy.stats import mode
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def run(image, tags, model, max_epochs=500, tensorboard=True, lr=0.15, momentum=0.4, filename = "", starttime = 0, stopping = ["gradient", -0.0025]):
  def gen_preview(tags, shape, colors = None):
    return label2rgb(tags.reshape(shape[0], shape[1]), colors=colors)
  def get_output_size(input, model):
    return np.array(model(input).permute(1, 2, 0).shape)

  def curve(x, a, b): 
    return a/(x)+b
    

  def curve_deriv(x, a, b):
    return (-a)/((x)**2)
  
  def gen_cells(tags):
    cells = []
    for c in np.unique(tags):
      cells.append(np.where(c == tags)[0])
    return cells
  
  #colors = np.random.randint(255, size=(1000, 3))
  model = model.cuda()
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  if image.shape[2] == 3:
    preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  else:
    preprocess = transforms.Compose([
      transforms.ToTensor(),
    ])
  
  input = preprocess(image.copy()).unsqueeze(0).cuda()

  target_shape = get_output_size(input, model)
  #target_shape = image.shape
    
  if target_shape[0] != image.shape[0] or target_shape[1] != image.shape[1]:
    tags = cv2.resize(tags.reshape(image.shape[0], image.shape[1]), dsize=(target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST).flatten()

  cells = gen_cells(tags)
    
  if tensorboard:
    tb = SummaryWriter("{}/{}/{}".format("tensorboard", starttime, filename))
    tb.add_image("input/", image, 0, dataformats="HWC")
    tb.add_image("clustered/", gen_preview(tags, target_shape), 0, dataformats="HWC")

  loss_history = []
  loss_delta = np.zeros(max_epochs)
  current_delta = np.inf
  popt1 = None
  for epoch in range(max_epochs):
    raw = model(input)
    predicted = raw.permute(1, 2, 0).view(target_shape[0]*target_shape[1], -1)
    argmax = torch.argmax(predicted, dim=1).cpu().numpy()
    n_labels = len(np.unique(argmax))

    if epoch%3==0:
      target = np.zeros_like(argmax)
      for cell in cells:
        possible = argmax[cell].flatten()
        target[cell] = mode(possible)[0]
        
    optimizer.zero_grad()
    loss = loss_fn(predicted, torch.from_numpy(target).cuda().long())
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    loss_delta[epoch] = loss.item()
    
    #test = poly.polyfit(np.arange(len(loss_history)), np.array(loss_history), 2)
    #testder = poly.polyder(test)
    
    if(len(loss_history)>=25):
        popt1, pcov1 = curve_fit(curve, np.arange(20, len(loss_history)).astype(np.float64), np.array(loss_history)[20:].astype(np.float64))
    
    if epoch >= 20:
      current_delta = np.mean(loss_delta[epoch-19:epoch]-loss_delta[epoch-20:epoch-1])
    
    if tensorboard:
      if(len(loss_history)>=25):
        tb.add_scalar("approx/cov", np.sum(np.sqrt(np.diag(pcov1))), epoch)
        tb.add_scalar("approx/curve", curve(epoch, *popt1), epoch)
        tb.add_scalar("approx/deriv", curve_deriv(epoch, *popt1), epoch)
      tb.add_scalar("loss/delta_over_20", current_delta, epoch)
      tb.add_scalar("loss/loss", loss.item(), epoch)
      tb.add_scalar("labels/", n_labels, epoch)
      tb.add_image("target/", gen_preview(target, target_shape), epoch, dataformats="HWC")
      tb.add_image("preview/", gen_preview(argmax, target_shape), epoch, dataformats="HWC")
      tb.flush()

    if stopping[0] == "segments" and n_labels <= stopping[1]:
      break
    
    if epoch>=25:
        if (stopping[0] == "gradient" and curve_deriv(epoch, *popt1) >= stopping[1]) or n_labels<3:
          break

  if tensorboard and popt1 is not None:
    for i in range(1, max_epochs):
      tb.add_scalar("approx_final/curve", curve(i, *popt1), i)
      tb.add_scalar("approx_final/deriv", curve_deriv(i, *popt1), i)
    
    
    
  returns = {}
  returns["labels"] = argmax.astype(np.uint8).reshape(target_shape[0], target_shape[1])
  returns["epochs"] = epoch
  returns["n_labels"] = n_labels
  return returns
