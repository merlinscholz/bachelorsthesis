import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def applyFilter(A, g, n_clusters = 50, no_blur = False, no_spatial = False, no_norm = False, no_color = False, r_bank_results = False, mr = False, size = 0.75, w_color = 0, w_spatial = 0.66):
  if no_spatial == True:
    w_spatial = 0

  if no_color == True:
    w_color = 0

  if len(A.shape) == 2:
    A = np.expand_dims(A, axis = 2)

  if A.shape[2] == 3:
    Agray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
  elif A.shape[2] == 1:
    Agray = A

  numRows = Agray.shape[0]
  numCols = Agray.shape[1]
  sizeFactors = np.array(size)
  gabormag = np.ndarray(
    (Agray.shape[0], Agray.shape[1], g.shape[2]))
  sigmas = np.ndarray((g.shape[2] + 1))

  for i in range(g.shape[2]):
    gabormag[:, :, i] = cv2.filter2D(Agray, -1, cv2.resize(g[: ,: , i], (int(g[: ,: , i].shape[0]*size),
        int(g[: ,: , i].shape[1]*size)), interpolation = cv2.INTER_LANCZOS4), borderType = cv2.BORDER_REPLICATE)
    sigmas[i] = np.sqrt(2) * \
        g.shape[0] * size / 49

  if r_bank_results == True:
    returns = []
    for i in range(gabormag.shape[2]):
      returns.append(gabormag[: ,: , i])
    return returns
  
  if no_blur == False:
    for i in range(gabormag.shape[2]):
      gabormag[: ,: , i] = cv2.GaussianBlur(
        gabormag[: ,: , i], (0, 0), 3 * sigmas[i])

  X = np.arange(1, numCols + 1)
  Y = np.arange(1, numRows + 1)
  X, Y = np.meshgrid(X, Y)

  numPoints = numRows * numCols
  featureSet = gabormag

  if mr is True:
    argsort = np.argsort(featureSet[: ,: ,: -2].sum(axis = (0, 1)))
    maxresp = featureSet[: ,: , argsort]
    maxresp = maxresp[:, :, :6]
    rotinv = featureSet[: ,: , -2: ]
    featureSet = np.concatenate((maxresp, rotinv), 2)
    print(featureSet.shape)

  if A.shape[2] == 3:
    color_sums = A.sum(axis=2)
    color_avg = A / color_sums[:, :, np.newaxis]
    plt.figure()
    plt.imshow(color_avg)
    plt.show()
    featureSet = np.concatenate(
      (featureSet, np.expand_dims(color_avg[: ,: , 0], axis = 2)), 2)
    featureSet = np.concatenate(
      (featureSet, np.expand_dims(color_avg[: ,: , 1], axis = 2)), 2)
    featureSet = np.concatenate(
      (featureSet, np.expand_dims(color_avg[: ,: , 2], axis = 2)), 2)
  elif A.shape[2] == 1:
    featureSet = np.concatenate(
      (featureSet, np.expand_dims(A[: ,: , 0], axis = 2)), 2)

  featureSet = np.concatenate((featureSet, np.expand_dims(X, axis = 2)), 2)
  featureSet = np.concatenate((featureSet, np.expand_dims(Y, axis = 2)), 2)

  X = featureSet.reshape(numPoints, -1)

  X = X[: , ~np.isnan(X).any(axis = 0)]
  X = X[: , ~np.isinf(X).any(axis = 0)]
  if no_norm == False:
    X = X - X.mean(axis = 0)
    X = X / X.std(axis = 0, ddof = 1)

  X = X.reshape(A.shape[0], A.shape[1], -1)

  X[: ,: , -2: ] = X[: ,: , -2: ] * w_spatial
  if A.shape[2] == 3:
    X[: ,: , -5: -3] = X[: ,: , -5: -3] * w_color
  elif A.shape[2] == 1:
    X[: ,: , -3] = X[: ,: , -3] * w_color
  X = X.reshape(numPoints, -1)

  L = KMeans(n_clusters = n_clusters, n_init = 8,
    max_iter = 100, n_jobs = 16).fit(X).labels_

  return L