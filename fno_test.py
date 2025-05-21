import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from neuralop.models import FNO


npz_path = 'data.npz'
# Load cached archive
pkg = np.load(npz_path)
data = pkg['combined']
print(f"Loaded {npz_path}: combined{data.shape}")


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

i=2
ax.scatter(data[2, :, i], data[2, :, i+1], data[2, :, i+2])

plt.show()