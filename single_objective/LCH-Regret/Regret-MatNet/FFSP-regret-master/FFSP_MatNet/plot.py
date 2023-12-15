import torch
import numpy as np
import matplotlib.pyplot as plt

a = torch.load('checkpoint-200.pt', map_location='cpu')
val = np.asarray(a['result_log'][1]['train_score'])[:, 1]
b = torch.load('result/saved_ffsp20_model/checkpoint-200.pt', map_location='cpu')
val_b = np.asarray(b['result_log'][1]['train_score'])[:, 1]
plt.plot(val_b)
plt.plot(val)
plt.legend(['MatNet','MatNet-Regret'], fontsize=15, prop={'family': 'Times New Roman', 'size': 15})
plt.grid(alpha=0.35)
plt.xlabel('Epochs', {'family': 'Times New Roman', 'size': 15})
plt.ylabel('Obj.', {'family': 'Times New Roman', 'size': 15})
plt.title('Training FFSP-100', {'family': 'Times New Roman', 'size': 20})
plt.show()
