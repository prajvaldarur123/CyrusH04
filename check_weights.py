import numpy as np
w = np.load(r'C:\Users\prajval\robotics\cyrus\checkpoints\balance_12dof_weights.npz')
for k in w.files:
    print(k, w[k].shape, np.mean(w[k]), np.std(w[k]))
print("bias:", np.round(w['action_net.bias'], 3))
