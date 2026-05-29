import scipy.sparse as sp
import numpy as np
import pandas as pd

df = pd.read_csv("data/H-1.csv", header=None)
H1 = sp.csr_matrix(df.to_numpy(dtype=np.float32))
sp.save_npz("data/H-1.npz", H1)
