import numpy as np
import scipy.sparse

H = scipy.sparse.load_npz("data/H-2.npz").toarray()
np.savez("data/H-2-dense.npz", H)