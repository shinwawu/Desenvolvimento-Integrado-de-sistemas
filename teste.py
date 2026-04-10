import pandas as pd
import numpy as np

# 1. Carregar o arquivo CSV
df = pd.read_csv("data/H-2.csv")

# 2. Converter para um array NumPy
# Se o CSV for todo numérico:
dados_array = df.to_numpy()

# 3. Salvar como NPZ (compactado)
# Você pode dar nomes aos arrays dentro do arquivo
np.savez_compressed("data/H-2.npz", dados=dados_array)
