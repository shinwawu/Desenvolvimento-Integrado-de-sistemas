import numpy as np
import scipy.sparse as sp
import pandas as pd
import threading

models_h = {
    "60x60": {
        "S": 50816, "N": 3600, "shape": (60, 60),
        "path": "H-1.npz", 
        "matrix": None,
        "prioridade": "baixa",
        "thread_lock": threading.Lock(),
        "in_use": False,
    },
    "30x30": {
        "S": 27904, "N": 900, "shape": (30, 30),
        "path": "H-2.npz", 
        "matrix": None,
        "prioridade": "alta",
        "thread_lock": threading.Lock(),
        "in_use": False,
    }
}


def get_modelo(model_id: str):
    model_config = models_h[model_id]
    if model_config["matrix"] is not None:
        return model_config["matrix"]
    with model_config["thread_lock"]:
        if model_config["matrix"] is not None:
            return model_config["matrix"]
        print(f"\n Loading Model {model_id} (Model_config: {model_config['path']})...\n")
        try:
            matrix = sp.load_npz(model_config['path'])
            S_esperado, N_esperado = model_config['S'], model_config['N']
            if matrix.shape != (S_esperado, N_esperado):
                print(f"ERRO LAZY LOAD: Dimensões erradas em {model_config['path']}!")
                raise ValueError("Dimensões do modelo não correspondem ao esperado.")
            models_h[model_id]["matrix"] = matrix.tocsc()
            print(f"[LAZY LOAD]: Modelo {model_id} carregado e armazenado no cache.")
            return models_h[model_id]["matrix"]
        except FileNotFoundError:
            print(f"ERRO LAZY LOAD: Arquivo {model_config['path']} não encontrado!")
            raise
    

def CGNR_function(model_id: str, g: np.ndarray, tol: float = 1e-5, max_iter: int = 10):
    f = np.zeros_like(g)
    r = g - get_modelo(model_id) @ f