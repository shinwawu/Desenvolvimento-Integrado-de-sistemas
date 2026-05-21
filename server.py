import numpy as np
import scipy.sparse as sp
import pandas as pd
import threading
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
models_h = {
    "60x60": {
        "S": 50816,
        "N": 3600,
        "shape": (60, 60),
        "path": "data/H-1.npz",
        "matrix": None,
        "prioridade": "baixa",
        "thread_lock": threading.Lock(),
        "in_use": False,
    },
    "30x30": {
        "S": 27904,
        "N": 900,
        "shape": (30, 30),
        "path": "data/H-2.npz",
        "matrix": None,
        "prioridade": "alta",
        "thread_lock": threading.Lock(),
        "in_use": False,
    },
}


def get_modelo_matrix(model_id: str):
    model_config = models_h[model_id]
    if model_config["matrix"] is not None:
        return model_config["matrix"]
    with model_config["thread_lock"]:
        if model_config["matrix"] is not None:
            return model_config["matrix"]
        print(
            f"\n Loading Model {model_id} (Model_config: {model_config['path']})...\n"
        )
        try:
            matrix = sp.load_npz(model_config["path"])

            S_esperado, N_esperado = model_config["S"], model_config["N"]
            if matrix.shape != (S_esperado, N_esperado):
                print(f"ERRO LAZY LOAD: Dimensões erradas em {model_config['path']}!")
                raise ValueError("Dimensões do modelo não correspondem ao esperado.")
            models_h[model_id]["matrix"] = matrix.tocsc()
            print(f"[LAZY LOAD]: Modelo {model_id} carregado e armazenado no cache.")
            return models_h[model_id]["matrix"]
        except FileNotFoundError:
            print(f"ERRO LAZY LOAD: Arquivo {model_config['path']} não encontrado!")
            raise


def implementar_cgnr(H_sparse_csc, g, max_iter=10, tol=1e-4):
    f = np.zeros(H_sparse_csc.shape[1])
    r = g.copy()
    z = H_sparse_csc.transpose() @ r
    p = z.copy()
    norm_z_sq = np.dot(z, z)
    iteracoes = 0
    erro_atual = np.linalg.norm(r)
    for i in range(max_iter):
        iteracoes = i + 1
        w = H_sparse_csc @ p
        norm_w_sq = np.dot(w, w)
        if norm_w_sq == 0:
            break
        alpha = norm_z_sq / norm_w_sq
        f = f + alpha * p
        r = r - alpha * w
        erro_atual = np.linalg.norm(r)
        if erro_atual < tol:
            break
        z_new = H_sparse_csc.transpose() @ r
        norm_z_new_sq = np.dot(z_new, z_new)
        if norm_z_sq == 0:
            break
        beta = norm_z_new_sq / norm_z_sq
        p = z_new + beta * p
        norm_z_sq = norm_z_new_sq
    return f, iteracoes, erro_atual


def CGNR_function(
    matrix_sparse: sp.csc_matrix, g: np.ndarray, tol: float = 1e-5, max_iter: int = 10
):
    g = np.asarray(g, dtype=np.float64).ravel()
    f = np.zeros(matrix_sparse.shape[1], dtype=np.float64)
    r = g - (matrix_sparse @ f)
    z = matrix_sparse.T @ r
    p = z.copy()
    norm_z_sq = float(np.dot(z, z))
    error = float(np.linalg.norm(r))

    for k in range(max_iter):
        w = matrix_sparse @ p
        norm_w_sq = float(np.dot(w, w))
        if norm_w_sq == 0.0:
            break

        alpha = norm_z_sq / norm_w_sq

        f = f + alpha * p
        r = r - alpha * w

        error = float(np.linalg.norm(r))
        if error < tol:
            break

        z_new = matrix_sparse.T @ r
        norm_z_new_sq = float(np.dot(z_new, z_new))
        if norm_z_sq == 0.0:
            break

        beta = norm_z_new_sq / norm_z_sq
        p = z_new + beta * p
        norm_z_sq = norm_z_new_sq

    return f, k + 1, error


def run_cgnr(model_id: str, g: np.ndarray):
    config = models_h[model_id]
    H_matrix = get_modelo_matrix(model_id)
    if H_matrix is None:
        print(f"Erro: Modelo {model_id} não pôde ser carregado.")
        return None
    f, iteracoes, erro_final = CGNR_function(H_matrix, g, 1e-5, 10)
    print(
        f"modelo {model_id}: convergiu em {iteracoes} iteracoes com erro final {erro_final:.6e}"
    )
    img = construir_imgem(f, config["shape"])

    return img


def construir_imgem(f: np.ndarray, shape: tuple):
    f_img = f.reshape(shape, order="F")
    f_img_normalized = (f_img - f_img.min()) / (f_img.max() - f_img.min())
    return f_img_normalized


class Sinal(BaseModel):
    g: list[float]


@app.post("/reconstruct/{model_id}", status_code=200)
async def reconstruct(model_id: str, sinal: Sinal):
    if model_id not in models_h:
        return {"error": f"Modelo {model_id} não encontrado."}

    g_array = np.array(sinal.g, dtype=np.float64)
    img = run_cgnr(model_id, g_array)

    if img is None:
        return {"error": f"Falha na reconstrução para o modelo {model_id}."}

    return {
        "message": f"Reconstrução para o modelo {model_id} finalizada.",
        "image": img.tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
