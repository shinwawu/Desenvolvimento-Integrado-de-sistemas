import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
# requisitos funcionais:
# TODO :
# - o servidor deve ser ter os algoritmos de reconstrucao CGNR e CGNE implementados. (CGNR feito, falta o CGNE)
# - Executar até que o erro (𝜖) seja menor do que 1e10-4  ou o número de iteração chegar a 10. (CGNE falta implementar)
# - Criar um relatório comparativo analisando os resultados obtidos com as duas versões. (falta o CGNE para comparar)
# - O objetivo principal deve ser reconstruir o maior número de imagens no menor tempo possível. (feito, falta o CGNE para comparar)



# Configurações dos modelos disponíveis
MODELS_CONFIG = {
   "60x60": {"S": 50816, "N": 3600, "shape": (60, 60), "path": "data/H-1.npz"},
   "30x30": {"S": 27904, "N": 900,  "shape": (30, 30), "path": "data/H-2.npz"},
}
# listar os modelos disponíveis
MODELS: dict = {}

# listar as sequencias de sinais g recebidos de cada cliente armazenar os sinais recebidos em partes, e só executar a reconstrução quando o cliente indicar que o envio está completo
CLIENT_SIGNALS: dict = {
    "cliente_id": {
        "model_id": "",
        "g_parts": [],
        "complete": False
    }
}

# carregar o modelo quand
def load_model(model_id: str, cfg: dict) -> dict:
    path = Path(cfg["path"])
    if not path.exists():
        print(f"error: {path} matriz nao encontrado do modelo {model_id}")
        return None

    H = sp.load_npz(path)
    if H.shape != (cfg["S"], cfg["N"]):
        raise ValueError(f"{path} diferenca no tamanho {H.shape},esperado {(cfg['S'], cfg['N'])}")

    # converter para float32 e csr para melhor desempenho
    H = H.astype(np.float32).tocsr()
    Ht = H.T.tocsr()  

    
    print(f"modelo carregado {model_id}: matriz h={H.shape}, matriz h transposta={Ht.shape}")
    return {"H": H, "Ht": Ht, "shape": cfg["shape"], "S": cfg["S"], "N": cfg["N"]}

# gerencia o ciclo de vida do app, e carrega os modelos ao iniciar e limpa ao finalizar
@asynccontextmanager
async def lifespan(app: FastAPI):
    for mid, cfg in MODELS_CONFIG.items():
        model = load_model(mid, cfg)
        if model is not None:
            # se o modelo for carregado com sucesso, adiciona ao dicionário global
            MODELS[mid] = model
    # yield serve para indicar que a inicialização está completa e o app pode começar a aceitar requisições
    yield
    # ao finalizar, limpa os modelos para liberar memória
    MODELS.clear()
# cria a instancia do fastapi
# definindo o ciclo de vida do app para carregar os modelos ao iniciar e limpar ao finalizar
# optamos por usar ORJSONResponse para melhorar a performance na serialização de respostas JSON
app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)

# função de reconstrucao usando o metodo CGNR
def cgnr_function(matriz_h: sp.csr_matrix, matriz_h_t: sp.csr_matrix, g: np.ndarray,
        max_iter: int = 10, tol: float = 1e-5):

   f = np.zeros(matriz_h.shape[1], dtype=np.float32)
   r = g.copy()
   z = matriz_h_t @ r
   p = z.copy()
   norm_z_sq = float(z @ z)
   err = float(np.linalg.norm(r))
   for k in range(max_iter):
       w = matriz_h @ p
       norm_w_sq = float(w @ w)
       if norm_w_sq == 0.0:
           return f, k, err
       alpha = norm_z_sq / norm_w_sq
       f += alpha * p
       r -= alpha * w
       err = float(np.linalg.norm(r))
       if err < tol:
           return f, k + 1, err
       z = matriz_h_t @ r
       norm_z_new_sq = float(z @ z)
       if norm_z_sq == 0.0:
           return f, k + 1, err
       beta = norm_z_new_sq / norm_z_sq
       p *= beta
       p += z
       norm_z_sq = norm_z_new_sq
   return f, max_iter, err

def cgne_function(matriz_h: sp.csr_matrix, matriz_h_t: sp.csr_matrix, g: np.ndarray,
        max_iter: int = 10, tol: float = 1e-5):
    f = np.zeros(matriz_h.shape[1], dtype=np.float32)
    r = g.copy() - matriz_h @ f
    p = matriz_h_t @ r
    for k in range(max_iter):
        a = r.transpose() @ r / (p.transpose() @ p)
        f += f + a * p
        r -= a * matriz_h @ p

        err = float(np.linalg.norm(r))
        if err < tol:
            return f, k + 1, err
        beta = r.transpose() @ r / (a * p.transpose() @ p)
        p = matriz_h_t @ r + beta * p
    return f, max_iter, err

def reconstruct_image(algorithm: str, model_id: str, g: np.ndarray) -> np.ndarray:
    # carrega o modelo correspondente ao model_id e verifica se o tamanho do sinal g é compatível com o modelo
    m = MODELS[model_id]
    if g.size != m["S"]:
        print(f"error: tamanho do sinal g={g.size} diferente do esperado {m['S']} para o modelo {model_id}")
        return {"error": f"Tamanho do sinal g={g.size} diferente do esperado {m['S']} para o modelo {model_id}"}
    if algorithm == "CGNR":
        f, iters, err = cgnr_function(m["H"], m["Ht"], g)
    elif algorithm == "CGNE":
        f, iters, err = cgne_function(m["H"], m["Ht"], g)
    else:
        return {"error": f"algoritmo '{algorithm}' não suportado"}
    print(f"reconstrução {model_id} completa: iters={iters}, erro final={err:.6f}")
    # reshape a imagem para o formato original usando ordem 'F' (coluna principal) para garantir a correspondência correta dos pixels
    img = f.reshape(m["shape"], order="F")

    lo, hi = float(img.min()), float(img.max())
    span = hi - lo
    # normaliza a imagem para o intervalo [0, 1], se span for zero, retorna uma imagem de zeros
    return (img - lo) / span if span > 0 else np.zeros_like(img),iters

#classe p receber o sinal g no formato JSON, onde g é uma lista de floats
class Sinal(BaseModel):
   g: list[float]

#endpoint p receber o sinal g e retornar a imagem reconstruida, verificando se o model_id é válido 
# e se o tamanho do sinal g é compatível com o modelo, caso contrário retorna um erro
@app.post("/reconstruct/{model_id}")
async def reconstruct(cliente_id: str, algorithm: str, model_id: str, sinal: Sinal, complete: bool = False):
    print(f"recebendo sinal g para {model_id} do cliente {cliente_id}, parte com {len(sinal.g)} elementos, complete={complete}")
    # acumular partes de sinais g enviados em partes, e só executar a reconstrução quando o cliente indicar que o envio está completo
    if CLIENT_SIGNALS.get(cliente_id) is None:
        CLIENT_SIGNALS[cliente_id] = {"algorithm": algorithm, "model_id": model_id, "g_parts": [], "complete": False}
    CLIENT_SIGNALS[cliente_id]["g_parts"].extend(sinal.g)
    CLIENT_SIGNALS[cliente_id]["complete"] = complete
    if not complete:
        return {"message": f"sinal g recebido para {model_id}, aguardando mais partes..."}

    # verifica se o model_id é válido, caso contrário retorna um erro com a lista de modelos disponíveis

    if model_id not in MODELS:
        return {"error": f"modelo '{model_id}' não encontrado. segue os modelos disponiveis: {list(MODELS.keys())}"}
    g = np.asarray(CLIENT_SIGNALS[cliente_id]["g_parts"], dtype=np.float32)
    try:
        # executa a funcao de reconstrucao em uma thread separada p evitar
        # bloquuear o loop de eventos
        result = await asyncio.to_thread(reconstruct_image, CLIENT_SIGNALS[cliente_id]["algorithm"], model_id, g)
        if isinstance(result, dict) and "error" in result:
            return {"error": f"erro na reconstrução da imagem: {result['error']}"}
        img, iters = result
    except ValueError as e:
        return {"error": f"erro na reconstrução da imagem: {str(e)}"}
    # retorna a imagem reconstruida como uma lista de floats,
    # junto com uma mensagem indicando que a reconstrução foi completa para o modelo especificado
    CLIENT_SIGNALS[cliente_id]["g_parts"] = []
    CLIENT_SIGNALS[cliente_id]["complete"] = False
    return {"message": f"reconstrucao completa para {model_id}", "image": img.tolist(), "iters": iters}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=2)