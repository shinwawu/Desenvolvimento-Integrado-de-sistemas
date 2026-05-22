import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asyncio import run
from threading import Thread
import asyncio
import time
import random
# requisitos funcionais:
# TODO :
# - o cliente deve enviar sinais g para o servdiro em intervalos aleatorios entre 01 e 0.5 segundos. (Feito)
# - o ganho de sinal e o modelo da imagem deverão ser definidos aleatoriamente. (HALF MADE)
# - gerar um relatório com todas as imagens reconstruídas com as seguintes informações: imagem gerada, número de iterações e tempo de reconstrução. (FEITO)
# - a sequencia de sinais (g) enviados deverão ser os mesmos para as duas versões de algoritmos de reconstrução (HALF MADE, CGNR feito, falta o CGNE)
HOST = "127.0.0.1"
PORT = 8000
NUM_CLIENTS = 100
relatorio = {
    "client_id": [],
    "algorithm": [],
    "model_id": [],
    "image_number": [],
    "iters": [],
    "reconstruction_time": []
}
imagem_modelo = {
    1: {"path": "g-30x30-1.csv", "model_id": "30x30"},
    2: {"path": "g-30x30-2.csv", "model_id": "30x30"},
    3: {"path": "g-30x30-3.csv", "model_id": "30x30"},
}
algorithms = ["CGNR", "CGNE"]  # lista de algoritmos de reconstrução disponíveis

# each client will randomly select one of the three signal files and one of the two algorithms to send to the server. The signals will be sent in random intervals, and the client will save the reconstructed images and generate a report with the details of each reconstruction.
sinais = {
    1: pd.read_csv(imagem_modelo[1]["path"], header=None).to_numpy(dtype=np.float64).ravel(),
    2: pd.read_csv(imagem_modelo[2]["path"], header=None).to_numpy(dtype=np.float64).ravel(),
    3: pd.read_csv(imagem_modelo[3]["path"], header=None).to_numpy(dtype=np.float64).ravel(),
}

async def get_imagem(client_id: int, algorithm: str, model_id: str, sinal: np.ndarray):
    url = f"http://{HOST}:{PORT}/reconstruct/{model_id}"
    partes_sequencia = np.array_split(sinal, np.random.randint(1, 10))
    for i, parte in enumerate(partes_sequencia):
        await asyncio.sleep(random.uniform(0.1, 0.5))
        payload = {"g": parte.tolist()}
        params = {
            "cliente_id": client_id,
            "algorithm": algorithm,
            "model_id": model_id,
            "complete": i == len(partes_sequencia) - 1
        }
        try:
            response = await asyncio.to_thread(requests.post, url, params=params, json=payload)
            response.raise_for_status()
            if params["complete"]:
                return response.json()
        except requests.RequestException as e:
            print(f"[client {client_id}] request error: {e}")
            return {"error": f"erro na reconstrução da imagem: {str(e)}"}


async def inicializar_cliente(client_id: int):
    img_random = random.randint(1, 3)
    algo_random = random.choice(algorithms)
    value = imagem_modelo[img_random]
    print(f"[client {client_id}] sending sinais da imagem {img_random}...")
    try:
        response = await get_imagem(client_id, algo_random, value["model_id"], sinais[str(img_random)])
    except Exception as e:
        print(f"[client {client_id}] unexpected error: {e}")
        return

    if "error" in response:
        print(f"[client {client_id}] server error for img{img_random}: {response['error']}")
        return

    img_data, iters = response.get("image"), response.get("iters")
    if img_data is not None:
        img_array = np.array(img_data)
        plt.imsave(f"reconstructed_client{client_id}_img{img_random}.png", img_array, cmap="gray")
        print(f"[client {client_id}] saved reconstructed_client{client_id}_img{img_random}.png and iters={iters}")
        relatorio["client_id"].append(client_id)
        relatorio["algorithm"].append(algo_random)
        relatorio["model_id"].append(value["model_id"])
        relatorio["image_number"].append(img_random)
        relatorio["iters"].append(iters)
        relatorio["reconstruction_time"].append(response.get("reconstruction_time"))
        relatorio_df = pd.DataFrame(relatorio)
        relatorio_df.to_csv(f"relatorio_{client_id}_reconstrucoes.csv", index=False)
    else:
        print(f"[client {client_id}] no image in response for img{img_random}")

def run_cliente(client_id: int):
    run(inicializar_cliente(client_id))


if __name__ == "__main__":
    start = time.time()
    threads = [Thread(target=run_cliente, args=(i + 1,)) for i in range(NUM_CLIENTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\nAll {NUM_CLIENTS} clients finished in {time.time() - start:.2f}s")

