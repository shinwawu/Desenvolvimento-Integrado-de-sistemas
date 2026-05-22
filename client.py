import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asyncio import run
from threading import Thread
import asyncio
import time

HOST = "127.0.0.1"
PORT = 8000
NUM_CLIENTS = 100

teste = {
    "g-1-30x30": {"path": "g-30x30-1.csv", "model_id": "30x30"},
    "g-2-30x30": {"path": "g-30x30-2.csv", "model_id": "30x30"},
}

# pre-load signals once so all threads share the same data
sinais = {
    key: pd.read_csv(value["path"], header=None).to_numpy(dtype=np.float64).ravel()
    for key, value in teste.items()
}


async def get_imagem(client_id: int, model_id: str, sinal: np.ndarray):
    url = f"http://{HOST}:{PORT}/reconstruct/{model_id}"
    response = requests.post(url, json={"g": sinal.tolist()})
    if response.status_code == 200:
        print(f"[client {client_id}] ok: model_id={model_id}")
        return response.json()
    else:
        raise Exception(
            f"[client {client_id}] failed: model_id={model_id}, status={response.status_code}"
        )


async def inicializar_cliente(client_id: int):
    for key, value in teste.items():
        print(f"[client {client_id}] sending {key}...")
        try:
            response = await get_imagem(client_id, value["model_id"], sinais[key])
        except Exception as e:
            print(e)
            continue

        if "error" in response:
            print(f"[client {client_id}] server error for {key}: {response['error']}")
            continue

        img_data = response.get("image")
        if img_data is not None:
            img_array = np.array(img_data)
            plt.imsave(f"reconstructed_client{client_id}_{key}.png", img_array, cmap="gray")
            print(f"[client {client_id}] saved reconstructed_client{client_id}_{key}.png")
        else:
            print(f"[client {client_id}] no image in response for {key}")


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

