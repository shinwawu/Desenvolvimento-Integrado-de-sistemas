import requests
from pydantic import BaseModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asyncio import run
from threading import Thread
HOST = "127.0.0.1"
PORT = 8000

teste = {
    "g-1-30x30": {"path": "g-30x30-1.csv", "model_id": "30x30"},
    "g-2-30x30": {"path": "g-30x30-2.csv", "model_id": "30x30"},
}


async def get_imagem(model_id: str, sinal: list[float]):
    url = f"http://{HOST}:{PORT}/reconstruct/{model_id}"
    response = requests.post(url, json={"g": sinal.tolist()})
    if response.status_code == 200:
        print(f"Successfully got image for model_id {model_id}.")
        return response.json()
    else:
        raise Exception(
            f"Failed to get image for model_id {model_id}. Status code: {response.status_code}"
        )


async def inicializar_cliente():
    for key, value in teste.items():
        print(f"Processing {key} with model_id {value['model_id']}...")
        g = pd.read_csv(value["path"], header=None).to_numpy(dtype=np.float64).ravel()

        response = await get_imagem(value["model_id"], g)

        if "error" in response:
            print(f"Error for {key}: {response['error']}")
            continue

        img_data = response.get("image")
        if img_data is not None:

            img_array = np.array(img_data)
            plt.imsave(f"reconstructed_{key}.png", img_array, cmap="gray")
            print(f"Image for {key} saved successfully.")
        else:
            print(f"No image found in response for {key}.")


if __name__ == "__main__":
    thread = Thread(target=lambda: run(inicializar_cliente()))
    thread.start()
    thread.join()

