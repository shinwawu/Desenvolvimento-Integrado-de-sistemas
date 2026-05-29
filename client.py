import requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend headless, seguro para uso em threads
import matplotlib.pyplot as plt
from threading import Thread, Lock
import asyncio
import time
import random

HOST = "127.0.0.1"
PORT = 8000
NUM_CLIENTS = 500

# serve para proteger o acesso concorrente a relatorio_rows, onde cada thread de cliente registra seus resultados
relatorio_lock = Lock()
relatorio_rows: list[dict] = []

imagem_modelo = {
    1: {"path": "g-30x30-1.csv", "model_id": "30x30"},
    2: {"path": "g-30x30-2.csv", "model_id": "30x30"},
    3: {"path": "g-30x30-3.csv", "model_id": "30x30"},
    4: {"path": "g-60x60-1.csv", "model_id": "60x60"},
    5: {"path": "g-60x60-2.csv", "model_id": "60x60"},
    6: {"path": "g-60x60-3.csv", "model_id": "60x60"},
}
algorithms = ["CGNR", "CGNE"]


sinais = {
    k: pd.read_csv(v["path"], header=None).to_numpy(dtype=np.float64).ravel()
    for k, v in imagem_modelo.items()
}


# cada cliente envia a mesma sequencia de g para ambos os algoritmos
async def enviar_sequencia(
    cliente_id: str, algorithm: str, model_id: str, partes: list[np.ndarray]
):

    url = f"http://{HOST}:{PORT}/reconstruct/{model_id}"
    for i, parte in enumerate(partes):
        await asyncio.sleep(random.uniform(0.1, 0.5))
        payload = {"g": parte.tolist()}
        params = {
            "cliente_id": cliente_id,
            "algorithm": algorithm,
            "model_id": model_id,
            "complete": i == len(partes) - 1,
        }
        try:
            # realiza o envio da parte atual para o servidor, aguardando a resposta de forma assíncrona
            response = await asyncio.to_thread(
                requests.post, url, params=params, json=payload
            )
            # verifica se a resposta do servidor indica sucesso, caso contrário, lança uma exceção
            response.raise_for_status()
            if params["complete"]:
                return response.json()
        except requests.RequestException as e:
            print(f"[{cliente_id}] request error: {e}")
            return {"error": str(e)}


# salva a imagem reconstruida
def salvar_imagem(
    path: str,
    img: np.ndarray,
    *,
    algorithm: str,
    start_time: str,
    end_time: str,
    iters: int,
):

    h, w = img.shape
    fig, ax = plt.subplots(figsize=(5.5, 6.3))
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    caption = (
        f"Algoritmo : {algorithm}\n"
        f"Inicio    : {start_time}\n"
        f"Termino   : {end_time}\n"
        f"Tamanho   : {w} x {h} px\n"
        f"Iteracoes : {iters}"
    )
    fig.text(
        0.5, 0.02, caption, ha="center", va="bottom", family="monospace", fontsize=9
    )
    fig.subplots_adjust(bottom=0.22, top=0.97)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# inicializa o cliente, seleciona imagem e ganho aleatorios, divide o sinal em partes, envia para ambos os algoritmos e registra os resultados
async def inicializar_cliente(client_id: int):
    img_random = random.randint(1, 6)
    gain = round(random.uniform(0.5, 1.5), 4)
    value = imagem_modelo[img_random]
    print(
        f"[client {client_id}] img={img_random} gain={gain} model={value['model_id']}"
    )

    # ganho aplicado antes do split, e o split e o MESMO para os dois algoritmos
    sinal = sinais[img_random] * gain
    n_parts = int(np.random.randint(1, 10))
    partes = np.array_split(sinal, n_parts)

    # mesma sequencia de g enviada em paralelo para CGNR e CGNE
    tasks = [
        enviar_sequencia(f"{client_id}-{algo}", algo, value["model_id"], partes)
        for algo in algorithms
    ]
    results = await asyncio.gather(*tasks)

    for algo, response in zip(algorithms, results):
        if not isinstance(response, dict):
            print(f"[client {client_id}-{algo}] sem resposta")
            continue
        if "error" in response:
            print(f"[client {client_id}-{algo}] falha no servidor: {response['error']}")
            continue
        img_data = response.get("image")
        if img_data is None:
            print(f"[client {client_id}-{algo}] resposta sem imagem")
            continue
        iters = response.get("iters")
        recon_time = response.get("reconstruction_time")
        final_error = response.get("final_error")
        start_time = response.get("start_time")
        end_time = response.get("end_time")
        converged = final_error is not None and final_error < 1e-4

        img_array = np.array(img_data)
        png_path = f"reconstructed_client{client_id}_{algo}_img{img_random}.png"
        salvar_imagem(
            png_path,
            img_array,
            algorithm=algo,
            start_time=start_time,
            end_time=end_time,
            iters=iters,
        )
        # convergencia significa que o erro final ficou abaixo de 1e-4
        # e ok significa que o processo de reconstrução foi concluído sem erros, mesmo que não tenha convergido
        status = "OK" if converged else "NAO-CONVERGIU"
        print(
            f"[client {client_id}-{algo}] {png_path} iters={iters} eps={final_error:.3e} t={recon_time:.3f}s {status}"
        )

        with relatorio_lock:
            relatorio_rows.append(
                {
                    "client_id": client_id,
                    "algorithm": algo,
                    "model_id": value["model_id"],
                    "image_number": img_random,
                    "signal_gain": gain,
                    "image_file": png_path,
                    "iters": iters,
                    "final_error": final_error,
                    "converged": converged,
                    "start_time": start_time,
                    "end_time": end_time,
                    "reconstruction_time": recon_time,
                }
            )


def run_cliente(client_id: int):
    asyncio.run(inicializar_cliente(client_id))


if __name__ == "__main__":
    start = time.time()
    threads = [Thread(target=run_cliente, args=(i + 1,)) for i in range(NUM_CLIENTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    df = pd.DataFrame(relatorio_rows)
    if not df.empty:
        df = df.sort_values(["client_id", "algorithm"])
    df.to_csv("relatorio_reconstrucoes.csv", index=False)
    print(f"\nAll {NUM_CLIENTS} clients finished in {time.time() - start:.2f}s")
    print(f"Reconstructions: {len(df)} -> relatorio_reconstrucoes.csv")
