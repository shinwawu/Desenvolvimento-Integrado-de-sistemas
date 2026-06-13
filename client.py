import requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend headless, seguro para uso em threads
import matplotlib.pyplot as plt
from threading import Thread, Lock
import asyncio
import os
import time
import random

HOST = "127.0.0.1"
PORT = 8000
NUM_CLIENTS = 30  # reduzido pra teste de 3 instancias paralelas

# identificador unico desta instancia do client.py. permite rodar varias instancias
# em paralelo sem colidir cliente_id no servidor nem sobrescrever os arquivos de saida.
INSTANCE_ID = f"i{os.getpid()}"

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


def aplicar_ganho_sinal(g: np.ndarray) -> np.ndarray:
    """Calculo do ganho de sinal conforme especificacao:

        gamma_l = 100 + (1/20) * l * sqrt(l)
        g[l]    = g[l] * gamma_l

    onde l e o indice 1-based da amostra. Amplifica as amostras finais
    compensando a atenuacao do sinal recebido (brilho do sinal).
    """
    l = np.arange(1, len(g) + 1, dtype=np.float64)
    gamma = 100.0 + (1.0 / 20.0) * l * np.sqrt(l)
    return g * gamma


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
    tempo_inicio: str,
    tempo_final: str,
    iters: int,
):

    h, w = img.shape
    fig, ax = plt.subplots(figsize=(5.5, 6.3))
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    caption = (
        f"Algoritmo : {algorithm}\n"
        f"Inicio    : {tempo_inicio}\n"
        f"Termino   : {tempo_final}\n"
        f"Tamanho   : {w} x {h} px\n"
        f"Iteracoes : {iters}"
    )
    fig.text(
        0.5, 0.02, caption, ha="center", va="bottom", family="monospace", fontsize=9
    )
    fig.subplots_adjust(bottom=0.22, top=0.97)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# cada cliente sorteia 1 imagem, 1 algoritmo, e decide aleatoriamente se aplica
# a correcao de ganho (brilho) antes de enviar
async def inicializar_cliente(client_id: int):
    img_random = random.randint(1, 6)
    algo_random = random.choice(algorithms)
    aplicar_ganho = random.choice([True, False])
    value = imagem_modelo[img_random]
    print(
        f"[client {client_id}] img={img_random} algo={algo_random} "
        f"ganho={'sim' if aplicar_ganho else 'nao'} model={value['model_id']}"
    )

    g = sinais[img_random]
    if aplicar_ganho:
        g = aplicar_ganho_sinal(g)
    n_parts = int(np.random.randint(1, 10))
    partes = np.array_split(g, n_parts)

    cliente_id = f"{INSTANCE_ID}-{client_id}"
    response = await enviar_sequencia(
        cliente_id, algo_random, value["model_id"], partes
    )

    if not isinstance(response, dict):
        print(f"[client {client_id}] sem resposta")
        return
    if "error" in response:
        print(f"[client {client_id}] falha no servidor: {response['error']}")
        return
    img_data = response.get("image")
    if img_data is None:
        print(f"[client {client_id}] resposta sem imagem")
        return

    iters = response.get("iters")
    tempo_reconstrucao = response.get("tempo_reconstrucao") or response.get("reconstruction_time")
    erro_final = response.get("erro_final") or response.get("final_error")
    tempo_inicio = response.get("tempo_inicio") or response.get("start_time")
    tempo_final = response.get("tempo_fim") or response.get("end_time")
    converg = erro_final is not None and erro_final < 1e-4

    img_array = np.array(img_data)
    png_path = f"reconstructed_{INSTANCE_ID}_client{client_id}_{algo_random}_img{img_random}.png"
    salvar_imagem(
        png_path,
        img_array,
        algorithm=algo_random,
        tempo_inicio=tempo_inicio,
        tempo_final=tempo_final,
        iters=iters,
    )
    status = "OK" if converg else "NAO-CONVERGIU"
    print(
        f"[client {client_id}] {png_path} iters={iters} eps={erro_final:.3e} "
        f"t={tempo_reconstrucao:.3f}s {status}"
    )

    with relatorio_lock:
        relatorio_rows.append(
            {
                "client_id": client_id,
                "algorithm": algo_random,
                "model_id": value["model_id"],
                "image_number": img_random,
                "ganho_aplicado": aplicar_ganho,
                "image_file": png_path,
                "iters": iters,
                "erro_final": erro_final,
                "converg": converg,
                "tempo_inicio": tempo_inicio,
                "tempo_final": tempo_final,
                "reconstruction_time": tempo_reconstrucao,
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
    csv_path = f"relatorio_{INSTANCE_ID}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll {NUM_CLIENTS} clients finished in {time.time() - start:.2f}s")
    print(f"Reconstructions: {len(df)} -> {csv_path}")
