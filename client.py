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
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "300"))

# identificador unico desta instancia do client.py. permite rodar varias instancias
# em paralelo sem colidir cliente_id no servidor nem sobrescrever os arquivos de saida.
INSTANCE_ID = f"i{os.getpid()}"

# serve para proteger o acesso concorrente a relatorio_rows, onde cada thread de cliente registra seus resultados
relatorio_lock = Lock()
relatorio_rows: list[dict] = []

# contadores de envios aceitos/rejeitados por modelo, alimentados pelas threads de cliente.
# uma tarefa e "aceita" quando o servidor devolve uma imagem; e "rejeitada" quando ocorre
# erro de request, rejeicao por memoria (HTTP 503) ou resposta sem imagem.
contadores = {
    "30x30": {"aceitas": 0, "rejeitadas": 0},
    "60x60": {"aceitas": 0, "rejeitadas": 0},
}


def registrar_envio(model_id: str, aceita: bool):
    """Incrementa, de forma thread-safe, o contador de envios do modelo dado."""
    chave = "aceitas" if aceita else "rejeitadas"
    with relatorio_lock:
        contadores[model_id][chave] += 1


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


# intervalo entre consultas de polling e tempo maximo de espera pelo resultado
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 120.0


# modelo assincrono: o cliente dispara o request (apos um intervalo aleatorio),
# recebe na hora um job_id e depois faz polling ate o resultado ficar pronto.
# a conexao do POST nao fica presa durante a reconstrucao.
async def enviar_sinal(cliente_id: str, algorithm: str, model_id: str, g: np.ndarray):

    url = f"http://{HOST}:{PORT}/reconstruct/{model_id}"
    # intervalo de tempo aleatorio antes de disparar o request
    await asyncio.sleep(random.uniform(0.1, 0.5))
    payload = {"g": g.tolist()}
    params = {
        "cliente_id": cliente_id,
        "algorithm": algorithm,
        "model_id": model_id,
        "complete": True,
    }
    try:
        # dispara o sinal; o servidor responde imediatamente com o job_id
        response = await asyncio.to_thread(
            requests.post, url, params=params, json=payload
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"[{cliente_id}] request error: {e}")
        return {"error": str(e)}

    # se o servidor recusou na validacao (modelo/tamanho), ja vem o erro aqui
    job_id = data.get("job_id")
    if not job_id:
        return data

    # consulta o resultado periodicamente ate concluir
    return await aguardar_resultado(cliente_id, job_id)


# faz polling em GET /result/{job_id} ate o job sair de "pending"
async def aguardar_resultado(cliente_id: str, job_id: str):
    url = f"http://{HOST}:{PORT}/result/{job_id}"
    deadline = time.time() + POLL_TIMEOUT
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            response = await asyncio.to_thread(requests.get, url)
        except requests.RequestException as e:
            print(f"[{cliente_id}] poll error: {e}")
            return {"error": str(e)}

        if response.status_code == 404:
            return {"error": f"job {job_id} nao encontrado"}
        try:
            data = response.json()
        except ValueError:
            return {"error": f"resposta invalida do servidor: {response.text[:120]}"}

        status = data.get("status")
        if status == "pending":
            if time.time() > deadline:
                return {"error": f"timeout aguardando job {job_id}"}
            continue
        # status "done" (com imagem) ou "error" -> entrega para o chamador
        return data


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

    cliente_id = f"{INSTANCE_ID}-{client_id}"
    response = await enviar_sinal(cliente_id, algo_random, value["model_id"], g)

    if not isinstance(response, dict):
        print(f"[client {client_id}] sem resposta")
        registrar_envio(value["model_id"], aceita=False)
        return
    if "error" in response:
        print(f"[client {client_id}] falha no servidor: {response['error']}")
        registrar_envio(value["model_id"], aceita=False)
        return
    img_data = response.get("image")
    if img_data is None:
        print(f"[client {client_id}] resposta sem imagem")
        registrar_envio(value["model_id"], aceita=False)
        return

    registrar_envio(value["model_id"], aceita=True)

    iters = response.get("iters")
    tempo_reconstrucao = response.get("tempo_reconstrucao") or response.get(
        "reconstruction_time"
    )
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


# escreve no arquivo f e no stdout o resumo de tarefas aceitas/rejeitadas por modelo
def gerar_relatorio_geral(f, contadores, pid_cliente, num_tarefas):
    f.write(f"\n\n--- Relatório de Envio (PID: {pid_cliente}) ---\n")
    print(f"\n--- Relatório de Envio (PID: {pid_cliente}) ---")

    f.write(f"Total de {num_tarefas} tarefas tentadas.\n")
    print(f"Total de {num_tarefas} tarefas tentadas.")

    linha_60 = f"  - Modelo 60x60: {contadores['60x60']['aceitas']} Aceitas, {contadores['60x60']['rejeitadas']} Rejeitadas.\n"
    linha_30 = f"  - Modelo 30x30: {contadores['30x30']['aceitas']} Aceitas, {contadores['30x30']['rejeitadas']} Rejeitadas.\n"

    f.write(linha_60)
    f.write(linha_30)
    print(linha_60, end="")
    print(linha_30, end="")

    f.write("\n-------------------------------------------\n")
    print("\n-------------------------------------------")


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

    relatorio_path = f"relatorio_clientes_{INSTANCE_ID}.txt"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        gerar_relatorio_geral(f, contadores, os.getpid(), NUM_CLIENTS)
    print(f"Relatorio de envio -> {relatorio_path}")
