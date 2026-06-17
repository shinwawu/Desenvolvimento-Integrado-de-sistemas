import asyncio
import itertools
import os
import time
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import psutil
import requests
import scipy.sparse as sp
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, ORJSONResponse
from pydantic import BaseModel

# TODO:
# task 1: arrumar o ganho de sinal que tem o significado errado, ela tem a ver com o brilho do sinal
# task 2: normalizar para o absoluto antes de converter para escala de cinza, para evitar que o brilho do sinal afete a escala de cinza da imagem reconstruida


# Configurações dos modelos disponíveis
MODELS_CONFIG = {
    "60x60": {"S": 50816, "N": 3600, "shape": (60, 60), "path": "data/H-1.npz"},
    "30x30": {"S": 27904, "N": 900, "shape": (30, 30), "path": "data/H-2.npz"},
}
# listar os modelos disponíveis
MODELS: dict = {}

# porta deste worker — usada como prefixo do job_id para o proxy saber rotear o
# GET /result de volta ao worker dono do job (mesma ideia do servidor Rust).
# Definida via env var pelo proxy ao spawnar cada worker; default 8000 (modo solo).
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8000"))

# controle de concorrencia e admissao
MAX_REQUEST = max(2, (os.cpu_count() or 4) * 2)

# controle de memoria para que o servidor nao fique sobrecarregado
MEMO_MINIMA = 0.5  # piso abaixo do qual o request espera (nao rejeita)
TEMPO_DE_ESPERA = 300  # so rejeita em ultimo caso, depois de 5 min esperando
# frequencia de re-checagem da memoria durante a espera (evita busy-loop)
TEMPO_VERIFICACAO = 0.5

# tempo maximo p reconstrucao d img
TEMPO_CONSTRUCAO = 30.0

request_max = asyncio.Semaphore(MAX_REQUEST)

# jobs assincronos: o POST cria um job e devolve o job_id na hora; o processamento
# roda em background (asyncio.create_task) e o cliente consulta GET /result/{job_id}.
# O store fica na memoria do processo, entao o modelo job_id+polling requer WORKERS=1.
# Em uvicorn multi-worker nao ha sticky routing: o polling pode cair num worker que
# nao tem o job (404). Para multi-worker seria preciso um store compartilhado.
# Acesso ao dict e seguro sem lock: tudo roda no mesmo event loop (asyncio e single
# thread) e a reconstrucao pesada vai para uma thread que nao toca em `jobs`.
jobs: dict[str, dict] = {}
_job_seq = itertools.count()
metricas = {
    "no_processo": 0,
    "na_fila": 0,
    "completos": 0,
    "rejeitados": 0,
    "esperando_memo": 0,
    "timeout": 0,
    "falha": 0,
    "times_ms": deque(maxlen=500),
}


# carregar o modelo quand
def load_model(model_id: str, cfg: dict) -> dict:
    path = Path(cfg["path"])
    if not path.exists():
        print(f"error: {path} matriz nao encontrado do modelo {model_id}")
        return None

    H = sp.load_npz(path)
    if H.shape != (cfg["S"], cfg["N"]):
        raise ValueError(
            f"{path} diferenca no tamanho {H.shape},esperado {(cfg['S'], cfg['N'])}"
        )

    # converter para float32 e csr para melhor desempenho
    H = H.astype(np.float32).tocsr()
    Ht = H.T.tocsr()

    print(
        f"modelo carregado {model_id}: matriz h={H.shape}, matriz h transposta={Ht.shape}"
    )
    return {"H": H, "Ht": Ht, "shape": cfg["shape"], "S": cfg["S"], "N": cfg["N"]}


# monitor de memoria, q verifica a memoria do sistema e do processo e printa
async def memoria_monitor(intervalo_s: float = 2.0):
    GB = 1024**3
    processo = psutil.Process()
    while True:
        try:
            vm = psutil.virtual_memory()
            uso_da_app = processo.memory_info().rss
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[memoria {ts}] sistema usada={vm.used/GB:.2f}GB "
                f"disponivel={vm.available/GB:.2f}GB / total={vm.total/GB:.2f}GB "
                f"({vm.percent:.1f}%) | uso da app ={uso_da_app/GB:.2f}GB",
                flush=True,
            )
        except Exception as e:
            print(f"[memoria] erro: {e}", flush=True)
        await asyncio.sleep(intervalo_s)


# gerencia o ciclo de vida do app, e carrega os modelos ao iniciar e limpa ao finalizar
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    inicia o server, carrega os modelos e o monitor de memoria.
    """
    for mid, cfg in MODELS_CONFIG.items():
        model = load_model(mid, cfg)
        if model is not None:
            MODELS[mid] = model
    # sob o proxy, quem monitora memoria (1 linha agregada/s) e o proxy; os workers
    # nao monitoram para nao poluir o terminal com N linhas por segundo.
    monitor_task = None
    if not os.environ.get("DISABLE_MONITOR"):
        monitor_task = asyncio.create_task(memoria_monitor())
    yield
    if monitor_task is not None:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    MODELS.clear()


# cria a instancia do fastapi
# definindo o ciclo de vida do app para carregar os modelos ao iniciar e limpar ao finalizar
# optamos por usar ORJSONResponse para melhorar a performance na serialização de respostas JSON
app = FastAPI(lifespan=lifespan)


# função de reconstrucao usando o metodo CGNR
def cgnr_function(
    matriz_h: sp.csr_matrix,
    matriz_h_t: sp.csr_matrix,
    g: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
):

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


def cgne_function(
    matriz_h: sp.csr_matrix,
    matriz_h_t: sp.csr_matrix,
    g: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
):
    f = np.zeros(matriz_h.shape[1], dtype=np.float32)
    r = g.copy() - matriz_h @ f
    p = matriz_h_t @ r
    rtr = float(r @ r)
    for k in range(max_iter):
        Hp = matriz_h @ p
        ptp = float(p @ p)
        if ptp == 0.0:
            return f, k, float(np.linalg.norm(r))
        a = rtr / ptp
        f += a * p
        r -= a * Hp
        err = float(np.linalg.norm(r))
        if err < tol:
            return f, k + 1, err
        rtr_new = float(r @ r)
        beta = rtr_new / rtr
        p = matriz_h_t @ r + beta * p
        rtr = rtr_new
    return f, max_iter, err


def reconstruct_image(algorithm: str, model_id: str, g: np.ndarray) -> np.ndarray:
    # carrega o modelo correspondente ao model_id e verifica se o tamanho do sinal g é compatível com o modelo
    m = MODELS[model_id]
    if g.size != m["S"]:
        print(
            f"error: tamanho do sinal g={g.size} diferente do esperado {m['S']} para o modelo {model_id}"
        )
        return {
            "error": f"Tamanho do sinal g={g.size} diferente do esperado {m['S']} para o modelo {model_id}"
        }
    if algorithm == "CGNR":
        f, iters, err = cgnr_function(m["H"], m["Ht"], g)
    elif algorithm == "CGNE":
        f, iters, err = cgne_function(m["H"], m["Ht"], g)
    else:
        return {"error": f"algoritmo '{algorithm}' não suportado"}
    print(f"reconstrução {model_id} completa: iters={iters}, erro final={err:.6f}")
    # reshape a imagem para o formato original usando ordem 'F' (coluna principal) para garantir a correspondência correta dos pixels
    f = abs(
        f
    )  # normaliza para o absoluto antes de converter para escala de cinza, para evitar que o brilho do sinal afete a escala de cinza da imagem reconstruida
    img = f.reshape(m["shape"], order="F")

    lo, hi = float(img.min()), float(img.max())
    span = hi - lo
    # normaliza a imagem para o intervalo [0, 1], se span for zero, retorna uma imagem de zeros
    norm = (img - lo) / span if span > 0 else np.zeros_like(img)
    return norm, iters, err


# classe p receber o sinal g no formato JSON, onde g é uma lista de floats
class Sinal(BaseModel):
    g: list[float]


# Modelo assincrono: o POST valida o request, cria um job, dispara a reconstrucao
# em background (asyncio.create_task) e responde na hora com {job_id}. O cliente
# consulta o resultado depois em GET /result/{job_id}.
# `cliente_id` e `complete` ficam no signature por compatibilidade com o cliente
# (que ainda os envia) mas nao sao usados — nao ha mais estado por cliente.


@app.post("/reconstruct/{model_id}")
async def reconstruct(
    cliente_id: str, algorithm: str, model_id: str, sinal: Sinal, complete: bool = True
):
    """
    Recebe o sinal g completo, cria um job, dispara a reconstrucao em background
    e responde imediatamente com o job_id (a conexao do POST nao fica presa).
    """
    # validacoes rapidas: falham na hora (sincronas), sem criar job.
    if model_id not in MODELS:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": f"modelo '{model_id}' não encontrado. segue os modelos disponiveis: {list(MODELS.keys())}",
            },
        )

    esperado = MODELS[model_id]["S"]
    if len(sinal.g) != esperado:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": f"tamanho do sinal g={len(sinal.g)} diferente do esperado {esperado} para o modelo {model_id}",
            },
        )

    # array de g vem direto do request (sem acumulacao por cliente)
    g = np.asarray(sinal.g, dtype=np.float32)

    # cria o job (estado pending) e dispara o processamento em background.
    # prefixo = porta do worker, para o proxy rotear o /result de volta pra ca.
    job_id = f"{WORKER_PORT}-{next(_job_seq)}"
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(processar_reconstrucao(job_id, algorithm, model_id, g))

    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


# Executa a reconstrucao de um job em background e grava o resultado final em
# jobs[job_id]. Aplica a mesma politica de admissao (espera memoria, semaforo,
# timeout) que a versao sincrona usava.
async def processar_reconstrucao(
    job_id: str, algorithm: str, model_id: str, g: np.ndarray
):
    GB = 1024**3
    # admission control por pressao de memoria: nao rejeita por padrao, espera ate a
    # memoria aliviar. So rejeita em ultimo caso, depois de TEMPO_DE_ESPERA.
    if psutil.virtual_memory().available < MEMO_MINIMA * GB:
        metricas["esperando_memo"] += 1
        espera_deadline = time.monotonic() + TEMPO_DE_ESPERA
        while psutil.virtual_memory().available < MEMO_MINIMA * GB:
            if time.monotonic() > espera_deadline:
                metricas["rejeitados"] += 1
                jobs[job_id] = {
                    "status": "error",
                    "error": (
                        f"memoria abaixo de {MEMO_MINIMA}GB por mais de "
                        f"{TEMPO_DE_ESPERA}s; rejeitando para evitar deadlock"
                    ),
                    "_http": 503,
                }
                return
            await asyncio.sleep(TEMPO_VERIFICACAO)

    metricas["na_fila"] += 1
    try:
        async with request_max:
            metricas["na_fila"] -= 1
            metricas["no_processo"] += 1
            try:
                start_dt = datetime.now()
                t0 = time.perf_counter()
                result = await asyncio.wait_for(
                    asyncio.to_thread(reconstruct_image, algorithm, model_id, g),
                    timeout=TEMPO_CONSTRUCAO,
                )
                elapsed = time.perf_counter() - t0
                end_dt = datetime.now()
            except asyncio.TimeoutError:
                metricas["timeout"] += 1
                jobs[job_id] = {
                    "status": "error",
                    "error": f"timeout > {TEMPO_CONSTRUCAO}s na reconstrução",
                    "_http": 504,
                }
                return
            except Exception as e:
                metricas["falha"] += 1
                jobs[job_id] = {
                    "status": "error",
                    "error": f"erro na reconstrução da imagem: {str(e)}",
                    "_http": 500,
                }
                return
            finally:
                metricas["no_processo"] -= 1
    except Exception:
        if metricas["na_fila"] > 0:
            metricas["na_fila"] -= 1
        metricas["falha"] += 1
        jobs[job_id] = {
            "status": "error",
            "error": "falha inesperada no processamento do job",
            "_http": 500,
        }
        return

    if isinstance(result, dict) and "error" in result:
        metricas["falha"] += 1
        jobs[job_id] = {
            "status": "error",
            "error": f"erro na reconstrução da imagem: {result['error']}",
            "_http": 500,
        }
        return

    img, iters, err = result
    metricas["completos"] += 1
    metricas["times_ms"].append(elapsed * 1000)

    jobs[job_id] = {
        "status": "done",
        "message": f"reconstrucao completa para {model_id}",
        "image": img.tolist(),
        "iters": iters,
        "erro_final": err,
        "tempo_reconstrucao": elapsed,
        "tempo_inicio": start_dt.isoformat(timespec="milliseconds"),
        "tempo_fim": end_dt.isoformat(timespec="milliseconds"),
        "_http": 200,
    }


# GET /result/{job_id}: devolve o estado do job. pending => {"status":"pending"};
# pronto => o corpo final (com a imagem ou o erro) e o job e removido do store;
# inexistente => 404. So um fetch terminal consome o job.
@app.get("/result/{job_id}")
async def result(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "error": f"job {job_id} nao encontrado"},
        )
    if job["status"] == "pending":
        return {"status": "pending"}
    # terminal: remove e devolve o corpo com o status HTTP apropriado.
    jobs.pop(job_id, None)
    body = dict(job)
    http = body.pop("_http", 200)
    return JSONResponse(status_code=http, content=body)


# funcao que calcula o percentil p de uma lista de valores ordenados, retornando None se a lista estiver vazia
def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    n = len(sorted_values)
    k = max(0, min(n - 1, int(round(p * (n - 1)))))
    return sorted_values[k]


def metrics():
    times = sorted(metricas["times_ms"])
    vm = psutil.virtual_memory()
    GB = 1024**3
    return {
        "limits": {
            "MAX_REQUEST": MAX_REQUEST,
            "queue_policy": (
                "unbounded; wait until memory recovers (no reject by count); "
                "last-resort reject only after TEMPO_DE_ESPERA"
            ),
            "MEMO_MINIMA": MEMO_MINIMA,
            "TEMPO_DE_ESPERA": TEMPO_DE_ESPERA,
            "TEMPO_CONSTRUCAO": TEMPO_CONSTRUCAO,
        },
        "counters": {
            "completos": metricas["completos"],
            "rejeitados": metricas["rejeitados"],
            "esperando_memo": metricas["esperando_memo"],
            "timeout": metricas["timeout"],
            "falha": metricas["falha"],
        },
        "gauges": {
            "no_processo": metricas["no_processo"],
            "na_fila": metricas["na_fila"],
        },
        "latency_ms": {
            "sample_size": len(times),
            "p50": _percentile(times, 0.50),
            "p90": _percentile(times, 0.90),
            "p99": _percentile(times, 0.99),
        },
        "memory_gb": {
            "system_used": round(vm.used / GB, 2),
            "system_available": round(vm.available / GB, 2),
            "system_total": round(vm.total / GB, 2),
            "process_rss": round(psutil.Process().memory_info().rss / GB, 2),
        },
    }


# =============================================================================
#             proxy: load balancer com roteamento de job para o worker
# =============================================================================
#
# Mesma arquitetura do servidor Rust:
#
#   cliente HTTP ─────► proxy (porta 8000) ─────┬─► worker 1 (porta 8001)
#                            │                   ├─► worker 2 (porta 8002)
#                            └ memory monitor    └─► worker N (porta 800N)
#
# - POST /reconstruct e distribuido por round-robin entre os workers.
# - O job_id devolvido vem prefixado com a porta do worker dono (`porta-seq`),
#   entao o GET /result/{job_id} e roteado de volta para o mesmo worker — e o
#   equivalente do sticky routing, agora por job em vez de por cliente_id.
# - Cada worker tem seu proprio store de jobs em memoria; nao precisa replicar.

# portas dos workers, preenchidas no __main__ do proxy.
WORKER_PORTS: list[int] = []
_rr = itertools.count()  # contador round-robin do proxy


# monitor de memoria do proxy: 1 linha agregada/s somando o RSS do proxy + toda a
# sua arvore de processos (workers e os launchers do venv) — equivalente ao
# spawn_memory_monitor do Rust. Somamos a arvore (children recursive) porque o
# python.exe do venv no Windows e um launcher: o worker real e um neto do proxy,
# nao o pid direto devolvido pelo subprocess.Popen.
async def proxy_memoria_monitor(root_pid: int, intervalo_s: float = 1.0):
    GB = 1024**3
    while True:
        try:
            vm = psutil.virtual_memory()
            rss = 0
            try:
                raiz = psutil.Process(root_pid)
                for pr in [raiz, *raiz.children(recursive=True)]:
                    try:
                        rss += pr.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[memoria {ts}] sistema usada={vm.used/GB:.2f}GB "
                f"disponivel={vm.available/GB:.2f}GB / total={vm.total/GB:.2f}GB "
                f"({vm.percent:.1f}%) | uso da app ={rss/GB:.2f}GB",
                flush=True,
            )
        except Exception as e:
            print(f"[memoria] erro: {e}", flush=True)
        await asyncio.sleep(intervalo_s)


@asynccontextmanager
async def proxy_lifespan(app: FastAPI):
    monitor_task = asyncio.create_task(proxy_memoria_monitor(os.getpid()))
    yield
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass


proxy_app = FastAPI(lifespan=proxy_lifespan)


@proxy_app.post("/reconstruct/{model_id}")
async def proxy_reconstruct(model_id: str, request: Request):
    # round-robin: distribuicao uniforme entre os workers.
    port = WORKER_PORTS[next(_rr) % len(WORKER_PORTS)]
    body = await request.body()
    params = dict(request.query_params)
    url = f"http://127.0.0.1:{port}/reconstruct/{model_id}"

    def _forward():
        return requests.post(
            url, params=params, data=body,
            headers={"content-type": "application/json"}, timeout=300,
        )

    try:
        r = await asyncio.to_thread(_forward)
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": f"proxy -> worker:{port} falhou: {e}"},
        )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@proxy_app.get("/result/{job_id}")
async def proxy_result(job_id: str):
    # o job_id e prefixado com a porta do worker dono (`porta-seq`), entao
    # roteamos o polling de volta para o mesmo worker que criou o job.
    prefixo = job_id.split("-", 1)[0]
    try:
        port = int(prefixo)
    except ValueError:
        port = None
    if port not in WORKER_PORTS:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": f"job_id invalido: {job_id}"},
        )
    url = f"http://127.0.0.1:{port}/result/{job_id}"
    try:
        r = await asyncio.to_thread(requests.get, url, timeout=30)
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": f"proxy -> worker:{port} falhou: {e}"},
        )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@proxy_app.get("/health")
async def proxy_health():
    return {"status": "ok", "mode": "proxy", "workers": WORKER_PORTS}


def _esperar_health(port: int, timeout_s: int = 60) -> bool:
    """Espera o worker responder em /health (carrega os modelos antes)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _parse_arg(args: list[str], flag: str, default: str) -> str:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return default


@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODELS.keys())}


if __name__ == "__main__":
    import sys
    import subprocess
    import uvicorn

    args = sys.argv[1:]
    is_worker = "--worker" in args
    port = int(_parse_arg(args, "--port", "8000"))

    # backlog=8192 / limit_concurrency=2000: ver justificativa no fim do arquivo.
    if is_worker:
        # modo worker: serve o `app` (reconstrucao) na sua porta. job_id usa WORKER_PORT.
        uvicorn.run(
            "server:app", host="0.0.0.0", port=port,
            backlog=8192, limit_concurrency=2000,
        )
    else:
        # modo proxy: spawna N workers como processos filhos e roteia para eles.
        n_workers = int(os.environ.get("WORKERS", str(max(2, (os.cpu_count() or 4) // 2))))
        worker_ports = [port + 1 + i for i in range(n_workers)]
        WORKER_PORTS[:] = worker_ports

        children: list[subprocess.Popen] = []
        print(f"[proxy] iniciando {n_workers} workers nas portas {worker_ports}")
        for p in worker_ports:
            env = os.environ.copy()
            env["WORKER_PORT"] = str(p)      # prefixo do job_id
            env["DISABLE_MONITOR"] = "1"     # so o proxy monitora memoria
            child = subprocess.Popen(
                [sys.executable, "server.py", "--worker", "--port", str(p)], env=env
            )
            children.append(child)

        # espera todos os workers responderem /health (carregam modelos antes)
        for p in worker_ports:
            if not _esperar_health(p, 60):
                print(f"[proxy] worker {p} nao subiu em 60s")
        print(f"[proxy] {n_workers} workers prontos em {worker_ports}")

        try:
            uvicorn.run(
                proxy_app, host="0.0.0.0", port=port,
                backlog=8192, limit_concurrency=2000,
            )
        finally:
            # cleanup: encerra os workers ao sair (normal ou Ctrl-C)
            print("\n[proxy] encerrando workers...")
            for c in children:
                c.terminate()
            for c in children:
                try:
                    c.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    c.kill()
