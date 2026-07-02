import asyncio
import csv
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import subprocess
import threading
import uvicorn
import socket
import concurrent.futures
from urllib3.util.retry import Retry
# Configurações dos modelos disponíveis
MODELS_CONFIG = {
    "60x60": {"S": 50816, "N": 3600, "shape": (60, 60), "path": "data/H-1.npz"},
    "30x30": {"S": 27904, "N": 900, "shape": (30, 30), "path": "data/H-2.npz"},
}
# listar os modelos disponíveis
MODELS: dict = {}

WORKER_PORT = int(os.environ.get("WORKER_PORT", "8000"))

# Concorrencia TOTAL alvo da maquina = nucleos * FATOR (um pouco de oversubscription
# esconde a latencia de polling/serializacao). Esse total e DIVIDIDO entre os
# workers — cada worker recebe sua fatia em MAX_REQUEST. Antes era cpu*2 POR worker,
# o que estourava (N_workers * cpu*2 reconstrucoes simultaneas); agora o total fica
# saudavel (~nucleos*1.5) independente de quantos workers existam.
FATOR_CONCORRENCIA = 1.5
_N_WORKERS = max(1, int(os.environ.get("N_WORKERS", "1")))  # setado pelo supervisor
MAX_REQUEST = max(1, round((os.cpu_count() or 4) * FATOR_CONCORRENCIA / _N_WORKERS))
MEMO_MINIMA = 0.5  # piso abaixo do qual o request espera
TEMPO_DE_ESPERA = 300  # so rejeita em ultimo caso, depois de 5 min esperando

# frequencia de re-checagem da memoria 
TEMPO_VERIFICACAO = 0.5
# tempo maximo p reconstrucao d img 
TEMPO_CONSTRUCAO = 120.0
GB = 1024**3
class LimiteDinamico:
    """Limite de reconstrucoes simultaneas que SOBE e DESCE em runtime conforme a
    carga (logica do SystemMonitor.swift, aplicada continuamente). Diferente do
    numero de workers — que e fixo apos o boot — este teto muda a cada poll do
    monitor_concorrencia. Usado como `async with request_max:`.

    `acquire` espera ate haver vaga (em_execucao < maximo); reconstrucoes ja em
    andamento nunca sao interrompidas quando o teto baixa — apenas nao entram
    novas ate o nº cair abaixo do novo teto."""

    def __init__(self, maximo: int):
        self.maximo = maximo
        self.em_execucao = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            await self._cond.wait_for(lambda: self.em_execucao < self.maximo)
            self.em_execucao += 1
        return self

    async def __aexit__(self, *exc):
        async with self._cond:
            self.em_execucao -= 1
            self._cond.notify(1)

    async def set_maximo(self, novo: int):
        async with self._cond:
            aumentou = novo > self.maximo
            self.maximo = novo
            if aumentou:
                # abriu vaga(s): acorda os que esperam para reavaliarem a condicao
                self._cond.notify_all()


request_max = LimiteDinamico(MAX_REQUEST)
# dicionario de jobs
jobs: dict[str, dict] = {}
# contador de job_id, para gerar ids unicos e ordenados por chegada
_job_seq = itertools.count()
#metrics
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

# carregar o modelo quando solicitado
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


# limite minimo de concorrencia (nunca trava o worker totalmente)
MIN_CONCORRENTE = 1
# o throttle de concorrencia reage a MEMORIA DISPONIVEL, nao a CPU%: a reconstrucao
# e CPU-bound (cada uma ~1 core, memoria desprezivel), entao CPU alta e DESEJAVEL
# (cores ocupados = trabalho). O que pode travar a maquina e swap -> reagimos ao
# sinal honesto, a RAM livre. Acima de RAM_FOLGA usa o teto cheio; abaixo de
# RAM_CRITICA cai ao minimo; entre os dois, linear.
RAM_FOLGA_GB = 1.5
RAM_CRITICA_GB = 0.5  # alinhado com MEMO_MINIMA (o guard de OOM de ultimo caso)


async def monitor_concorrencia(intervalo_s: float = 1.0):
    """Ajusta o limite de reconstrucoes simultaneas (request_max) em tempo real, a
    cada `intervalo_s`, conforme a MEMORIA DISPONIVEL. Normalmente fica no teto
    (cores ocupados, nada ocioso); so reduz quando a RAM livre aperta, evitando
    contribuir para swap/travamento. Roda em cada worker."""
    anterior = None
    while True:
        try:
            disp_gb = psutil.virtual_memory().available / GB
            if disp_gb >= RAM_FOLGA_GB:
                novo = MAX_REQUEST
            elif disp_gb <= RAM_CRITICA_GB:
                novo = MIN_CONCORRENTE
            else:
                ratio = (disp_gb - RAM_CRITICA_GB) / (RAM_FOLGA_GB - RAM_CRITICA_GB)
                novo = MIN_CONCORRENTE + int(ratio * (MAX_REQUEST - MIN_CONCORRENTE))
            novo = max(MIN_CONCORRENTE, min(novo, MAX_REQUEST))
            if novo != anterior:
                await request_max.set_maximo(novo)
                print(
                    f"[concorrencia] disponivel={disp_gb:.2f}GB -> "
                    f"limite={novo}/{MAX_REQUEST} (em_execucao={request_max.em_execucao})",
                    flush=True,
                )
                anterior = novo
        except Exception as e:
            print(f"[concorrencia] erro: {e}", flush=True)
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
    # ajuste dinamico do limite de concorrencia por carga — roda SEMPRE (em cada
    # worker), pois e o que throttla as reconstrucoes, nao apenas um log.
    conc_task = asyncio.create_task(monitor_concorrencia())
    #limpa os jobs prontos a cada 1s, que ainda estao na memoria apos 1 min
    gc_task = asyncio.create_task(_gc_jobs())
    yield
    for task in (gc_task, conc_task, monitor_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    MODELS.clear()


# cria a instancia do fastapi
# definindo o ciclo de vida do app para carregar os modelos ao iniciar e limpar ao finalizar
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


# funcao assincrona: p receber sinal, e criar um processo e realizar reconstrucao em background e retornar o id do job
@app.post("/reconstruct/{model_id}")
async def reconstruct(
    cliente_id: str, algorithm: str, model_id: str, sinal: Sinal, complete: bool = True
):
    #verifica se o modelo existe
    if model_id not in MODELS:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": f"modelo '{model_id}' não encontrado. segue os modelos disponiveis: {list(MODELS.keys())}",
            },
        )
    # verifica a compatibilidade
    esperado = MODELS[model_id]["S"]
    if len(sinal.g) != esperado:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": f"tamanho do sinal g={len(sinal.g)} diferente do esperado {esperado} para o modelo {model_id}",
            },
        )

    #sinal
    g = np.asarray(sinal.g, dtype=np.float32)

    # cria o job (estado pending) e dispara o processamento em background.
    # o worker responsavel pelo processo é o prefixo do id, e o proxy redireciona a consulta do job para a porta do worker
    job_id = f"{WORKER_PORT}-{next(_job_seq)}"
    jobs[job_id] = {"status": "pending"}
    asyncio.create_task(processar_reconstrucao(job_id, algorithm, model_id, g))

    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


# Executa a reconstrucao de um job em background e grava o resultado final em jobs[job_id].
async def processar_reconstrucao(
    job_id: str, algorithm: str, model_id: str, g: np.ndarray
):
    
    # controle de recurso, esperando ter memoria minima livre, entao ele aguarda ate no maximo 5 min
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
                #cria uma thread para construcao da imagem
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


GRACA_RESULT_S = 60.0

# consulta o job, retorna ou pendente ou o resultado
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
    # terminal: marca o tempo da primeira leitura terminal para o GC e responde.
    # Em retries subsequentes, retorna o MESMO corpo (idempotente).
    job.setdefault("_done_at", time.monotonic())
    body = {k: v for k, v in job.items() if not k.startswith("_")}
    http = job.get("_http", 200)
    return JSONResponse(status_code=http, content=body)

#remove os processos lidos ha mais de 60s
async def _gc_jobs():
    while True:
        try:
            agora = time.monotonic()
            expirados = [
                jid for jid, j in jobs.items()
                if "_done_at" in j and agora - j["_done_at"] > GRACA_RESULT_S
            ]
            for jid in expirados:
                jobs.pop(jid, None)
        except Exception as e:
            print(f"[_gc_jobs] erro: {e}")
        await asyncio.sleep(1.0)


# funcao que calcula o percentil p de uma lista de valores ordenados, retornando None se a lista estiver vazia
def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    n = len(sorted_values)
    k = max(0, min(n - 1, int(round(p * (n - 1)))))
    return sorted_values[k]

# metricas 
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



#
# Mesma arquitetura do servidor Rust: load balancer com workers
#
# - POST /reconstruct e distribuido por round-robin entre os workers.
# Entao as requisicoes sao distribuidas de modo balanceado entre os workers
# - O job_id devolvido vem prefixado com a porta do worker dono (`porta-seq`),
#   entao o GET /result/{job_id} e roteado de volta para o mesmo worker 
# Entao o 
# - Cada worker tem seu proprio store de jobs em memoria;

# portas dos workers backend. Cada processo de proxy (uvicorn multi-worker) herda
# a lista via env var WORKER_PORTS_ENV, setada pelo supervisor antes de subir os
# proxies. No supervisor fica vazia (ele nao serve requests).
WORKER_PORTS: list[int] = [
    int(p) for p in os.environ.get("WORKER_PORTS_ENV", "").split(",") if p
]
_rr = itertools.count()  # contador round-robin (por processo de proxy)

# Session compartilhada do proxy para o hop proxy->worker (keep-alive + pool).
# Inicializada no startup de cada processo de proxy (proxy_lifespan).
_proxy_session: "requests.Session | None" = None

# pool de threads de forward POR PROCESSO de proxy. Como agora ha varios processos
# de proxy dividindo a carga, cada um precisa de um pool menor.
PROXY_POOL = int(os.environ.get("PROXY_POOL", "128"))


# monitor de memoria para o proxy
# ele monitora o sistema e o processo do proxy
# imprime a memoria usada e disponivel do sistema
class MonitorRecursos:
    """Amostra CPU% e RSS da arvore de processos do server (supervisor + proxies +
    workers) a cada `intervalo_s`, grava num CSV e, ao parar, gera um grafico PNG
    para analisar o uso de CPU/memoria ao longo da vida do servidor.

    Roda numa thread daemon. stop() sinaliza, espera a thread fechar o CSV e entao
    gera o grafico na thread principal (matplotlib Agg) — feito ANTES de matar os
    workers, com o server ainda no ar."""

    def __init__(
        self,
        root_pid: int,
        csv_path: str = "recursos_python_server.csv",
        png_path: str = "recursos_python_server.png",
        intervalo_s: float = 1.0,
    ):
        self.root_pid = root_pid
        self.csv_path = csv_path
        self.png_path = png_path
        self.intervalo_s = intervalo_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._amostras: list[list] = []

    def start(self):
        self._thread.start()

    def stop(self, gerar_grafico: bool = True):
        self._stop.set()
        self._thread.join(timeout=10)
        if gerar_grafico:
            self._gerar_grafico()

    def _arvore(self):
        try:
            raiz = psutil.Process(self.root_pid)
            return [raiz, *raiz.children(recursive=True)]
        except psutil.Error:
            return []

    def _run(self):
        GB = 1024**3
        # mantem um psutil.Process por pid entre os ticks: cpu_percent(interval=None)
        # so retorna valor real a partir da 2a chamada (a 1a so fixa o baseline).
        trackers: dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # prime CPU do sistema
        t0 = time.perf_counter()
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["t_s", "cpu_app_pct", "rss_app_gb", "cpu_sys_pct",
                 "mem_used_gb", "mem_avail_gb", "n_procs"]
            )
            while not self._stop.is_set():
                procs = self._arvore()
                vivos = set()
                cpu_app = 0.0
                rss_app = 0
                for pr in procs:
                    pid = pr.pid
                    vivos.add(pid)
                    if pid not in trackers:
                        trackers[pid] = pr
                        try:
                            pr.cpu_percent(interval=None)  # baseline
                        except psutil.Error:
                            pass
                    try:
                        cpu_app += trackers[pid].cpu_percent(interval=None)
                        rss_app += trackers[pid].memory_info().rss
                    except psutil.Error:
                        pass
                for pid in [p for p in trackers if p not in vivos]:
                    del trackers[pid]
                vm = psutil.virtual_memory()
                cpu_sys = psutil.cpu_percent(interval=None)
                row = [
                    round(time.perf_counter() - t0, 1),
                    round(cpu_app, 1),
                    round(rss_app / GB, 3),
                    round(cpu_sys, 1),
                    round(vm.used / GB, 3),
                    round(vm.available / GB, 3),
                    len(procs),
                ]
                w.writerow(row)
                f.flush()
                self._amostras.append(row)
                # tambem ecoa no terminal (mantem o comportamento antigo de 1 linha/s)
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[recursos {ts}] cpu_app={row[1]:.0f}% rss_app={row[2]:.2f}GB "
                    f"cpu_sys={row[3]:.0f}% disponivel={row[5]:.2f}GB",
                    flush=True,
                )
                self._stop.wait(self.intervalo_s)

    def _gerar_grafico(self):
        if not self._amostras:
            print("[recursos] sem amostras, grafico nao gerado", flush=True)
            return
        # import lazy: so o supervisor plota; nao infla a RAM dos workers/proxies
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [a[0] for a in self._amostras]
        cpu_app = [a[1] for a in self._amostras]
        rss_app = [a[2] for a in self._amostras]
        cpu_sys = [a[3] for a in self._amostras]
        mem_avail = [a[5] for a in self._amostras]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(ts, cpu_app, color="tab:blue", label="CPU app (%)")
        ax1.plot(ts, cpu_sys, color="tab:orange", alpha=0.5, label="CPU sistema (%)")
        ax1.set_ylabel("CPU (%)")
        ax1.set_title("Uso de recursos - server Python")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax2.plot(ts, rss_app, color="tab:red", label="RSS app (GB)")
        ax2.plot(ts, mem_avail, color="tab:green", alpha=0.5, label="RAM disponivel (GB)")
        ax2.set_ylabel("Memoria (GB)")
        ax2.set_xlabel("tempo (s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=110)
        plt.close(fig)
        print(f"[recursos] salvo: {self.csv_path} + {self.png_path}", flush=True)

# gerencia o ciclo de vida do proxy
@asynccontextmanager
async def proxy_lifespan(app: FastAPI):
    # inicializa o executor e a session do proxy
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=PROXY_POOL, thread_name_prefix="proxy-fwd"
    )
    asyncio.get_running_loop().set_default_executor(executor)
    # o executor serve para offloadar o I/O dos requests para threads

    # proxy session serve para manter as conexoes vivas e limitar a quantidade de conexoes simultaneas com os workers
    global _proxy_session
    _proxy_session = requests.Session()

    # o adapter serve para configurar o pool de conexoes e os retrys das conexoes com os workers
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max(4, len(WORKER_PORTS)),
        pool_maxsize=PROXY_POOL,
        max_retries=Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        ),
    )
    # montamos o adapter para http://, que e o esquema usado para falar com os workers (http://<worker_port>)
    _proxy_session.mount("http://", adapter)

    yield
    executor.shutdown(wait=False, cancel_futures=True)
    _proxy_session.close()


proxy_app = FastAPI(lifespan=proxy_lifespan)

# endpoint de reconstrucao do proxy, ele recebe a request do cliente, extrai o model_id e o corpo, escolhe um worker por round-robin e encaminha a request p worker.
@proxy_app.post("/reconstruct/{model_id}")
async def proxy_reconstruct(model_id: str, request: Request):
    # roundrobin usado para distribuir a carga entre os workers de modo que cada worker receba de forma equilibrada as requests
    port = WORKER_PORTS[next(_rr) % len(WORKER_PORTS)]
    body = await request.body()
    params = dict(request.query_params)
    url = f"http://127.0.0.1:{port}/reconstruct/{model_id}"

    def _forward():
        return _proxy_session.post(
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

# endpoint do resultado, ele extrai a porta do worker que esta responsavel pelo job a partir do jobid, e encaminha a request p worker.
# ele retorna o resultado do worker ao cliente
@proxy_app.get("/result/{job_id}")
async def proxy_result(job_id: str):
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
        r = await asyncio.to_thread(_proxy_session.get, url, timeout=30)
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": f"proxy -> worker:{port} falhou: {e}"},
        )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")

# endpoint para verificar a saude do proxy
@proxy_app.get("/health")
async def proxy_health():
    return {"status": "ok", "mode": "proxy", "workers": WORKER_PORTS}

# processo p verificar se o worker subiu e esta respondendo em /health, com timeout de 60s. Retorna True se responder 200, False se nao responder em 60s.
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

# verifica se a porta esta livre
def _porta_livre(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0

#elimina o processo e seus filhos p liberar recursos
def _matar_arvore(proc) -> None:
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    alvos = parent.children(recursive=True)
    alvos.append(parent)
    for p in alvos:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, vivos = psutil.wait_procs(alvos, timeout=5)
    for p in vivos:
        try:
            p.kill()
        except psutil.Error:
            pass

# funcao p verificar se o server esta vivo e os modelos carregados
@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODELS.keys())}


MIN_WORKERS = 1
# RAM media que cada worker ocupa so com os modelos H carregados (~0.8GB medido
# nesta base). E o custo FIXO por worker, e portanto o que limita quantos cabem.
RAM_POR_WORKER_GB = 0.8
# folga reservada p/ o SO + proxies + working set transitorio das reconstrucoes.
MARGEM_GB = 1.0


def calcular_topologia() -> tuple[int, int]:
    """Calcula (n_workers, n_proxies) escalando com os DOIS recursos da maquina:
    workers = min(teto_cpu, teto_ram). Maquina mais forte (mais nucleos/RAM) sobe
    os dois tetos -> mais workers; mais fraca -> menos. O teto de RAM e o que evita
    saturar/swappar (cada worker custa ~0.8GB fixo). Usa RAM *disponivel*, entao
    tambem adapta ao que ja esta rodando na maquina.

    As env vars WORKERS / PROXIES, se setadas, sobrescrevem o calculo."""
    cpu = os.cpu_count() or 2
    disp_gb = psutil.virtual_memory().available / GB

    # teto por CPU: paralelismo util ~ nucleos. teto por RAM: quantos workers cabem.
    teto_cpu = max(2, cpu // 2)
    teto_ram = max(1, int((disp_gb - MARGEM_GB) / RAM_POR_WORKER_GB))
    n_workers = min(teto_cpu, teto_ram)

    # proxies sao leves (so encaminham, nao carregam modelo): bound por CPU, com teto.
    n_proxies = max(1, min(cpu // 2, 4))

    # env vars sobrescrevem o calculo automatico
    n_workers = int(os.environ.get("WORKERS", n_workers))
    n_proxies = int(os.environ.get("PROXIES", n_proxies))
    print(
        f"[supervisor] topologia: cpu={cpu} disponivel={disp_gb:.1f}GB -> "
        f"workers={n_workers} (teto_cpu={teto_cpu}, teto_ram={teto_ram}) proxies={n_proxies}",
        flush=True,
    )
    return n_workers, n_proxies


if __name__ == "__main__":
  
    args = sys.argv[1:]
    is_worker = "--worker" in args
    port = int(_parse_arg(args, "--port", "8000"))

    
    if is_worker:
       #inicia um worker para servir o app principal.
        uvicorn.run(
            "server:app", host="0.0.0.0", port=port,
            backlog=8192, limit_concurrency=2000,
        )
    else:
        # orquestrador: inicia os workers backend e o proxy frontend.
        # worker é um processo independente que realiza o processamento pesado de reconstrução de imagens. 
        # Cada worker carrega os modelos em memória e expõe a API para receber solicitações de reconstrução e retornar resultados.
        # O proxy, por outro lado, é responsável por receber as solicitações dos clientes e distribuí-las entre os workers disponíveis, além de monitorar a memória do sistema para evitar sobrecarga.
        n_workers, n_proxies = calcular_topologia()
        #ajusta as portas dos workers para evitar conflitos, cada worker recebe uma porta única incrementada a partir da porta base (8000). O proxy escuta na porta 8000 e encaminha as solicitações para os workers nas portas 8001, 8002, etc.
        worker_ports = [port + 1 + i for i in range(n_workers)]

        # se a porta ja estiver em uso, aborta
        ocupadas = [p for p in (port, *worker_ports) if not _porta_livre(p)]
        if ocupadas:
            print(
                f"[supervisor] portas ja em uso: {ocupadas}. Encerre processos antigos "
                f"(ex: taskkill /F /IM python.exe) antes de subir o servidor."
            )
            sys.exit(1)

        # lista de processos filhos (workers)
        children: list[subprocess.Popen] = []
        print(f"[supervisor] iniciando {n_workers} workers nas portas {worker_ports}")
        for p in worker_ports:
            env = os.environ.copy()
            env["WORKER_PORT"] = str(p)      # prefixo do job_id
            env["DISABLE_MONITOR"] = "1"     # so o supervisor monitora memoria
            env["N_WORKERS"] = str(n_workers)  # p/ o worker dividir a concorrencia total
            child = subprocess.Popen(
                [sys.executable, "server.py", "--worker", "--port", str(p)], env=env
            )
            children.append(child)

        # aqui faz a verfificao p saber se os workers subiram. caso nao subiram em 60s, manda um aviso no terminal
        for p in worker_ports:
            if not _esperar_health(p, 60):
                print(f"[supervisor] worker {p} nao subiu em 60s")
        print(f"[supervisor] {n_workers} workers prontos em {worker_ports}")

        # passa as portas dos workers para os processos de proxy 
        os.environ["WORKER_PORTS_ENV"] = ",".join(str(p) for p in worker_ports)

        # monitora CPU+memoria da arvore toda (supervisor + proxies + workers),
        # gravando recursos_python_server.csv a cada 1s.
        monitor = MonitorRecursos(os.getpid())
        monitor.start()

        print(f"[supervisor] subindo {n_proxies} processos de proxy na porta {port}")
        try:
            #roda o proxy com multiworkers do server
            uvicorn.run(
                "server:proxy_app", host="0.0.0.0", port=port, workers=n_proxies,
                backlog=8192, limit_concurrency=2000,
            )
        finally:
            # para o monitor e gera o grafico ANTES de matar os workers (server ainda
            # vivo). So depois encerra a arvore de processos filhos.
            monitor.stop(gerar_grafico=True)
            print("\n[supervisor] encerrando workers...")
            for c in children:
                _matar_arvore(c)
