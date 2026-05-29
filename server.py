import asyncio
import os
import time
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import psutil
import scipy.sparse as sp
from fastapi import FastAPI
from fastapi.responses import JSONResponse, ORJSONResponse
from pydantic import BaseModel


# Configurações dos modelos disponíveis
MODELS_CONFIG = {
    "60x60": {"S": 50816, "N": 3600, "shape": (60, 60), "path": "data/H-1.npz"},
    "30x30": {"S": 27904, "N": 900, "shape": (30, 30), "path": "data/H-2.npz"},
}
# listar os modelos disponíveis
MODELS: dict = {}

# salva os chunks de sinais g enviados pelos clientes
CLIENT_SIGNALS: dict = {}

# ---- controle de saturacao ----
# limite de reconstrucoes simultaneas: BLAS multi-thread aproveita os cores, mas
# competir cache em N threads piora throughput; cpu_count() * 2 e um teto seguro
MAX_INFLIGHT = max(2, (os.cpu_count() or 4) * 2)
# fila e ilimitada por contagem (requisito: suportar muitos usuarios simultaneos).
# o unico motivo para rejeitar e proteger o servidor de OOM: se a memoria
# disponivel cair abaixo deste piso, novas requisicoes recebem 503.
MIN_AVAILABLE_GB = 0.5        # piso abaixo do qual o request espera (nao rejeita)
MEMORY_WAIT_DEADLINE_S = 300  # so rejeita em ultimo caso, depois de 5 min esperando
MEMORY_POLL_S = 0.5           # frequencia de re-checagem da memoria
RECON_TIMEOUT_S = 30.0
CLIENT_SIGNALS_TTL_S = 60.0  # buffers inativos sao descartados depois disso
GC_INTERVAL_S = 10.0

inflight_sem = asyncio.Semaphore(MAX_INFLIGHT)
metrics_state = {
    "inflight": 0,
    "queued": 0,
    "completed": 0,
    "rejected": 0,
    "memory_waited": 0,
    "timeout": 0,
    "failed": 0,
    "gc_evicted": 0,
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


# GC dos buffers ociosos em CLIENT_SIGNALS para evitar vazamento sob carga
async def client_signals_gc():
    while True:
        await asyncio.sleep(GC_INTERVAL_S)
        try:
            cutoff = time.monotonic() - CLIENT_SIGNALS_TTL_S
            stale = [
                cid
                for cid, v in CLIENT_SIGNALS.items()
                if isinstance(v, dict)
                and v.get("last_touched", 0) < cutoff
                and not v.get("complete", False)
            ]
            for cid in stale:
                CLIENT_SIGNALS.pop(cid, None)
            if stale:
                metrics_state["gc_evicted"] += len(stale)
                print(f"[gc] removidos {len(stale)} buffers ociosos", flush=True)
        except Exception as e:
            print(f"[gc] erro: {e}", flush=True)


# monitor de memoria: imprime uso/disponibilidade do sistema a cada 1s
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
    for mid, cfg in MODELS_CONFIG.items():
        model = load_model(mid, cfg)
        if model is not None:
            MODELS[mid] = model
    # tasks em background: monitor de memoria + GC de buffers ociosos
    monitor_task = asyncio.create_task(memoria_monitor())
    gc_task = asyncio.create_task(client_signals_gc())
    yield
    for t in (monitor_task, gc_task):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    MODELS.clear()


# cria a instancia do fastapi
# definindo o ciclo de vida do app para carregar os modelos ao iniciar e limpar ao finalizar
# optamos por usar ORJSONResponse para melhorar a performance na serialização de respostas JSON
app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)


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
    img = f.reshape(m["shape"], order="F")

    lo, hi = float(img.min()), float(img.max())
    span = hi - lo
    # normaliza a imagem para o intervalo [0, 1], se span for zero, retorna uma imagem de zeros
    norm = (img - lo) / span if span > 0 else np.zeros_like(img)
    return norm, iters, err


# classe p receber o sinal g no formato JSON, onde g é uma lista de floats
class Sinal(BaseModel):
    g: list[float]


# recebe os chunks de g. No chunk final (complete=True), executa a reconstrucao
# de forma sincrona protegida por semaphore (admission control) + memoria.
@app.post("/reconstruct/{model_id}")
async def reconstruct(
    cliente_id: str, algorithm: str, model_id: str, sinal: Sinal, complete: bool = False
):
    if CLIENT_SIGNALS.get(cliente_id) is None:
        CLIENT_SIGNALS[cliente_id] = {
            "algorithm": algorithm,
            "model_id": model_id,
            "g_parts": [],
            "complete": False,
        }
    CLIENT_SIGNALS[cliente_id]["g_parts"].extend(sinal.g)
    CLIENT_SIGNALS[cliente_id]["complete"] = complete
    CLIENT_SIGNALS[cliente_id]["last_touched"] = time.monotonic()
    if not complete:
        return {
            "message": f"sinal g recebido para {model_id}, aguardando mais partes..."
        }

    if model_id not in MODELS:
        return {
            "error": f"modelo '{model_id}' não encontrado. segue os modelos disponiveis: {list(MODELS.keys())}"
        }

    # admission control por pressao de memoria: nao rejeita por padrao -- espera
    # ate a memoria aliviar (workers vao terminando e liberando). so rejeita em
    # ultimo caso, se a espera passar de MEMORY_WAIT_DEADLINE_S (proteção contra
    # deadlock em OOM real).
    GB = 1024 ** 3
    if psutil.virtual_memory().available < MIN_AVAILABLE_GB * GB:
        metrics_state["memory_waited"] += 1
        espera_deadline = time.monotonic() + MEMORY_WAIT_DEADLINE_S
        while psutil.virtual_memory().available < MIN_AVAILABLE_GB * GB:
            if time.monotonic() > espera_deadline:
                metrics_state["rejected"] += 1
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": (
                            f"memoria abaixo de {MIN_AVAILABLE_GB}GB por mais de "
                            f"{MEMORY_WAIT_DEADLINE_S}s; rejeitando para evitar deadlock"
                        )
                    },
                )
            await asyncio.sleep(MEMORY_POLL_S)

    g = np.asarray(CLIENT_SIGNALS[cliente_id]["g_parts"], dtype=np.float32)
    algo_choice = CLIENT_SIGNALS[cliente_id]["algorithm"]
    # libera o buffer antes da reconstrucao para reduzir pico de RAM
    CLIENT_SIGNALS[cliente_id]["g_parts"] = []
    CLIENT_SIGNALS[cliente_id]["complete"] = False

    metrics_state["queued"] += 1
    try:
        async with inflight_sem:
            metrics_state["queued"] -= 1
            metrics_state["inflight"] += 1
            try:
                start_dt = datetime.now()
                t0 = time.perf_counter()
                result = await asyncio.wait_for(
                    asyncio.to_thread(reconstruct_image, algo_choice, model_id, g),
                    timeout=RECON_TIMEOUT_S,
                )
                elapsed = time.perf_counter() - t0
                end_dt = datetime.now()
            except asyncio.TimeoutError:
                metrics_state["timeout"] += 1
                return JSONResponse(
                    status_code=504,
                    content={"error": f"timeout > {RECON_TIMEOUT_S}s na reconstrução"},
                )
            except Exception as e:
                metrics_state["failed"] += 1
                return {"error": f"erro na reconstrução da imagem: {str(e)}"}
            finally:
                metrics_state["inflight"] -= 1
    except Exception:
        if metrics_state["queued"] > 0:
            metrics_state["queued"] -= 1
        raise

    if isinstance(result, dict) and "error" in result:
        metrics_state["failed"] += 1
        return {"error": f"erro na reconstrução da imagem: {result['error']}"}

    img, iters, err = result
    metrics_state["completed"] += 1
    metrics_state["times_ms"].append(elapsed * 1000)

    return {
        "message": f"reconstrucao completa para {model_id}",
        "image": img.tolist(),
        "iters": iters,
        "final_error": err,
        "reconstruction_time": elapsed,
        "start_time": start_dt.isoformat(timespec="milliseconds"),
        "end_time": end_dt.isoformat(timespec="milliseconds"),
    }


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    n = len(sorted_values)
    k = max(0, min(n - 1, int(round(p * (n - 1)))))
    return sorted_values[k]


@app.get("/metrics")
async def metrics():
    times = sorted(metrics_state["times_ms"])
    vm = psutil.virtual_memory()
    GB = 1024**3
    return {
        "limits": {
            "max_inflight": MAX_INFLIGHT,
            "queue_policy": (
                "unbounded; wait until memory recovers (no reject by count); "
                "last-resort reject only after memory_wait_deadline_s"
            ),
            "min_available_gb": MIN_AVAILABLE_GB,
            "memory_wait_deadline_s": MEMORY_WAIT_DEADLINE_S,
            "recon_timeout_s": RECON_TIMEOUT_S,
            "client_buffer_ttl_s": CLIENT_SIGNALS_TTL_S,
        },
        "counters": {
            "completed": metrics_state["completed"],
            "rejected": metrics_state["rejected"],
            "memory_waited": metrics_state["memory_waited"],
            "timeout": metrics_state["timeout"],
            "failed": metrics_state["failed"],
            "gc_evicted": metrics_state["gc_evicted"],
        },
        "gauges": {
            "inflight": metrics_state["inflight"],
            "queued": metrics_state["queued"],
            "client_buffers": sum(
                1
                for v in CLIENT_SIGNALS.values()
                if isinstance(v, dict) and v.get("g_parts")
            ),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
