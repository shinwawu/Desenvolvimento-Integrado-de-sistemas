"""
Comparativo Python vs Rust: 3 clientes em paralelo x 300 reqs cada (900 total)
contra cada servidor, na mesma maquina.

Cada invocacao do comparativo sorteia um seed aleatorio que e logado no resumo
final — assim cada execucao pega uma combinacao diferente de tasks. A
comparacao e estatistica (3x300=900 amostras por server) ja que o client.py
faz suas escolhas (img/algo/ganho) com seu proprio random.
"""

import argparse
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import psutil

IS_WINDOWS = os.name == "nt"
# nome do binario Rust: com .exe no Windows, sem extensao no macOS/Linux
RUST_BIN_NAME = "Desenvolvimento-Integrado-de-sistemas" + (".exe" if IS_WINDOWS else "")
RUST_BIN_DEFAULT = str(Path("target/release") / RUST_BIN_NAME)

HOST = "127.0.0.1"
PORT = 8000
N_INSTANCIAS = 3
N_TASKS_POR_INSTANCIA = 300
# numero de workers backend por servidor: 2 para ambos (comparacao simetrica).
N_WORKERS_PYTHON = 2
N_WORKERS_RUST = 2


def porta_em_uso() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((HOST, PORT))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        s.close()
        return False


def aguardar_porta_livre(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not porta_em_uso():
            return True
        time.sleep(0.5)
    return False


def aguardar_servidor_pronto(timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for path in ("/health", "/openapi.json"):
            try:
                urllib.request.urlopen(f"http://{HOST}:{PORT}{path}", timeout=2)
                return True
            except Exception:
                continue
        time.sleep(0.5)
    return False


def limpar_artefatos() -> None:
    """Limpa artefatos do run atual (com INSTANCE_ID=iXXXX, sem sufixo de
    server). Nao toca nos arquivos ja preservados (com prefixo do server)."""
    for p in Path(".").glob("reconstructed_i*.png"):
        p.unlink(missing_ok=True)
    for p in Path(".").glob("relatorio_i*.csv"):
        p.unlink(missing_ok=True)
    for p in Path(".").glob("relatorio_clientes_i*.txt"):
        p.unlink(missing_ok=True)


def limpar_relatorios_antigos() -> None:
    """Chamado uma vez no inicio de comparativo.py para apagar relatorios e
    imagens preservados de invocacoes anteriores (com sufixo _python / _rust)."""
    for prefixo in (
        "relatorio_python_",
        "relatorio_rust_",
        "relatorio_clientes_python_",
        "relatorio_clientes_rust_",
        "reconstructed_python_",
        "reconstructed_rust_",
    ):
        for p in Path(".").glob(f"{prefixo}*"):
            p.unlink(missing_ok=True)


def preservar_relatorios(server_nome: str) -> None:
    """Renomeia os artefatos do run que acabou de terminar (relatorios e
    imagens) para incluir o nome do server (python/rust), evitando que o
    proximo run os sobrescreva ou que limpar_artefatos os apague."""
    nome = server_nome.lower()
    for p in Path(".").glob("relatorio_i*.csv"):
        # relatorio_iXXXX.csv -> relatorio_<nome>_iXXXX.csv
        novo = p.with_name(f"relatorio_{nome}_{p.name[len('relatorio_'):]}")
        p.rename(novo)
    for p in Path(".").glob("relatorio_clientes_i*.txt"):
        # relatorio_clientes_iXXXX.txt -> relatorio_clientes_<nome>_iXXXX.txt
        novo = p.with_name(
            f"relatorio_clientes_{nome}_{p.name[len('relatorio_clientes_'):]}"
        )
        p.rename(novo)
    for p in Path(".").glob("reconstructed_i*.png"):
        # reconstructed_iXXXX_...png -> reconstructed_<nome>_iXXXX_...png
        novo = p.with_name(f"reconstructed_{nome}_{p.name[len('reconstructed_'):]}")
        p.rename(novo)


def rodar_clientes_paralelos(
    server_nome: str, seeds: list[int], timeout_s: int = 1200
) -> dict:
    """Spawna N_INSTANCIAS clients.py em paralelo via runner.py com um seed
    deterministico por instancia. Passar a mesma lista de seeds em duas rodadas
    (Python e Rust) garante mesmo workload nas duas. Stagger fixo de 2s entre
    starts evita pico de 900 conexoes simultaneas no mesmo milissegundo."""
    env = {**os.environ, "NUM_CLIENTS": str(N_TASKS_POR_INSTANCIA)}
    procs = []
    logs = []
    t0 = time.perf_counter()
    for i in range(N_INSTANCIAS):
        log = open(f".comparativo_{server_nome}_client_{i}.log", "w")
        logs.append(log)
        p = subprocess.Popen(
            [sys.executable, "-u", "runner.py", str(seeds[i])],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        if i < N_INSTANCIAS - 1:
            time.sleep(2.0)
    ok = fail = 0
    for p in procs:
        try:
            if p.wait(timeout=timeout_s) == 0:
                ok += 1
            else:
                fail += 1
        except subprocess.TimeoutExpired:
            p.kill()
            fail += 1
    for lf in logs:
        lf.close()
    return {
        "ok_instances": ok,
        "fail_instances": fail,
        "duracao_s": round(time.perf_counter() - t0, 1),
    }


def _relatorio_vazio(esperado: int) -> dict:
    return {
        "csvs": 0,
        "rows": 0,
        "esperado": esperado,
        "convergidos": 0,
        "cgnr": 0,
        "cgne": 0,
        "com_ganho": 0,
        "sem_ganho": 0,
        "m_30x30": 0,
        "m_60x60": 0,
        "p50_recon_s": 0.0,
        "p99_recon_s": 0.0,
        "media_recon_s": 0.0,
        "max_recon_s": 0.0,
    }


def analisar_csvs(esperado: int) -> dict:
    csvs = sorted(Path(".").glob("relatorio_i*.csv"))
    if not csvs:
        return _relatorio_vazio(esperado)
    # um client que ficou sem servico (todas as reqs deram timeout, ex: servidor
    # saturado/sem memoria) grava um CSV vazio (so o newline, sem header). Pular
    # esses para nao abortar a comparacao inteira com EmptyDataError — o relatorio
    # ainda reflete as reconstrucoes que os outros clients conseguiram.
    frames = []
    for c in csvs:
        try:
            frames.append(pd.read_csv(c))
        except pd.errors.EmptyDataError:
            print(f"  [aviso] {c.name} vazio (client sem reconstrucoes) — ignorado")
    if not frames:
        return _relatorio_vazio(esperado)
    df = pd.concat(frames, ignore_index=True)
    return {
        "csvs": len(frames),
        "rows": len(df),
        "esperado": esperado,
        "convergidos": int(df["converg"].sum()),
        "cgnr": int((df["algorithm"] == "CGNR").sum()),
        "cgne": int((df["algorithm"] == "CGNE").sum()),
        "com_ganho": int(df["ganho_aplicado"].sum()),
        "sem_ganho": int((~df["ganho_aplicado"]).sum()),
        "m_30x30": int((df["model_id"] == "30x30").sum()),
        "m_60x60": int((df["model_id"] == "60x60").sum()),
        "p50_recon_s": round(float(df["reconstruction_time"].median()), 4),
        "p99_recon_s": round(float(df["reconstruction_time"].quantile(0.99)), 4),
        "media_recon_s": round(float(df["reconstruction_time"].mean()), 4),
        "max_recon_s": round(float(df["reconstruction_time"].max()), 4),
    }


def matar_workers_fantasma(nome: str) -> None:
    """uvicorn workers=N (Python) spawna children via multiprocessing.spawn que
    herdam o socket; Rust multi-process spawna children do mesmo binario. Ambos
    podem ficar segurando a porta 8000 mesmo depois do parent morrer."""
    if nome == "Rust":
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/IM", RUST_BIN_NAME],
                capture_output=True,
                check=False,
            )
        else:
            # pkill -f casa pelo command line (path do binario + children spawnados),
            # cobrindo o parent e os processos filhos que herdam a porta 8000.
            subprocess.run(
                ["pkill", "-f", "Desenvolvimento-Integrado-de-sistemas"],
                capture_output=True,
                check=False,
            )
    elif nome == "Python":
        if IS_WINDOWS:
            for pat in ("%server.py%", "%multiprocessing-fork%"):
                subprocess.run(
                    f"wmic process where \"commandline like '{pat}' and not commandline like '%wmic%'\" delete",
                    shell=True,
                    capture_output=True,
                    check=False,
                )
        else:
            # parent (server.py) + workers uvicorn (multiprocessing spawn/fork)
            for pat in ("server.py", "multiprocessing"):
                subprocess.run(
                    ["pkill", "-f", pat],
                    capture_output=True,
                    check=False,
                )


def aguardar_memoria_liberada(alvo_gb: float = 1.5, timeout_s: int = 30) -> None:
    """Depois de matar um server, os workers (cada um segura ~600MB de modelos)
    morrem mas o SO leva um instante para reclamar as paginas. Sem esperar isso, o
    proximo server sobe com a memoria ainda ocupada e cai no wait-loop de admission
    control (MEMO_MINIMA=0.5GB nos dois servers), throttlando o throughput a ~zero —
    foi o que afundou o Rust quando rodou logo depois do Python.

    Faz poll de available ate (a) passar de `alvo_gb`, ou (b) estabilizar (parar de
    subir — sinal de que o SO ja reclamou o que ia reclamar). Segue mesmo assim no
    timeout: o macOS contabiliza cache como usado e pode reportar available baixo de
    forma persistente, entao nunca bloqueamos indefinidamente."""
    GB = 1024 ** 3
    inicio = time.time()
    anterior = -1.0
    estaveis = 0
    print("  aguardando SO reclamar memoria do server anterior...")
    while time.time() - inicio < timeout_s:
        avail = psutil.virtual_memory().available / GB
        if avail >= alvo_gb:
            print(f"  memoria liberada: {avail:.2f}GB disponivel")
            return
        # available parou de subir por algumas amostras -> reclamacao terminou
        if anterior >= 0 and avail - anterior < 0.02:
            estaveis += 1
            if estaveis >= 3:
                print(f"  memoria estabilizou em {avail:.2f}GB (alvo {alvo_gb}GB nao atingido)")
                return
        else:
            estaveis = 0
        anterior = avail
        time.sleep(0.5)
    avail = psutil.virtual_memory().available / GB
    print(f"  timeout ({timeout_s}s) esperando memoria — seguindo com {avail:.2f}GB disponivel")


def rodar_contra_server(
    nome: str,
    cmd: list,
    seeds: list[int],
    n_workers: int,
    ready_timeout: int = 120,
    env: dict | None = None,
) -> dict:
    print(
        f"\n{'='*64}\n  {nome}: {N_INSTANCIAS} clients.py x {N_TASKS_POR_INSTANCIA} reqs ({n_workers} workers)\n{'='*64}"
    )
    if not aguardar_porta_livre(timeout_s=20):
        raise RuntimeError(f"porta {PORT} ocupada antes de subir {nome}")
    limpar_artefatos()
    # captura stdout/stderr do server pra .comparativo_<nome>_server.log para
    # poder diagnosticar quando o ready check falha (ex: bind error, worker
    # nao subiu, OOM). Antes ia para DEVNULL e silenciava todos os erros.
    server_log = open(f".comparativo_{nome}_server.log", "w", buffering=1)
    proc = subprocess.Popen(
        cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env
    )
    try:
        print(f"  [{nome}] aguardando ready...")
        if not aguardar_servidor_pronto(timeout_s=ready_timeout):
            raise RuntimeError(
                f"{nome} nao respondeu em {ready_timeout}s "
                f"(veja .comparativo_{nome}_server.log)"
            )
        print(f"  [{nome}] pronto, disparando carga")
        carga = rodar_clientes_paralelos(nome, seeds)
        relat = analisar_csvs(esperado=N_INSTANCIAS * N_TASKS_POR_INSTANCIA)
        rps = relat["rows"] / carga["duracao_s"] if carga["duracao_s"] > 0 else 0.0
        print(f"\n  duracao:       {carga['duracao_s']}s")
        print(f"  reconstrucoes: {relat['rows']}/{relat['esperado']}")
        print(f"  convergidos:   {relat['convergidos']}/{relat['rows']}")
        print(f"  modelos:       30x30={relat['m_30x30']}  60x60={relat['m_60x60']}")
        print(f"  throughput:    {rps:.2f} req/s")

        # preserva relatorio_iXXXX.csv e relatorio_clientes_iXXXX.txt com sufixo
        # _python/_rust para o proximo run nao sobrescrever nem deletar.
        preservar_relatorios(nome)
        return {"server": nome, "throughput_rps": round(rps, 2), **carga, **relat}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        server_log.close()
        matar_workers_fantasma(nome)
        aguardar_porta_livre(timeout_s=20)
        time.sleep(2)


def gerar_relatorios(resultados: list[dict], seed: int) -> None:
    """Gera UM unico CSV (relatorio_comparativo.csv) onde cada linha e uma
    metrica e as colunas trazem o valor de cada versao + analise (delta, razao,
    vencedor). Se so uma versao rodou, emite a tabela simples (1 linha por server)."""
    if not resultados:
        print("nenhum resultado para reportar")
        return

    out_path = Path("relatorio_comparativo.csv")
    servers = {r["server"]: r for r in resultados}

    print("\n" + "=" * 64)
    print(f"  RESUMO COMPARATIVO  (seed={seed})")
    print("=" * 64)

    if not {"Python", "Rust"}.issubset(servers):
        # so uma versao rodou: relatorio simples 1 linha por server
        df = pd.DataFrame(resultados)
        df["seed"] = seed
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        print(f"\nsalvo em {out_path}")
        return

    py, rs = servers["Python"], servers["Rust"]

    metricas = [
        ("throughput_rps", "vazão (req/s)"),
        ("duracao_s", "duração total (s)"),
        ("rows", "reconstruções"),
        ("m_30x30", "reconstruções 30x30"),
        ("m_60x60", "reconstruções 60x60"),
    ]

    linhas = []
    for chave, label in metricas:
        if chave not in py or chave not in rs:
            continue
        linhas.append(
            {
                "metrica": label,
                "python": round(float(py[chave]), 4),
                "rust": round(float(rs[chave]), 4),
            }
        )

    df_final = pd.DataFrame(linhas)
    df_final.to_csv(out_path, index=False)

    print(
        f"  seed={seed}  carga={N_INSTANCIAS}x{N_TASKS_POR_INSTANCIA}={N_INSTANCIAS * N_TASKS_POR_INSTANCIA} reqs por server"
    )
    print()
    print(df_final.to_string(index=False))
    print(f"\nsalvo em {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-bin", default=RUST_BIN_DEFAULT)
    ap.add_argument("--only", choices=["python", "rust", "ambos"], default="ambos")
    args = ap.parse_args()

    # apaga relatorios preservados de invocacoes anteriores antes de comecar
    limpar_relatorios_antigos()

    # seed base aleatorio por invocacao, registrado no relatorio. Os subprocessos
    # client.py sao seeded via runner.py com seeds derivados deste base — assim
    # Python e Rust recebem EXATAMENTE o mesmo workload (mesma sequencia de img,
    # algo, ganho, intervalos), permitindo comparacao pareada.
    seed = random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)
    instance_seeds = [rng.randint(0, 2**31 - 1) for _ in range(N_INSTANCIAS)]
    print(f"seed base: {seed}  seeds das instancias: {instance_seeds}")

    # contagem de workers assimetrica: Python via env var WORKERS; Rust via --workers
    py_cmd = [sys.executable, "-u", "server.py"]
    py_env = {**os.environ, "WORKERS": str(N_WORKERS_PYTHON)}
    rust_path = Path(args.rust_bin)

    resultados = []
    if args.only in ("python", "ambos"):
        resultados.append(
            rodar_contra_server(
                "Python", py_cmd, instance_seeds, N_WORKERS_PYTHON, env=py_env
            )
        )
    if args.only in ("rust", "ambos"):
        # se o Python acabou de rodar, espera a memoria dos workers ser reclamada
        # antes de subir o Rust (senao o Rust inicia sob pressao de memoria e o
        # admission control o throttla — comparacao injusta por ordem de execucao).
        if args.only == "ambos":
            aguardar_memoria_liberada()
        if not rust_path.exists():
            print(f"[Rust] binario nao encontrado: {rust_path} — pulando")
        else:
            rust_cmd = [str(rust_path), "--workers", str(N_WORKERS_RUST)]
            resultados.append(
                rodar_contra_server("Rust", rust_cmd, instance_seeds, N_WORKERS_RUST)
            )

    gerar_relatorios(resultados, seed=seed)


if __name__ == "__main__":
    main()
