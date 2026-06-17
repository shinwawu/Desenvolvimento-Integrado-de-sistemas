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

HOST = "127.0.0.1"
PORT = 8000
N_INSTANCIAS = 3
N_TASKS_POR_INSTANCIA = 300


def porta_em_uso() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((HOST, PORT)); s.close(); return True
    except (ConnectionRefusedError, OSError):
        s.close(); return False


def aguardar_porta_livre(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not porta_em_uso(): return True
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
    for p in Path(".").glob("reconstructed_*.png"):
        p.unlink(missing_ok=True)
    for p in Path(".").glob("relatorio_i*.csv"):
        p.unlink(missing_ok=True)


def rodar_clientes_paralelos(server_nome: str, stagger_s: float = 10.0,
                              timeout_s: int = 1200) -> dict:
    """Spawna N_INSTANCIAS clients.py em paralelo. Stagger pequeno entre starts
    evita pico de 900 conexoes simultaneas no mesmo milissegundo."""
    env = {**os.environ, "NUM_CLIENTS": str(N_TASKS_POR_INSTANCIA)}
    procs = []
    logs = []
    t0 = time.perf_counter()
    for i in range(N_INSTANCIAS):
        log = open(f".comparativo_{server_nome}_client_{i}.log", "w")
        logs.append(log)
        p = subprocess.Popen(
            [sys.executable, "-u", "client.py"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        procs.append(p)
        if i < N_INSTANCIAS - 1:
            time.sleep(stagger_s)
    ok = fail = 0
    for p in procs:
        try:
            if p.wait(timeout=timeout_s) == 0: ok += 1
            else: fail += 1
        except subprocess.TimeoutExpired:
            p.kill(); fail += 1
    for lf in logs: lf.close()
    return {"ok_instances": ok, "fail_instances": fail,
            "duracao_s": round(time.perf_counter() - t0, 1)}


def analisar_csvs(esperado: int) -> dict:
    csvs = sorted(Path(".").glob("relatorio_i*.csv"))
    if not csvs:
        return {"csvs": 0, "rows": 0, "esperado": esperado, "convergidos": 0,
                "cgnr": 0, "cgne": 0, "com_ganho": 0, "sem_ganho": 0,
                "p50_recon_s": 0.0, "p99_recon_s": 0.0,
                "media_recon_s": 0.0, "max_recon_s": 0.0}
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    return {
        "csvs": len(csvs),
        "rows": len(df),
        "esperado": esperado,
        "convergidos": int(df["converg"].sum()),
        "cgnr": int((df["algorithm"] == "CGNR").sum()),
        "cgne": int((df["algorithm"] == "CGNE").sum()),
        "com_ganho": int(df["ganho_aplicado"].sum()),
        "sem_ganho": int((~df["ganho_aplicado"]).sum()),
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
        subprocess.run(
            ["taskkill", "/F", "/IM", "Desenvolvimento-Integrado-de-sistemas.exe"],
            capture_output=True, check=False,
        )
    elif nome == "Python":
        for pat in ("%server.py%", "%multiprocessing-fork%"):
            subprocess.run(
                f'wmic process where "commandline like \'{pat}\' and not commandline like \'%wmic%\'" delete',
                shell=True, capture_output=True, check=False,
            )


def rodar_contra_server(nome: str, cmd: list, ready_timeout: int = 120) -> dict:
    print(f"\n{'='*64}\n  {nome}: {N_INSTANCIAS} clients.py x {N_TASKS_POR_INSTANCIA} reqs\n{'='*64}")
    if not aguardar_porta_livre(timeout_s=20):
        raise RuntimeError(f"porta {PORT} ocupada antes de subir {nome}")
    limpar_artefatos()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        print(f"  [{nome}] aguardando ready...")
        if not aguardar_servidor_pronto(timeout_s=ready_timeout):
            raise RuntimeError(f"{nome} nao respondeu em {ready_timeout}s")
        print(f"  [{nome}] pronto, disparando carga")
        carga = rodar_clientes_paralelos(nome)
        relat = analisar_csvs(esperado=N_INSTANCIAS * N_TASKS_POR_INSTANCIA)
        rps = relat["rows"] / carga["duracao_s"] if carga["duracao_s"] > 0 else 0.0
        print(f"\n  duracao:       {carga['duracao_s']}s")
        print(f"  reconstrucoes: {relat['rows']}/{relat['esperado']}")
        print(f"  convergidos:   {relat['convergidos']}/{relat['rows']}")
        print(f"  throughput:    {rps:.2f} req/s")
        if relat['rows']:
            print(f"  latencia recon: p50={relat['p50_recon_s']*1000:.0f}ms "
                  f"p99={relat['p99_recon_s']*1000:.0f}ms "
                  f"media={relat['media_recon_s']*1000:.0f}ms "
                  f"max={relat['max_recon_s']*1000:.0f}ms")
        return {"server": nome, "throughput_rps": round(rps, 2), **carga, **relat}
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)
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

    print("\n" + "="*64)
    print(f"  RESUMO COMPARATIVO  (seed={seed})")
    print("="*64)

    if not {"Python", "Rust"}.issubset(servers):
        # so uma versao rodou: relatorio simples 1 linha por server
        df = pd.DataFrame(resultados)
        df["seed"] = seed
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        print(f"\nsalvo em {out_path}")
        return

    py, rs = servers["Python"], servers["Rust"]

    # metrica -> (label legivel, "maior" melhor | "menor" melhor)
    metricas = [
        ("throughput_rps",   "vazão (req/s)",          "maior"),
        ("duracao_s",        "duração total (s)",      "menor"),
        ("rows",             "reconstruções",          "maior"),
    ]

    linhas = []
    for chave, label, melhor in metricas:
        if chave not in py or chave not in rs:
            continue
        vpy = float(py[chave])
        vrs = float(rs[chave])
        if melhor == "maior":
            vencedor = "Rust" if vrs > vpy else ("Python" if vpy > vrs else "empate")
        else:
            vencedor = "Rust" if vrs < vpy else ("Python" if vpy < vrs else "empate")
        linhas.append({
            "metrica": label,
            "python":  round(vpy, 4),
            "rust":    round(vrs, 4),
            "vencedor": vencedor,
        })

    df_final = pd.DataFrame(linhas)
    df_final.to_csv(out_path, index=False)

    print(f"  seed={seed}  carga={N_INSTANCIAS}x{N_TASKS_POR_INSTANCIA}={N_INSTANCIAS * N_TASKS_POR_INSTANCIA} reqs por server")
    print()
    print(df_final.to_string(index=False))
    cont = pd.Series([l["vencedor"] for l in linhas]).value_counts().to_dict()
    print(f"\n  Rust venceu em {cont.get('Rust', 0)}/{len(linhas)} metricas "
          f"(Python: {cont.get('Python', 0)}, empate: {cont.get('empate', 0)})")
    print(f"\nsalvo em {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-bin", default="target/release/Desenvolvimento-Integrado-de-sistemas.exe")
    ap.add_argument("--only", choices=["python", "rust", "ambos"], default="ambos")
    args = ap.parse_args()

    # seed aleatorio por invocacao, registrado no relatorio para rastreabilidade
    seed = random.SystemRandom().randint(0, 2**31 - 1)
    random.seed(seed)
    print(f"seed desta execucao: {seed}")

    py_cmd = [sys.executable, "-u", "server.py"]
    rust_path = Path(args.rust_bin)

    resultados = []
    if args.only in ("python", "ambos"):
        resultados.append(rodar_contra_server("Python", py_cmd))
    if args.only in ("rust", "ambos"):
        if not rust_path.exists():
            print(f"[Rust] binario nao encontrado: {rust_path} — pulando")
        else:
            resultados.append(rodar_contra_server("Rust", [str(rust_path)]))

    gerar_relatorios(resultados, seed=seed)


if __name__ == "__main__":
    main()
