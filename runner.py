"""
Wrapper para client.py: define RUNNER_SEED no ambiente antes de executar para
que cada thread cliente do client.py use seu proprio random.Random(seed, client_id)
em vez do modulo global. Garante que duas chamadas com o mesmo seed produzam
exatamente a mesma sequencia de escolhas (img, algo, ganho, sleep_time) por
client_id, permitindo comparacao pareada Python x Rust com workload identico.

Uso interno (chamado por comparativo.py):
    python runner.py <seed>
"""

import os
import runpy
import sys

if len(sys.argv) < 2:
    print("uso: runner.py <seed>", file=sys.stderr)
    sys.exit(2)

os.environ["RUNNER_SEED"] = sys.argv[1]
sys.argv = ["client.py"]
runpy.run_path("client.py", run_name="__main__")
