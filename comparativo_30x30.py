"""
Versao do comparativo que roda a carga usando SOMENTE o modelo 30x30.

Equivale a `python comparativo.py --modelo 30x30`: cada cliente sorteia apenas
imagens do modelo 30x30 (via MODELO_ALVO, lido pelo client.py). O resultado vai
para relatorio_comparativo_30x30.csv.

Aceita os mesmos argumentos extras do comparativo (ex: --only python, --rust-bin).
"""

import sys

import comparativo

if __name__ == "__main__":
    # injeta --modelo 30x30 (a menos que o usuario tenha passado --modelo)
    if "--modelo" not in sys.argv:
        sys.argv += ["--modelo", "30x30"]
    comparativo.main()
