"""
Versao do comparativo que roda a carga usando SOMENTE o modelo 60x60.

Equivale a `python comparativo.py --modelo 60x60`: cada cliente sorteia apenas
imagens do modelo 60x60 (via MODELO_ALVO, lido pelo client.py). O resultado vai
para relatorio_comparativo_60x60.csv.

Aceita os mesmos argumentos extras do comparativo (ex: --only python, --rust-bin).
"""

import sys

import comparativo

if __name__ == "__main__":
    # injeta --modelo 60x60 (a menos que o usuario tenha passado --modelo)
    if "--modelo" not in sys.argv:
        sys.argv += ["--modelo", "60x60"]
    comparativo.main()
