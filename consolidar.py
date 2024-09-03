import pandas as pd
import numpy as np
from mpmath import mp
import os


def read_zero_file(filename):
    try:
        with open(filename, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {filename}")
        return []


def calculate_zeta(s):
    mp.dps = 15  # Set precision
    return complex(mp.zeta(s))


def process_zeros(zeros):
    s_values = [complex(0.5, t) for t in zeros]
    zeta_values = [calculate_zeta(s) for s in s_values]
    return pd.DataFrame({'s': s_values, 'zeta(s)': zeta_values})


def main():
    # Usar os nomes de arquivo exatos que você tem no projeto
    files = [
        "zeros1.txt",
        "zeros2.txt",
        "zeros3.txt",
        "zeros4.txt",
        "zeros5.txt"
    ]

    all_data = []

    for file in files:
        print(f"Processando {file}...")
        zeros = read_zero_file(file)
        if zeros:
            df = process_zeros(zeros)
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        output_file = "combined_zeta_data.csv"
        combined_data.to_csv(output_file, index=False)
        print(f"Consolidação de dados concluída. Saída salva em '{output_file}'")
    else:
        print("Nenhum dado foi processado. Verifique se os arquivos existem e estão no diretório correto.")


if __name__ == "__main__":
    main()