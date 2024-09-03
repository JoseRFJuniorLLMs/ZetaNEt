import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Função para calcular a função zeta
def zeta(s):
    sum = 0.0
    n = 1
    term = 1.0 / n ** s

    while term > 1e-10:  # Continua até o termo ser muito pequeno
        sum += term
        n += 1
        term = 1.0 / n ** s

    return sum


# Função para gerar um pedaço do dataset zeta
def generate_zeta_chunk(params, start_idx, end_idx):
    start, step = params['start'], params['step']

    # Cria um array de valores s
    s_values = np.arange(start + start_idx * step, start + end_idx * step, step)
    # Calcula os valores da função zeta para cada valor de s
    zeta_values = np.array([zeta(s) for s in s_values])

    # Retorna um DataFrame com os resultados
    return pd.DataFrame({"s": s_values, "zeta(s)": zeta_values})


# Função principal para gerar o dataset
def generate_zeta_dataset(start, end, step, filename):
    params = {'start': start, 'end': end, 'step': step}
    num_points = int((end - start) / step)  # Número total de pontos a serem gerados
    chunk_size = (num_points + cpu_count() - 1) // cpu_count()  # Divide o trabalho em chunks para cada núcleo

    def chunk_args(start_idx, end_idx):
        return (params, start_idx, end_idx)

    with Pool(cpu_count()) as pool:
        # Gera os chunks de dados em paralelo com progresso
        results = []
        for chunk in tqdm(range(0, num_points, chunk_size), desc="Generating chunks", unit="chunk"):
            start_idx = chunk
            end_idx = min(start_idx + chunk_size, num_points)
            results.append(pool.apply_async(generate_zeta_chunk, chunk_args(start_idx, end_idx)))

        # Coleta os resultados
        results = [r.get() for r in results]

    # Concatena todos os DataFrames dos chunks
    df = pd.concat(results)
    df.to_csv(filename, index=False)  # Salva o DataFrame em um arquivo CSV
    print("Dataset salvo como", filename)


# Função principal do programa
def main():
    start_time = time.time()

    start_s = 1.0  # Início do intervalo para s
    end_s = 2.0  # Fim do intervalo para s
    step_s = (end_s - start_s) / 6250  # 62500000 Passo para gerar aproximadamente 1 GB de dados
    filename = "zeta_dataset.csv"  # Nome do arquivo para salvar os dados

    # Gera o dataset e o salva no arquivo
    generate_zeta_dataset(start_s, end_s, step_s, filename)

    end_time = time.time()
    print("Tempo de execução:", (end_time - start_time), "segundos")  # Exibe o tempo total de execução


if __name__ == "__main__":
    main()
