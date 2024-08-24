from python import Python
from algorithm import parallelize
from math import sqrt, pow
from sys.info import num_cores
from time import time  # Mudei de now() para time()

# Estrutura para parâmetros de geração dos dados
struct ZetaParams:
    var start: Float64  # Início do intervalo
    var end: Float64    # Fim do intervalo
    var step: Float64   # Passo entre os valores

# Função para calcular a função zeta
fn zeta(s: Float64) -> Float64:
    var sum: Float64 = 0.0
    var n: Int = 1
    var term: Float64 = 1.0 / pow(n.cast[Float64](), s)
    
    while term > 1e-10:  # Continua até o termo ser muito pequeno
        sum += term
        n += 1
        term = 1.0 / pow(n.cast[Float64](), s)
    
    return sum

# Função para gerar um pedaço do dataset zeta
fn generate_zeta_chunk(params: ZetaParams, start_idx: Int, end_idx: Int) -> PythonObject:
    let np = Python.import_module("numpy")
    let pd = Python.import_module("pandas")
    
    # Cria um array de valores s
    var s_values = np.arange(params.start + start_idx * params.step, 
                             params.start + end_idx * params.step, 
                             params.step)
    # Calcula os valores da função zeta para cada valor de s
    var zeta_values = np.array([zeta(s) for s in s_values])
    
    # Retorna um DataFrame com os resultados
    return pd.DataFrame({"s": s_values, "zeta(s)": zeta_values})

# Função principal para gerar o dataset
fn generate_zeta_dataset(start: Float64, end: Float64, step: Float64, filename: String):
    let params = ZetaParams(start, end, step)
    let num_points = ((end - start) / step).cast[Int]()  # Número total de pontos a serem gerados
    let chunk_size = (num_points + num_cores() - 1) // num_cores()  # Divide o trabalho em chunks para cada núcleo
    
    # Gera os chunks de dados em paralelo
    var results = parallelize[generate_zeta_chunk, PythonObject](params, num_points, chunk_size)
    
    let pd = Python.import_module("pandas")
    var df = pd.concat(results)  # Concatena todos os DataFrames dos chunks
    df.to_csv(filename, index=False)  # Salva o DataFrame em um arquivo CSV
    print("Dataset salvo como", filename)

# Função principal do programa
fn main():
    let start_time = time()  # Mudei de now() para time()
    
    let start_s: Float64 = 1.0  # Início do intervalo para s
    let end_s: Float64 = 2.0    # Fim do intervalo para s
    let step_s: Float64 = 0.01  # Passo para os valores de s
    let filename = "zeta_dataset.csv"  # Nome do arquivo para salvar os dados
    
    # Gera o dataset e o salva no arquivo
    generate_zeta_dataset(start_s, end_s, step_s, filename)
    
    let end_time = time()  # Mudei de now() para time()
    print("Tempo de execução:", (end_time - start_time) / 1e9, "segundos")  # Exibe o tempo total de execução

main()
