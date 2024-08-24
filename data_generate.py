import numpy as np
import pandas as pd
from mpmath import zeta

def generate_zeta_dataset(start, end, step, filename):
    # Gerar uma lista de valores de s
    s_values = np.arange(start, end + step, step)
    
    # Calcular os valores correspondentes de zeta(s)
    zeta_values = [float(zeta(s)) for s in s_values]
    
    # Criar um DataFrame para armazenar os resultados
    df = pd.DataFrame({
        's': s_values,
        'zeta(s)': zeta_values
    })
    
    # Salvar o DataFrame em um arquivo CSV
    df.to_csv(filename, index=False)
    print(f"Dataset salvo como {filename}")

# Configurações do dataset
start_s = 1.0   # Valor inicial de s
end_s = 2.0     # Valor final de s
step_s = 0.01   # Incremento de s
filename = "zeta_dataset.csv"

# Gerar o dataset
generate_zeta_dataset(start_s, end_s, step_s, filename)
