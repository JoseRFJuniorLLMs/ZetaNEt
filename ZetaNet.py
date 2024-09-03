import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

# Função para limpar e formatar strings complexas
def parse_complex(value):
    value = value.strip()
    if 'j' not in value:
        value += 'j'
    value = value.replace(',', '.')
    value = value.replace('(', '').replace(')', '')
    try:
        return complex(value)
    except ValueError:
        return np.nan

# Definindo a arquitetura da rede neural usando PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Função para calcular o erro quadrático médio
def mean_squared_error(predictions, targets):
    return nn.MSELoss()(predictions, targets)

# Função para treinar o modelo
def train_model(model, x_train, y_train, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Função principal
def main():
    start_time = time()

    # Carregar dados
    data = pd.read_csv("/content/combined_zeta_data.csv")

    # Limpar e formatar dados complexos
    data['s'] = data['s'].apply(parse_complex)
    data['zeta(s)'] = data['zeta(s)'].apply(parse_complex)

    # Verificar valores inválidos após a conversão
    if data['s'].isnull().any() or data['zeta(s)'].isnull().any():
        invalid_s = data[data['s'].isnull()]
        invalid_zeta = data[data['zeta(s)'].isnull()]
        print("Valores inválidos na coluna 's':")
        print(invalid_s)
        print("\nValores inválidos na coluna 'zeta(s)']:")
        print(invalid_zeta)
        return

    # Separar parte real e imaginária
    data['s_real'] = data['s'].apply(lambda x: x.real)
    data['s_imag'] = data['s'].apply(lambda x: x.imag)
    data['zeta_real'] = data['zeta(s)'].apply(lambda x: x.real)
    data['zeta_imag'] = data['zeta(s)'].apply(lambda x: x.imag)

    # Converter para tensores
    x_data = torch.tensor(data[['s_real', 's_imag']].values, dtype=torch.float32)
    y_data = torch.tensor(data[['zeta_real', 'zeta_imag']].values, dtype=torch.float32)

    # Normalizar dados
    x_mean = x_data.mean(dim=0)
    x_std = x_data.std(dim=0)
    y_mean = y_data.mean(dim=0)
    y_std = y_data.std(dim=0)
    x_std[x_std == 0] = 1  # Evitar divisão por zero
    y_std[y_std == 0] = 1  # Evitar divisão por zero
    x_data = (x_data - x_mean) / x_std
    y_data = (y_data - y_mean) / y_std

    # Criar e treinar o modelo
    layer_sizes = [2, 64, 64, 32, 2]
    model = NeuralNetwork(layer_sizes)
    train_model(model, x_data, y_data, 0.001, 100)  # Ajuste o learning rate para 0.001

    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        predictions = model(x_data)
        mse = mean_squared_error(predictions, y_data)
        print("Mean Squared Error:", mse.item())

        # Denormalizar previsões para plotagem
        denorm_predictions = predictions * y_std + y_mean

        # Verificar dados para plotagem
        print("Dados Reais:")
        print(data[['s_real', 's_imag']].head())
        print("Previsões Denormalizadas:")
        print(denorm_predictions[:5])

        # Plotar resultados
        plt.figure(figsize=(10, 6))
        plt.scatter(data['s_real'], data['s_imag'], color='blue', label='Real', alpha=0.5)
        plt.scatter(denorm_predictions[:, 0], denorm_predictions[:, 1], color='red', label='Previsto', alpha=0.5)
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.title('Função Zeta de Riemann: Real vs Previsto')
        plt.legend()
        plt.savefig('riemann_zeta_plot_pytorch.png')
        plt.show()  # Exibe o gráfico diretamente se estiver usando um notebook

    end_time = time()
    print("Tempo de execução:", (end_time - start_time), "segundos")

if __name__ == "__main__":
    main()
