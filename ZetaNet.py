import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time


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

        # Exibir perda a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Função principal
def main():
    start_time = time()

    # Carregar dados
    data = pd.read_csv("large_dataset.csv")
    x_data = torch.tensor(data['s'].values.reshape(-1, 1), dtype=torch.float32)
    y_data = torch.tensor(data['zeta(s)'].values.reshape(-1, 1), dtype=torch.float32)

    # Normalizar dados
    x_mean = x_data.mean()
    x_std = x_data.std()
    y_mean = y_data.mean()
    y_std = y_data.std()
    x_data = (x_data - x_mean) / x_std
    y_data = (y_data - y_mean) / y_std

    # Criar e treinar o modelo
    layer_sizes = [1, 64, 64, 32, 1]
    model = NeuralNetwork(layer_sizes)
    train_model(model, x_data, y_data, 0.01, 100)

    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        predictions = model(x_data)
        mse = mean_squared_error(predictions, y_data)
        print("Mean Squared Error:", mse.item())

        # Denormalizar previsões para plotagem
        denorm_predictions = predictions * y_std + y_mean

        # Plotar resultados
        plt.figure(figsize=(10, 6))
        plt.scatter(data['s'], data['zeta(s)'], color='blue', label='Real', alpha=0.5)
        plt.scatter(data['s'], denorm_predictions.numpy(), color='red', label='Previsto', alpha=0.5)
        plt.xlabel('s')
        plt.ylabel('zeta(s)')
        plt.title('Função Zeta de Riemann: Real vs Previsto')
        plt.legend()
        plt.savefig('riemann_zeta_plot_pytorch.png')
        plt.close()

    end_time = time()
    print("Tempo de execução:", (end_time - start_time), "segundos")


if __name__ == "__main__":
    main()
