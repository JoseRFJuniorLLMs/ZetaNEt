import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


# Definindo a arquitetura da rede neural híbrida
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        # Camadas Convolucionais
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)

        # Camadas Totalmente Conectadas
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # Aplicando CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Preparar para LSTM
        x = x.permute(0, 2, 1)  # Ajustar a dimensão para (batch, seq_len, features)

        # Aplicar LSTM
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]  # Pegar a última camada oculta

        # Aplicar FCN
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Função para treinar o modelo
def train_model(model, x_train, y_train, x_val, y_val, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_predictions = model(x_val)
            val_loss = criterion(val_predictions, y_val)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")


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

    # Dividir os dados em conjuntos de treinamento e validação
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Preparar os dados para entrada no modelo
    x_train = x_train.unsqueeze(2)  # Adicionar uma dimensão para o canal, mantendo 2 como o número de canais
    x_val = x_val.unsqueeze(2)

    # Criar e treinar o modelo
    model = HybridModel()
    train_model(model, x_train, y_train, x_val, y_val, learning_rate=0.001, epochs=200)

    # Salvar o modelo
    model_path = '/content/riemann_zeta_hybrid.pth'
    torch.save(model.state_dict(), model_path)

    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        val_predictions = model(x_val)
        mse = nn.MSELoss()(val_predictions, y_val)
        print("Mean Squared Error:", mse.item())

        # Denormalizar previsões para plotagem
        denorm_predictions = val_predictions * y_std + y_mean

        # Plotar resultados
        plt.figure(figsize=(10, 6))
        plt.scatter(data['s_real'], data['s_imag'], color='blue', label='Real', alpha=0.5)
        plt.scatter(denorm_predictions[:, 0], denorm_predictions[:, 1], color='red', label='Previsto', alpha=0.5)
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.title('Função Zeta de Riemann: Real vs Previsto')
        plt.legend()
        plt.savefig('/content/riemann_zeta_hybrid_plot.png')
        plt.show()  # Exibe o gráfico diretamente se estiver usando um notebook

    end_time = time()
    print("Tempo de execução:", (end_time - start_time), "segundos")


if __name__ == "__main__":
    main()
