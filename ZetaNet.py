import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Define the dataset class
class RiemannZetaDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.s_values = torch.tensor(self.data['s'].values, dtype=torch.float32).unsqueeze(1)
        self.zeta_values = torch.tensor(self.data['zeta(s)'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.s_values[idx], self.zeta_values[idx]

# Define the neural network model
class RiemannZetaNet(nn.Module):
    def __init__(self):
        super(RiemannZetaNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for s, zeta in train_loader:
            optimizer.zero_grad()
            outputs = model(s)
            loss = criterion(outputs, zeta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for s, zeta in test_loader:
            outputs = model(s)
            predictions.extend(outputs.numpy().flatten())
            actuals.extend(zeta.numpy().flatten())
    
    return predictions, actuals

# Main function to run the entire process
def main():
    # Load and prepare the data
    dataset = RiemannZetaDataset('zeta_dataset.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = RiemannZetaNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    predictions, actuals = evaluate_model(model, test_loader)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset.s_values, dataset.zeta_values, color='blue', label='Actual', alpha=0.5)
    plt.scatter(dataset.s_values[test_size:], predictions, color='red', label='Predicted', alpha=0.5)
    plt.xlabel('s')
    plt.ylabel('zeta(s)')
    plt.title('Riemann Zeta Function: Actual vs Predicted')
    plt.legend()
    plt.savefig('riemann_zeta_plot.png')
    plt.close()

    print("Training and evaluation complete. Results plotted in 'riemann_zeta_plot.png'")

if __name__ == "__main__":
    main()