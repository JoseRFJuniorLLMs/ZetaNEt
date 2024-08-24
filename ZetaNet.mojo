from python import Python
from tensor import Tensor
from random import rand
from math import exp, log, sqrt
from time import now

struct Layer:
    var weights: Tensor[DType.float32]
    var biases: Tensor[DType.float32]
    
    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weights = rand[DType.float32](input_size, output_size) * 0.01
        self.biases = rand[DType.float32](1, output_size) * 0.01
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        return x @ self.weights + self.biases

fn relu(x: Tensor[DType.float32]) -> Tensor[DType.float32]:
    return x.maximum(0)

struct NeuralNetwork:
    var layers: DynamicVector[Layer]
    
    fn __init__(inout self, layer_sizes: DynamicVector[Int]):
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        var output = x
        for i in range(len(self.layers) - 1):
            output = relu(self.layers[i].forward(output))
        return self.layers[len(self.layers) - 1].forward(output)

fn mean_squared_error(predictions: Tensor[DType.float32], targets: Tensor[DType.float32]) -> Float32:
    return ((predictions - targets) ** 2).mean()

fn train_model(
    inout model: NeuralNetwork, 
    x_train: Tensor[DType.float32], 
    y_train: Tensor[DType.float32], 
    learning_rate: Float32, 
    epochs: Int
):
    let batch_size = 32
    let num_batches = x_train.dim(0) // batch_size
    
    for epoch in range(epochs):
        var total_loss: Float32 = 0
        for i in range(num_batches):
            let start = i * batch_size
            let end = (i + 1) * batch_size
            let x_batch = x_train[start:end, :]
            let y_batch = y_train[start:end, :]
            
            let predictions = model.forward(x_batch)
            let loss = mean_squared_error(predictions, y_batch)
            total_loss += loss
            
            # Backpropagation and weight update (simplified)
            for j in range(len(model.layers) - 1, -1, -1):
                let layer = model.layers[j]
                let error = predictions - y_batch if j == len(model.layers) - 1 else error @ layer.weights.T
                let gradient = x_batch.T @ error if j == 0 else relu(model.layers[j-1].forward(x_batch)).T @ error
                
                layer.weights -= learning_rate * gradient
                layer.biases -= learning_rate * error.mean(0)
        
        if (epoch + 1) % 10 == 0:
            print("Epoch", epoch + 1, "Loss:", total_loss / num_batches)

fn main():
    let start_time = now()
    
    # Load data
    let pd = Python.import_module("pandas")
    let np = Python.import_module("numpy")
    let data = pd.read_csv("zeta_dataset.csv")
    var x_data = Tensor[DType.float32](np.array(data['s']).reshape(-1, 1))
    var y_data = Tensor[DType.float32](np.array(data['zeta(s)']).reshape(-1, 1))
    
    # Normalize data
    let x_mean = x_data.mean()
    let x_std = x_data.std()
    let y_mean = y_data.mean()
    let y_std = y_data.std()
    x_data = (x_data - x_mean) / x_std
    y_data = (y_data - y_mean) / y_std
    
    # Create and train model
    var model = NeuralNetwork(DynamicVector[Int]([1, 64, 64, 32, 1]))
    train_model(model, x_data, y_data, 0.01, 100)
    
    # Evaluate model
    let predictions = model.forward(x_data)
    let mse = mean_squared_error(predictions, y_data)
    print("Mean Squared Error:", mse)
    
    # Denormalize predictions for plotting
    let denorm_predictions = predictions * y_std + y_mean
    
    # Plot results
    let plt = Python.import_module("matplotlib.pyplot")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['s'], data['zeta(s)'], color='blue', label='Actual', alpha=0.5)
    plt.scatter(data['s'], denorm_predictions.numpy(), color='red', label='Predicted', alpha=0.5)
    plt.xlabel('s')
    plt.ylabel('zeta(s)')
    plt.title('Riemann Zeta Function: Actual vs Predicted')
    plt.legend()
    plt.savefig('riemann_zeta_plot_mojo.png')
    plt.close()
    
    let end_time = now()
    print("Execution time:", (end_time - start_time) / 1e9, "seconds")

main()