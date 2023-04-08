# MNIST-Neural-Network
# Handwritten Digit Recognition with Two-Layer Neural Network

This repository contains the implementation of a two-layer neural network for handwritten digit recognition using the MNIST dataset.

## Prerequisites

- Python 3.x
- NumPy
- matplotlib
- python-mnist

## Getting Started

1. Clone this repository.
git clone https://github.com/yourusername/your-repository-name.git
cd MNIST-Neural-Network

2. Download the MNIST dataset and place it in the project folder.

3. Install the required packages.
pip install numpy matplotlib python-mnist
├── data_loader.py # Loads and preprocesses the MNIST dataset
├── neural_network.py # Defines the TwoLayerNeuralNetwork class
├── parameter_search.py # Performs hyperparameter search and model training
└── test.py # Tests the trained model and displays comparison images


## Training

To train the model and perform hyperparameter search, run the following command:
python parameter_search.py

This script will find the best hyperparameters, train the model, save the best model as 'best_model.npz', and plot the loss and accuracy curves.

## Testing

To test the trained model on the MNIST test dataset, run the following command:


This script will load the trained model, evaluate it on the test dataset, and display some comparison images of the true labels and predicted labels.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License.
