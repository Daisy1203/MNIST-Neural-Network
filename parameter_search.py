#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 08:55:20 2023

@author: daisy
"""

import os
print(os.getcwd())
os.chdir('/Users/daisy/Desktop')
import numpy as np
from data_loader import load_mnist_data
from neural_network import TwoLayerNeuralNetwork
import matplotlib.pyplot as plt


def plot_loss_accuracy(train_loss_history, val_loss_history, val_accuracy_history):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy History')
    
    plt.show()

def plot_weights(W1, W2):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(W1, cmap='gray', aspect='auto')
    plt.xlabel('Hidden Units')
    plt.ylabel('Input Features')
    plt.title('First Layer Weights')
    
    plt.subplot(1, 2, 2)
    plt.imshow(W2.T, cmap='gray', aspect='auto')
    plt.xlabel('Output Units')
    plt.ylabel('Hidden Units')
    plt.title('Second Layer Weights')
    
    plt.show()

def parameter_search():
    train_images, train_labels, val_images, val_labels, _, _ = load_mnist_data()

    best_val_acc = 0
    best_params = None
    best_nn = None

    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    hidden_sizes = [200, 300, 400, 500]
    reg_strengths = [0.0, 0.01, 0.05, 0.1, 0.5]

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                nn = TwoLayerNeuralNetwork(input_size=784, hidden_size=hs, output_size=10, reg_strength=reg)
                train_loss, val_loss, val_acc = nn.train(train_images, train_labels, val_images, val_labels, learning_rate=lr)
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_acc)
                current_val_acc = val_acc[-1]
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_params = (lr, hs, reg)
                    best_nn = nn

    print(f'Best parameters: Learning rate={best_params[0]}, Hidden size={best_params[1]}, Regularization strength={best_params[2]}')
    print(f'Best validation accuracy: {best_val_acc}')

    plot_loss_accuracy(train_loss_history[-1], val_loss_history[-1], val_accuracy_history[-1])
    plot_weights(best_nn.params['W1'], best_nn.params['W2'])
    best_nn.save_model('best_model.npz')
    return best_params
if __name__ == "__main__":
    parameter_search()
