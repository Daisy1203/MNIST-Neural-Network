#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 08:59:30 2023

@author: daisy
"""
import gzip
import pickle
import numpy as np
from parameter_search import parameter_search
from neural_network import TwoLayerNeuralNetwork
import os 
os.chdir('/Users/daisy/Desktop')
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_data

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()

# 定义测试最佳模型的函数
def test_best_model(best_nn, test_images, test_labels, num_samples=10):
    # 加载保存的模型。
    best_nn.load_model('best_model.npz')
    
    # 预测测试图像的标签。
    predicted_labels = best_nn.predict(test_images)

    # 计算分类精度。
    accuracy = np.mean(predicted_labels == test_labels)

    print(f'测试精度: {accuracy}')

    # 显示部分手写数字对比图像。
    display_comparison_images(test_images, test_labels, predicted_labels, num_samples)

# 定义显示对比图像的函数
def display_comparison_images(test_images, test_labels, predicted_labels, num_samples):
    indices = np.random.choice(test_images.shape[0], num_samples, replace=False)
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

    for i, idx in enumerate(indices):
        axes[0, i].imshow(test_images[idx].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Label: {test_labels[idx]}')

        axes[1, i].imshow(test_images[idx].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Predicted: {predicted_labels[idx]}')

    plt.show()

if __name__ == "__main__":
    # 加载MNIST测试数据。
    _, _, _, _, test_images, test_labels = load_mnist_data()
    
    best_params = parameter_search()
    # 用与最佳模型相同的结构初始化一个神经网络。
    best_nn = TwoLayerNeuralNetwork(input_size=784, hidden_size=best_params[1], output_size=10, reg_strength=best_params[2])

    # 测试最佳模型。
    test_best_model(best_nn, test_images, test_labels)
