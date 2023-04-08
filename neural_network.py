#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import numpy as np


# In[3]:


import numpy as np

class TwoLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_strength=0.0):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.reg_strength = reg_strength

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        a1 = X.dot(W1) + b1
        z1 = self.relu(a1)
        scores = z1.dot(W2) + b2
        
        return a1, z1, scores

    def loss(self, X, y):
        a1, z1, scores = self.forward(X)
        probs = self.softmax(scores)
        num_samples = X.shape[0]
        correct_logprobs = -np.log(probs[range(num_samples), y])
        data_loss = np.sum(correct_logprobs) / num_samples
        reg_loss = 0.5 * self.reg_strength * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        total_loss = data_loss + reg_loss
        return total_loss

    def predict(self, X):
        _, _, scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def backprop(self, X, y):
        num_samples = X.shape[0]
        a1, z1, scores = self.forward(X)
        probs = self.softmax(scores)

        grads = {}
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples

        grads['W2'] = np.dot(z1.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)

        dhidden = np.dot(dscores, self.params['W2'].T)
        dhidden[z1 <= 0] = 0

        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

        grads['W1'] += self.reg_strength * self.params['W1']
        grads['W2'] += self.reg_strength * self.params['W2']

        return grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, num_iters=2000, batch_size=100):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        train_loss_history = []
        val_loss_history = []
        val_accuracy_history = []

        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices.astype(np.int64)]

            grads = self.backprop(X_batch, y_batch)

            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            if it % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay
                train_loss = self.loss(X_batch, y_batch)
                val_loss = self.loss(X_val, y_val)
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()

                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_acc)

                print(f'Epoch {it // iterations_per_epoch}: Train acc={train_acc}, Val acc={val_acc}')

        return train_loss_history, val_loss_history, val_accuracy_history

    def save_model(self, file_name):
        np.savez(file_name, **self.params)

    def load_model(self, file_name):
        loaded_params = np.load(file_name)
        for param in self.params:
            self.params[param] = loaded_params[param]


# In[ ]:




