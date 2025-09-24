# Binary Image Classifier with a Feedforward Neural Network (CPU & GPU)

A simple feedforward neural network designed for generic binary image classification (e.g., Class A vs. Class B) using C++, OpenCV, and optional CUDA acceleration for a significant performance boost.

---

## Features

- **Data Preprocessing:** Loads images from specified folders and preprocesses them into a consistent grayscale format with a resolution of 32x32 pixels.
- **Neural Network Architecture:** A fully connected, two-layer neural network.
- **Activation Function:** The Sigmoid function is used for both the hidden and output layers, scaling the output to a range of 0 to 1, which is ideal for binary classification.
- **Performance:** A default CPU-only version is provided, with an optional CUDA-enabled build for leveraging NVIDIA GPUs.
- **Persistence:** The trained network weights are saved to a CSV file for future use without retraining.

---

## Architectural Details

The network consists of a simple but effective architecture for handling small-scale image classification tasks:

- **Input Layer:** Comprises 1024 neurons, corresponding to the 32x32 pixels of the flattened grayscale image.
- **Hidden Layer:** Contains 64 neurons. This layer is responsible for learning and extracting complex, non-linear features from the input data.
- **Output Layer:** A single neuron that produces a final output between 0 and 1, representing the probability of the input image belonging to Class B. A threshold (e.g., 0.5) can be used to make the final binary decision.

---

## Mathematical Concepts

The core functionality of this neural network is built upon two fundamental processes: **Feedforward Propagation** (for making predictions) and **Backpropagation** (for learning from errors).

### 1. Feedforward Propagation (Prediction)

This is the process of passing input data through the network's layers to generate a final prediction.

**Hidden Layer Output:**

\[ H = \sigma(W_1 X + b_1) \]

**Output Layer Output:**

\[ O = \sigma(W_2 H + b_2) \]

Where:

- X is the input vector (1024x1).
- W_1 is the weight matrix (64x1024) for the hidden layer.
- b_1 is the bias vector (64x1) for the hidden layer.
- W_2 is the weight matrix (1x64) for the output layer.
- b_2 is the bias vector (1x1) for the output layer.
- \(\sigma(z) = \frac{1}{1 + e^{-z}}\) is the Sigmoid Activation Function.

### 2. Backpropagation (Training)

This is the algorithm that allows the network to learn. It works by calculating the error of the prediction and then propagating that error backward through the network to update the weights and biases.

**Loss Function:**

The network's performance is measured using a Loss Function that quantifies the error between the predicted output (\(\hat{y}\)) and the true value (y). We use the Mean Squared Error (MSE).

\[ MSE = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

**Gradient Calculation:**

The core of backpropagation involves using the Chain Rule from calculus to compute the gradient of the loss function with respect to each weight and bias. This gradient indicates the direction and magnitude in which each parameter should be adjusted to minimize the error.

**Gradient Descent:**

Finally, Gradient Descent is used to update the weights and biases based on the calculated gradients.

\[ W_{new} = W_{old} - \alpha \frac{\partial E}{\partial W} \]

Where:

- \(\alpha\) is the Learning Rate, a hyperparameter that controls the step size of each update.
- \(\frac{\partial E}{\partial W}\) is the gradient of the error (E) with respect to the weight (W).

---

## Folder Structure

```
dataset/
├── water/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── other/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

> Folder names must match exactly the categories defined in the code (`water` and `other`).

---

## Build Instructions (CPU Only)

```bash
mkdir build
cd build
cmake ..
make
./nn_cpu /path/to/dataset
```

---

## Enabling CUDA (Optional)

```bash
cmake .. -DENABLE_CUDA=ON
make
./nn_gpu /path/to/dataset
```

---

## Network Save

The trained network weights are saved to `Network.csv` in the same directory. You can reload them later for inference.

---

## Usage

```bash
# CPU version
./nn_cpu /home/username/dataset

# GPU version
./nn_gpu /home/username/dataset
```

---

## License

MIT License

