# FashionMNIST
My attempt in implementing a network for the dataset Fashion MNIST which can be found at:
    https://github.com/zalandoresearch/fashion-mnist
    
My model:

    - 2 Layered CNN + Max Pooling + Batch Normalisation (Using ReLU activation)
    - 2 Fully Connected Layers
    
    Random Horizontal Flips of the Fashion MNIST images is included as preprocessing technique.
    
    Optimizer Used - SGD + Nesterov Momentum
    
Final Accuracy - 93.29%
