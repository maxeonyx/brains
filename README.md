# Brains
NOTE: this project is very early and was spawned from my own architecture research. contributions, opinions and issues are very welcome!

an Artificial Neural Network framework built on Tensorflow-rs bindings for creating architectures similar to keras but also with direct integration for custom layers in low level Tensorflow. Includes native checkpointing, inference, batch trainning and iterative trainning. See the unittests to get an idea of how things are called until documentation is created. Also ensure to enable the GPU flag for tensorflow if you want to offload computation.

Currently all inputs and outputs are represented as flattend 1D rust Vecs.

# TODO: 

add more generic typing, currently f32 for parameters and u64 for architecture size is standard.

add more layers

add a form of N dimensional convolution that maps to the 1D input vectors and hidden layers 

formalize documentation for layers (unscaled architectures) as seen below.

Architectures
# NormNet
Research in Neural Networks without bias and gradient based connection dropout by utilizing per node normalization.

It is hypothesized that neural networks struggle with transfer learning due to biases limiting their exploration of other local minimas.
it is with this hypothesis that norm net came to exist, attempting to eliminate biases while also allowing networks to perform gradient based dropout.
both gradient based dropout and normalized signals are thought to allow a network to recover previous minima and explore better given gradient descent 
without having to checkpoint the model. This is due to a networks parameters being more continuous without bias' subtree partitions of the network 
(nodes between two layers have certain signal "sweet spots" of expressivity for all non-linear activation functions).
Sparsity of the weights due to connection dropout also has a bloom filter hashing effect (for lack of a better analogy) that leaves more parameters to represent 
future domains outside of the initial trainning set.
Because generalization is thought by many to be closely linked to transferability of a model (its ability to solve diverse problem sets) it is the aim 
of this research to find networks that are able to generalize greater by being able to solve distinct problems with the same parameter set.


Please let me know what you think: ward.joshua92@yahoo.com
