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
