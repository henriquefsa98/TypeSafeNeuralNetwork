# TypeSafeNeuralNetwork

A Type Safe neural network developted using Haskell, with it's Layers dimensions typed, and Activations functions inserted on the structure!

This is a Graduation Project in Computing, made by Henrique Fantato, advised by Professors PhD. Emilio Francesquini and PhD Fabricio Olivetti.

This project motivation is developing a functional programming code in Haskell, applying Type-Driven development concepts, with the objective of 
creating a Type-Safe Neural Network, utilizing the HMatrix lib to provide and optimize Linear algebra calculations and Typed Structure. 
All of this to avoid constructions errors, that lead to calculations errors and could break all the execution of the neural network.

This repository contains a library for building and training typed neural networks. It provides functionality for defining network architectures, training the networks using backpropagation, and applying the trained networks for inference tasks.


## Features

- Definition of network architecture using type-level GADT's 
- Support for various activation functions, including linear, logistic, tangent, ReLU, Leaky ReLU, and ELU
- Serialization and deserialization of network structures
- Random initialization of network weights
- Training of networks using backpropagation and stochastic gradient descent
- Support for binary output and softmax output filters
- Evaluation of trained networks on input data


## Installation

To use this library, you need to have GHC (Glasgow Haskell Compiler) and Cabal installed on your system. Follow the steps below to install the library:

1. Clone this repository: git clone <repository_url>

2. Navigate to the project directory: cd TypeSafeNeuralNetwork

3. Build the project using Stack: Stack build

4. Run it!: Stack run
 



## Usage

To use the neural network implemented in this code, follow these steps:

1. Define the architecture of your neural network using the provided types, a list of Activations and the randomNet function. For example:

        networkExample :: Network 2 '[3, 2] 1 <- randomNet [Logistic, ReLu, Linear]


2. Train it by using the netTrain function:

        (_, _, trainedNetwork, outputS) <- netTrain initialNet learningRate numIterations samples  (inputSize, outputSize)    


3. Check the accuracy of the trained network by using checkAccuracy function: 

        putStrLn $ show (checkAccuracy outputS)

4. Print the output on stdin:

        putStrLn $ renderOutput outputS

5. Show the current structure of the trained network:

        print netTrained




## Development


 This Neural network implementation was developed with type safety in mind. First, a version without any type-safe was constructed, to serve as 
guidelines for the type-safe implementation. 
 
 The major problem with a type-unsafe implementation is that the user can define a Network Structure with incompatible layers sizes, breaking all the execution:



        data Weights = W { wBiases :: !(Vector Double), wNodes  :: !(Matrix Double)}  

        data Network :: * where
                    O     :: !Weights
                          -> Network
                    (:&~) :: !Weights
                          -> !Network
                          -> Network
        infixr 5 :&~
        

        (:&~) (W 2 5) (O 2 1)             Incompatible Weights! 


 And this kind of problem would only thrown an error at execution time, when the calculations between Vectors and Matrix breake the code from incompatible sizes.
 So an user is able do declare a non functional network, and it will only show an error at execution time. Futhermore, you can't choose an activation function for
 each layer of the network, limiting possible architectures and restricting the problems that the Neural network could solve. 



 To solve this problem, we can bring the size of the layers to the type level, by using typed vectors and matrix in the network definition and constructors and 
 inserting a activation data for each layer:

        data Weights i o = W { wBiases  :: !(SA.R o), wNodes   :: !(SA.L o i)}
            
        data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double

        data Network :: Nat -> [Nat] -> Nat -> Type where
        O     :: !(Weights i o) -> Activation
              -> Network i '[] o
        (:&~) :: (KnownNat h) => Weights i h -> Activation
              -> !(Network h hs o)
              -> Network i (h ': hs)  o
        infixr 5 :&~


 Now the network input, hidden and output sizes are all typed, and the constructor guarantee that all conected layers should have compatible sizes, and we define 
 each layer activation with the constructor.

 All other functions and definitions should guarantee all the sizes constraints to be implemented, so we get correctness from the types at the development of all functions.




## Future implementations


 - To get to a even more type-driven neural network, as future development, bringing the Activations to the type level would increase the correctness of code by requiring that all 
 activations to be compatible with the Weights sizes and that each layer must have a activation associated with it. 
 
 - Implement a existential wrapper for Network, to be able to create, save and load networks without the need of specifying the sizes at type level.

 - Implement the existential equivalent functions to be able to run the Existential Network.





## Acknowledgements


 This code is based on the work of Justin Le and his Type-Safe Neural Network, his work can be found at:
 https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html
 https://github.com/mstksg/inCode/tree/master


 Special thanks to both professors PhD Emilio Francesquini and PhD Fabricio Olivetti for all the guidance, help (and Debugging :D) 
 through the development of the project!

 https://www.ufabc.edu.br/ensino/docentes/emilio-de-camargo-francesquini
 https://www.ufabc.edu.br/ensino/docentes/fabricio-olivetti-de-franca




## References


- BRADY, E. Type-driven development with Idris. [S.l.]: Simon and Schuster, 2017. Citado
  2 vezes nas páginas 2 e 11.

- EISENBERG, R. A.; WEIRICH, S. Dependently typed programming with singletons.
  ACM SIGPLAN Notices, ACM New York, NY, USA, v. 47, n. 12, p. 117–130, 2012.
  Citado na página 4.

- JOHNSON, J.; PICTON, P. How to train a neural network:an introduction to the new
  computational paradigm. Complexity, v. 1, n. 6, p. 13–28, 1996. Citado na página 4.

- MAGUIRE, S. Thinking with Types, Type-Level Programming in Haskell. [S.l.]: LeanPub,
  2019. Citado na página 3.

- PICTON, P. What is a neural network? In: Introduction to Neural Networks. [S.l.]:
  Springer, 1994. p. 1–12. Citado na página 4.

- VOLDER, K. D. Type-oriented logic meta programming. Tese (Doutorado) — Citeseer,
  1998. Citado na página 2.

- WANG, S.-C. Artificial neural network. In: Interdisciplinary computing in java
  programming. [S.l.]: Springer, 2003. p. 81–100. Citado na página 4.

