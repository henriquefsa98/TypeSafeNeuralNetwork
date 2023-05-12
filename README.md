# TypeSafeNeuralNetwork

A Type Safe neural network developed using Haskell, with it's Layers dimensions typed, and Activation functions inserted on the structure!

This is a Graduation Project in Computing, made by Henrique Fantato, advised by Professors PhD. Emilio Francesquini and PhD. Fabricio Olivetti, 
at Federal University of ABC, Brazil.

This project's motivation is developing a functional programming code in Haskell, applying Type-Driven development concepts, with the objective of 
creating a Type-Safe Neural Network, utilizing the HMatrix lib to provide and optimize Linear algebra calculations and Typed Structure. 
All of this to avoid constructions errors, that lead to calculation errors and could break all the execution of the neural network.


This repository contains a library for building and training typed neural networks. It provides functionalities for defining network architectures, 
training the networks using backpropagation, and applying the trained networks for inference tasks.


## Features

- Definition of network architecture using type-level GADT's 
- Support for various activation functions, including Linear, Logistic, Tangent, ReLU, Leaky ReLU, and ELU
- Serialization and deserialization of network structures
- Random initialization of network weights
- Training of networks using backpropagation and stochastic gradient descent
- Support for binary output and softmax output filters
- Evaluation of trained networks on input data


## Installation

To use this library, you need to have GHC (Glasgow Haskell Compiler) and Cabal installed on your system. Follow the steps below to install the library:

1. Clone this repository: `git clone https://github.com/henriquefsa98/TypeSafeNeuralNetwork.git`

2. Navigate to the project directory: `cd TypeSafeNeuralNetwork`

3. Build the project using Stack: `Stack build`

4. Run it!: `Stack run`
 



## Usage

To use the neural network implemented in this code, follow these steps:

1. Define the architecture of your neural network using the provided types, a list of Activations and the randomNet function. For example:

 ```haskell
 networkExample :: Network 2 '[3, 2] 1 <- randomNet [Logistic, ReLu, Linear]
 ```


2. Train it by using the netTrain function:

 ```haskell
  (_, _, trainedNetwork, outputS) <- netTrain initialNet learningRate numIterations samples  (inputSize, outputSize)
 ```


3. Check the accuracy of the trained network by using checkAccuracy function: 

 ```haskell
 putStrLn $ show (checkAccuracy outputS)
 ```

4. Print the output on stdin:

 ```haskell
 putStrLn $ renderOutput outputS
 ```

5. Show the current structure of the trained network:

 ```haskell
 print netTrained
 ```




## Development


 This Neural network implementation was developed with type safety in mind. First, a version without any type-safe was constructed, to serve as 
guidelines for the type-safe implementation. 
 
 The major problem with a type-unsafe implementation is that the user can define a Network Structure with incompatible layers sizes, breaking all 
 the execution:



 ```haskell
 data Weights = W { wBiases :: !(Vector Double), wNodes  :: !(Matrix Double)}  
 data Network :: * where
             O     :: !Weights
                   -> Network
             (:&~) :: !Weights
                   -> !Network
                   -> Network
 infixr 5 :&~
 
 net = (:&~) (W 2 5) (O 2 1)             --Incompatible Weights!
 ```


  And this kind of problem would only throw an error at execution time, when the calculations between Vectors and Matrices break the code from incompatible sizes.
 So an user is able to declare a non functional network, and it will only show an error at execution time. Futhermore, you can't choose an activation function for
 each layer of the network, limiting possible architectures and restricting the problems that the Neural network could solve. 



 To solve this problem, we can bring the size of the layers to the type level, by using typed vectors and matrices in the network definition and constructors and 
 inserting a activation data for each layer:

 ```haskell
 data Weights i o = W { wBiases  :: !(SA.R o), wNodes   :: !(SA.L o i)}
     
 data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double
 data Network :: Nat -> [Nat] -> Nat -> Type where
 O     :: !(Weights i o) -> Activation
       -> Network i '[] o
 (:&~) :: (KnownNat h) => Weights i h -> Activation
       -> !(Network h hs o)
       -> Network i (h ': hs)  o
 infixr 5 :&~
 ```


 Now the network input, hidden layers and output sizes are all typed, and the constructor guarantee that all conected layers should have compatible sizes, and 
 we define each layer activation with the constructor.

 All other functions and definitions should guarantee that all the sizes constraints are respected to be implemented, so we get correctness from the types at 
 the development of all functions.



### Differences between Justin Le implementation


  The fundamental difference between this implementation and Justin Le's one is the inclusion of Activation in the Network construtors themselves. 
 Le's implementation used a static logistic activation, defined at the training function, so you can't choose neither another function or choose different 
 functions for each layer. This choice prevents the network from being able to resolve a lot of problems, and make the learning process even harder by not 
 being able to modify the network activations.

  By choosing to enable the insertion of Activation function in the constructors of the Network, the implementation of this repository need to refactor all 
 of the auxiliary functions and definitions of the Network and its Weights, in a manner that all code developed needs to account for the compatibility of 
 Nat sizes, and the existent activation function of each layer.

  For example, those are the original definition of the Network structure and constructors:

 ```haskell
 data Network :: Nat -> [Nat] -> Nat -> * where
                 O     :: !(Weights i o)
                         -> Network i '[] o
                 (:&~) :: KnownNat h
                  => !(Weights i h)
                 -> !(Network h hs o)
                 -> Network i (h ': hs) o
 infixr 5 :&~
 ```

  But now, those are the new Network structure and constructors:


 ```haskell
 data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double deriving (Show, Generic, Eq)

 data Network :: Nat -> [Nat] -> Nat -> Type where
                 O     :: !(Weights i o) -> Activation
                 -> Network i '[] o
                 (:&~) :: (KnownNat h) => Weights i h -> Activation
                 -> !(Network h hs o)
                 -> Network i (h ': hs)  o
 infixr 5 :&~
 ```
  
  Now, for each layer of the Network, we need to provide a Activation, making the net so more adaptive and powerfull. For each 
 Activation, we need to provide both the function and its derivative, like ReLu activation below:

 ```haskell


 -- ReLu activation and its derivative 
 relu ::  KnownNat i =>  SA.R i -> SA.R i
 relu x = SA.vecR $ VecSized.map (max 0) $ SA.rVec x


 relu' :: KnownNat i => SA.R i -> SA.R i
 relu' x = SA.vecR $ VecSized.map (\y -> if y > 0 then 1 else 0) $ SA.rVec x



 getFunctions :: (KnownNat i) => Activation -> (SA.R i -> SA.R i, SA.R i -> SA.R i)
 getFunctions f = case f of
                         ReLu      -> (relu, relu')

 ```

  All activation functions and their derivatives need to consider the Nat sizes compatibility at type level to be able to run 
 for all Networks, so that no activation should be able to modify the Vector sizes of any Network constructed, avoiding another 
 kind of problems.

  Le's implementation does not provide a typed training function, so this repository implement it, using Backpropagation and 
 stochastic gradient descent, for adjusting all of Network Weights according to prediction errors.

  Another new capability implementation is the definition of Filters, to be able to run a Network with filters on it's output. This 
 bring another level of adaptability for the Network, as some problems require a binary output, but a binary output in the trainig 
 phase is not able to backpropagate the current layer predictions errors to its previous layers, breaking all the learning proccess.


  Those are some NetFilter implementations available at this repositoty:

 ```haskell
 data NetFilter = BinaryOutput | SoftMax deriving Show

 -- Filter that makes the Network only output 0 or 1, a binary output
 binaryOutput :: (KnownNat i) => SA.R i -> SA.R i
 binaryOutput = SA.dvmap (\y -> if y > 0.5 then 1 else 0)

 -- Filter that makes the Network output a probability distribution, good for classifications
 softmaxOut :: (KnownNat i) => SA.R i -> SA.R i
 softmaxOut x = SA.vecR $ VecSized.map (/ total) $ VecSized.map exp $ SA.rVec x
               where
                   total =  VecSized.foldr (+) 0 $ VecSized.map exp $ SA.rVec x


 getFilter :: (KnownNat i) => NetFilter -> (SA.R i-> SA.R i)
 getFilter f = case f of
                 BinaryOutput   ->   binaryOutput
                 SoftMax        ->   softmaxOut


 -- Function to run the Network over a list of Samples, and applaying a NetFilter
 runNetFiltered :: (KnownNat i, KnownNat o) => Network i hs o -> [[Double]] -> (Int, Int) -> NetFilter -> [(SA.R i, SA.R o, SA.R o)]
 runNetFiltered net samples (inputD, outputD) filterF = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, nnFilter (runNet net ( SA.vector (take inputD x)))) | x <- samples ]

                                                             where
                                                             
                                                                 nnFilter = getFilter filterF

 ```

  There are anothers modifications and new implementations on this repository, those above are just for illustrate the differences between this repository and 
 Justin Le's one. 



## Future implementations


 - To get to a even more type-driven neural network, as future development, bringing the Activations to the type level would increase the correctness of code by requiring that all 
 activations to be compatible with the Weights sizes and that each layer must have a activation associated with it. At the current stage of development, the Activations are a list 
 argument that isn't required to be exactly the same length as the Network, so to avoid not having a Activation for a layer, the function 'getAct' was implemented to work around this flaw:

 ```haskell
 getAct :: [Activation] -> Activation
 getAct (a:_)  = a
 getAct []     = Linear
 ```
 
 - Implement a existential wrapper for Network, to be able to create, save and load networks without the need of specifying the sizes at type level, by creating Binary instances, 
 put/get methods and constructors.

 - Implement the existential equivalent functions to be able to run the Existential Network.





## Acknowledgements


 This code is based on the work of Justin Le and his Type-Safe Neural Network, his work can be found at:
 
 https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html
 https://github.com/mstksg/inCode/tree/master


 Special thanks to both professors PhD. Emilio Francesquini and PhD. Fabricio Olivetti for all the guidance and help (and Debugging :D) 
 through the development of the project!

 https://www.ufabc.edu.br/ensino/docentes/emilio-de-camargo-francesquini
 https://www.ufabc.edu.br/ensino/docentes/fabricio-olivetti-de-franca




## References


- BRADY, E. Type-driven development with Idris. [S.l.]: Simon and Schuster, 2017. Cited 
on pages 2 and 11.

- EISENBERG, R. A.; WEIRICH, S. Dependently typed programming with singletons.
  ACM SIGPLAN Notices, ACM New York, NY, USA, v. 47, n. 12, p. 117–130, 2012.
  Cited on page 4.

- JOHNSON, J.; PICTON, P. How to train a neural network:an introduction to the new
  computational paradigm. Complexity, v. 1, n. 6, p. 13–28, 1996. Cited on page 4.

- MAGUIRE, S. Thinking with Types, Type-Level Programming in Haskell. [S.l.]: LeanPub,
  2019. Cited on page 3..

- PICTON, P. What is a neural network? In: Introduction to Neural Networks. [S.l.]:
  Springer, 1994. p. 1–12. Cited on page 4.

- VOLDER, K. D. Type-oriented logic meta programming. Tese (Doutorado) — Citeseer,
  1998. Cited on page 2.

- WANG, S.-C. Artificial neural network. In: Interdisciplinary computing in java
  programming. [S.l.]: Springer, 2003. p. 81–100. Cited on page 4.

