{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS_GHC -Wno-missing-export-lists #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PolyKinds #-}



module Main where



import Control.Monad ()
import Control.Monad.Random

import GHC.Float ()
import Data.Maybe

import System.Environment
import Text.Read
import Data.Kind (Type)
import GHC.Generics

import GHC.TypeLits.Singletons
import Numeric.LinearAlgebra as NonStatic


import qualified Numeric.LinearAlgebra.Static as SA

import Data.Singletons
--import Prelude.Singletons
import Data.List.Singletons
import qualified Numeric.LinearAlgebra.Static.Vector as SA
import Data.Vector.Storable.Sized as VecSized (toList, map, foldr)
import Data.Binary as BinLib
import qualified Data.ByteString.Lazy as BSL
import Data.List (foldl')
import System.Random.Shuffle (shuffle')
import GHC.Natural (Natural)







-- Definition of Weights, a representation of a Layer of Neurons in the Neural Network

data Weights i o = W { wBiases  :: !(SA.R o)
                     , wNodes   :: !(SA.L o i)
                      }
                  deriving (Show, Generic)


instance (KnownNat i, KnownNat o) => Eq (Weights i o) where
  (==) :: Weights i o -> Weights i o -> Bool
  (==) (W b n) (W b2 n2) = (SA.rVec b == SA.rVec b2) && (SA.lVec n == SA.lVec n2)

instance (KnownNat i, KnownNat o) => Binary (Weights i o)


-- Definition of Data Activation, to be able to insert activation functions within which layer of the Network

data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double deriving (Show, Generic, Eq)

instance Binary Activation


-- Auxiliar function to get Activation function and it's derivative, pattern-matching with Activation constructors
getFunctions :: (KnownNat i) => Activation -> (SA.R i -> SA.R i, SA.R i -> SA.R i)
getFunctions f = case f of
                  Linear      -> (linear, linear')
                  Logistic    -> (logistic, logistic')
                  Tangent     -> (tangent, tangent')
                  ReLu        -> (relu, relu')
                  LeakyReLu   -> (lrelu, lrelu')
                  ELU a       -> (elu a, elu' a)


data Network :: Nat -> [Nat] -> Nat -> Type where
    O     :: !(Weights i o) -> Activation
          -> Network i '[] o
    (:&~) :: (KnownNat h) => Weights i h -> Activation
          -> !(Network h hs o)
          -> Network i (h ': hs)  o

infixr 5 :&~


-- Show instance definition to visualize the Network
instance (KnownNat i, KnownNat o) => Show (Network i hs o) where
  show :: Network i hs o -> String
  show (O a f)          =  "Output neurons: "        ++ show (wNodes a) ++ ", Weights: " ++ show (wBiases a) ++ ", Activation Function: " ++ show f
  show ((:&~) a f b)    =  "Input/Layer neurons: "   ++ show (wNodes a) ++ ", Weights: " ++ show (wBiases a) ++ ", Activation Function: " ++ show f ++ "\n" ++ show b


instance (KnownNat i, KnownNat o) => Eq (Network i hs o) where
  (==) (O w f)       (O w2 f2)        = w == w2 && f == f2
  (==) ((:&~) w f n) ((:&~) w2 f2 n2) = w == w2 && f == f2 && n == n2


-- Definition of existencial type for Network:

data OpaqueNet :: Nat -> Nat -> Type where
  ONet :: Network i hs o -> OpaqueNet i o

-- Show instance definition to visualize the Network
instance (KnownNat i, KnownNat o) => Show (OpaqueNet i o) where
  show :: OpaqueNet i o -> String
  show (ONet x) = show x

-- Definition of instance to serialize a Network and the put/get functions:


putNet :: (KnownNat i, KnownNat o)
       => Network i hs o
       -> Put
putNet = \case
            O     w f   -> put (w, f)
            (:&~) w f n -> put (w, f) *> putNet n

getNet :: forall i hs o. (KnownNat i, KnownNat o, SingI hs) => Get (Network i hs o)
getNet = go sing
  where
    go :: forall j js. (KnownNat j) => Sing js -> Get (Network j js o)
    go sizes = case sizes of
                  SNil -> do
                            (weights, activation) <- BinLib.get
                            return (O weights activation)

                  (SCons SNat ss) -> do
                                        (weights, activation) <- BinLib.get
                                        (:&~) weights activation <$> go ss



instance (KnownNat i, SingI hs, KnownNat o) => Binary (Network i hs o) where
  put = putNet
  get = getNet



-- Functions definitions to serialize and desserialize the Network

-- Serialize a Network to a ByteString
serializeNetwork :: (KnownNat i, SingI hs, KnownNat o) => Network i hs o -> BSL.ByteString
serializeNetwork = encode

-- Deserialize a Network from a ByteString
deserializeNetwork :: (KnownNat i, SingI hs, KnownNat o) => BSL.ByteString -> Network i hs o
deserializeNetwork = decode



-- Auxiliar definitions of activation functions and it's derivatives

-- Linear activation and its derivative 
linear :: SA.R i -> SA.R i
linear x = x

linear' :: KnownNat i => SA.R i -> SA.R i
linear' x =  SA.vecR $ VecSized.map (const 1) $ SA.rVec x


-- Logistic activation and its derivative 
logistic :: KnownNat i => SA.R i -> SA.R i
logistic x = SA.vecR $ VecSized.map (\y -> 1 / (1 + exp (-y))) $ SA.rVec x

logistic' :: KnownNat i => SA.R i -> SA.R i
logistic' x = logix * (1 - logix)
  where
    logix = logistic x



-- Tangent activation and its derivative 
tangent :: Floating a => a -> a
tangent x = (exp x - exp (-x)) / (exp x + exp (-x))

tangent' :: Floating a => a -> a
tangent' x = 1 + tangent x * tangent x


-- ReLu activation and its derivative 
relu ::  KnownNat i =>  SA.R i -> SA.R i
relu x = SA.vecR $ VecSized.map (max 0) $ SA.rVec x


relu' :: KnownNat i => SA.R i -> SA.R i
relu' x = SA.vecR $ VecSized.map (\y -> if y > 0 then 1 else 0) $ SA.rVec x


-- Leaky ReLu activation and its derivative 
lrelu :: KnownNat i => SA.R i -> SA.R i
lrelu y = SA.vecR $ VecSized.map (\x -> max x (0.01*x)) $ SA.rVec y

lrelu' :: KnownNat i => SA.R i -> SA.R i
lrelu' y = SA.vecR $ VecSized.map (\x -> if x > 0 then 1 else 0.01) $ SA.rVec y



-- ELU activation and its derivative 
elu :: (KnownNat i) => Double -> SA.R i -> SA.R i
elu a y = SA.vecR $ VecSized.map (\x -> if x >= 0 then x else a * (exp x - 1)) $ SA.rVec  y


elu' :: (KnownNat n) => Double -> SA.R n -> SA.R n
elu' a y = SA.vecR $ VecSized.map (\x -> if x >= 0 then 1 else a + a * (exp x - 1)) $ SA.rVec y





-- Definiton of NetFilter, class of filters to the output of the neural network

data NetFilter = BinaryOutput | SoftMax deriving Show


getFilter :: (KnownNat i) => NetFilter -> (SA.R i-> SA.R i)
getFilter f = case f of

                BinaryOutput   ->   binaryOutput
                SoftMax        ->   softmaxOut


-- Auxiliar definitions to  Filters, implementation of the filters themselves


-- Filter that makes the Network only output 0 or 1, a binary output
binaryOutput :: (KnownNat i) => SA.R i -> SA.R i
binaryOutput = SA.dvmap (\y -> if y > 0.5 then 1 else 0)


-- Filter that makes the Network output a probability distribution, good for classifications
softmaxOut :: (KnownNat i) => SA.R i -> SA.R i
softmaxOut x = SA.vecR $ VecSized.map (/ total) $ VecSized.map exp $ SA.rVec x
              where
                  total =  VecSized.foldr (+) 0 $ VecSized.map exp $ SA.rVec x



-- Definitions of functions to run the network itself


-- Function to run a layer (Weights) of the Network
runLayer :: (KnownNat i, KnownNat o) => Weights i o -> SA.R i -> SA.R o
runLayer (W wB wN) v = wB + (wN SA.#> v)

-- Function to run a Network
runNet :: (KnownNat i, KnownNat o) => Network i hs o -> SA.R i -> SA.R o
runNet = \case
   O w f -> \(!v)  ->          let (function, _) = getFunctions f
                                in function (runLayer w v)

   ((:&~) w f n') -> \(!v) -> let
                                  (function, _) = getFunctions f
                                  v' = function (runLayer w v)
                                in  runNet n' v'



-- Definitions of functions to run OpaqueNet:

runOpaqueNet :: (KnownNat i, KnownNat o)
             => OpaqueNet i o
             -> SA.R i
             -> SA.R o
runOpaqueNet (ONet n) = runNet n

numHiddens :: OpaqueNet i o -> Int
numHiddens (ONet n) = go n
  where
    go :: Network i hs o -> Int
    go = \case
        O _ _        -> 0
        (:&~) _ _ n' -> 1 + go n'


-- Definitions of functions to generate a random network


-- Generate randoms Weights
randomWeights :: (MonadRandom m, KnownNat i, KnownNat o) => m (Weights i o)
randomWeights = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wB = SA.randomVector  seed1 Uniform * 2 - 1
        wN = SA.uniformSample seed2 (-1) 1
    return $ W wB wN


-- Auxiliar function to get a Activation from the [Activation], safetly returning Linear activations for empty lists
getAct :: [Activation] -> Activation
getAct (a:_)  = a
getAct []     = Linear


{- 
-- Generate a Network with random Weights, based on input, hidden layers and output types, and a list of Activations
randomNet :: forall m i hs o. (MonadRandom m, KnownNat i, SingI hs, KnownNat o)
          => [Activation]
          -> m (Network i hs o)
randomNet  hiddenActivations = go hiddenActivations sing
  where
    go :: forall h  hs'. (KnownNat h)
       => [Activation]
       ->  Sing hs'
       -> m (Network h  hs' o)

    go actL sizes  = case sizes of
                        SNil           -> O     <$> randomWeights <*> pure (getAct actL)
                        SCons SNat ss  -> (:&~) <$> randomWeights <*> pure (getAct actL) <*> go (tail actL) ss
-}


randomNet' :: forall m i hs o. (MonadRandom m, KnownNat i, KnownNat o)
           => [Activation] -> Sing hs -> m (Network i hs o)
randomNet' actL = \case
    SNil            ->     O <$> randomWeights <*> pure (getAct actL)
    SNat `SCons` ss -> (:&~) <$> randomWeights <*> pure (getAct actL) <*> randomNet' (tail actL) ss

randomNet :: forall m i hs o. (MonadRandom m, KnownNat i, SingI hs, KnownNat o)
          => [Activation] -> m (Network i hs o)
randomNet actL = randomNet' actL sing





-- Definitions of functions to generate a random Opaque Network


randomONet :: (MonadRandom m, KnownNat i, KnownNat o)
              => [Natural] -> [Activation]
              -> m (OpaqueNet i o)
randomONet hs fs = case toSing hs of
                        SomeSing ss-> ONet <$> randomNet' fs ss


-- Training function, train the network for just one iteration on one sample

train :: forall i hs o. (KnownNat i, KnownNat o)
      => Double           -- ^ learning rate
      -> SA.R i           -- ^ input vector
      -> SA.R o           -- ^ target vector
      -> Network i hs o   -- ^ network to train
      -> Network i hs o
train rate x0 target = fst . go x0
  where
    go :: forall j js. KnownNat j
        => SA.R j    -- ^ input vector
        -> Network j js o          -- ^ network to train
        -> (Network j js o, SA.R j)
    -- handle the output layer
    go !x (O w@(W wB wN) f)
        = let y    = runLayer w x
              (function, derivative) = getFunctions f
              o    = function y
              -- the gradient (how much y affects the error)
              dEdy = derivative y * (o - target)

              -- new bias weights and node weights
              wB'  = wB - SA.konst rate * dEdy
              wN'  = wN - SA.konst rate * SA.outer dEdy x
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN SA.#> dEdy
          in  (O w' f, dWs)

    -- handle the inner layers
    go !x ((:&~ ) w@(W wB wN) f n)
        = let y          = runLayer w x
              (function, derivative) = getFunctions f
              o          = function y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = derivative y * dWs'

              -- new bias weights and node weights
              wB'  = wB - SA.konst rate * dEdy
              wN'  = wN - SA.konst rate * SA.outer dEdy  x
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN SA.#> dEdy
          in  ((:&~) w' f n', dWs)




-- Auxiliar functions to apply to samples
lastN :: Int -> [a] -> [a]
lastN n xs = drop (length xs - n) xs


-- Function to train the Network, based on a learning rate, number of iterations, a list of samples and the samples dimensions
netTrain :: (MonadRandom m, MonadIO m, KnownNat i, KnownNat o, SingI hs) =>  Network i hs o -> Double -> Int -> [[Double]] -> (Int, Int) -> m (Network i hs o, [(SA.R i, SA.R 0, SA.R o)], Network i hs o, [(SA.R i, SA.R o, SA.R o)])
netTrain initnet learningrate nruns samples (inputD, outputD) = do

    let inps = Prelude.map (SA.vector . take inputD) samples
    let outs = Prelude.map (SA.vector . lastN outputD) samples

    gen <- newStdGen

    let trained = trainNTimes initnet (inps, outs) nruns
          where
            trainNTimes :: (KnownNat i, SingI hs, KnownNat o) => Network i hs o -> ([SA.R i], [SA.R o]) -> Int -> Network i hs o
            trainNTimes net (i, o) n2
                | n2 <= 0 = net
                | otherwise = trainNTimes (foldl' trainEach net (zip i o)) shuffledSamples (n2 - 1)  -- Shuffle the samples at every iteration of training
                        where
                            trainEach :: (KnownNat i, KnownNat o) => Network i hs o -> (SA.R i, SA.R o) -> Network i hs o
                            trainEach nt (i2, o2) = train learningrate i2 o2 nt

                            zippedSamples = zip i o
                            shuffledSamples = unzip (shuffle' zippedSamples (length zippedSamples) gen)

        outMatInit = [( SA.vector $ take inputD x, SA.vector $ lastN outputD x, runNet initnet (SA.vector (take inputD x)))
                       | x <- samples ]

        outMat     = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, runNet trained (SA.vector (take inputD x)))
                       | x <- samples ]


    return (initnet, outMatInit, trained, outMat)



-- Network prediction with full response, inputs, expected and predicted outputs
netPredict :: (KnownNat i, KnownNat o) => Network i hs o -> [[Double]] -> (Int, Int) -> [(SA.R i, SA.R o, SA.R o)]
netPredict neuralnet samples (inputD, outputD) = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, runNet neuralnet (SA.vector (take inputD x))) | x <- samples ]




-- Function to run the Network over a list of Samples, and applaying a NetFilter
runNetFiltered :: (KnownNat i, KnownNat o) => Network i hs o -> [[Double]] -> (Int, Int) -> NetFilter -> [(SA.R i, SA.R o, SA.R o)]
runNetFiltered net samples (inputD, outputD) filterF = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, nnFilter (runNet net ( SA.vector (take inputD x)))) | x <- samples ]

                                                            where

                                                              nnFilter = getFilter filterF



-- Function to Output a list of responses from a Network
renderOutput :: (KnownNat i, KnownNat o) => [(SA.R i, SA.R o, SA.R o)] -> String
renderOutput samples = unlines $ Prelude.map render samples
                          where
                            render (inputs, outputs, netResult) = "Inputs: " ++ show inputs ++ ", Expected Outputs: " ++ show outputs ++ ", Neural Network Results: " ++ show netResult




-- Function to check the Network accuracy based on a list of samples and the expected outputs and the Network predictions
checkAccuracy :: (KnownNat o) =>[(SA.R i, SA.R o, SA.R o)] -> Double
checkAccuracy xs = 100 * Prelude.foldr checkAc 0 xs / fromIntegral(length xs)
                      where

                        checkAc (_, expO, netO) acc = if VecSized.toList (SA.rVec expO) == VecSized.toList (SA.rVec netO) then acc + 1 else acc



-- Auxiliar function to read samples from a String
stringToSamples :: String -> [[Double]]
stringToSamples x = Prelude.map (Prelude.map readSamples . words) (lines x)
                      where
                        readSamples y = read y :: Double


compareNets :: (Eq a) => a -> a -> Bool
compareNets a b = a == b



main :: IO ()
main = do
    args <- getArgs
    let n    :: Maybe Int    = readMaybe =<< (args !!? 0)
        rate :: Maybe Double = readMaybe =<< (args !!? 1)
    samplesFile <- readFile "/home/kali/Downloads/UFABC/PGC/Github/TypeSafeNeuralNetwork/inputs/inputs30K.txt"

    -- Read input file to get all samples to train the neural network!
    let samples = stringToSamples samplesFile
    putStrLn "\n\n50 primeiras Samples do txt:"
    print (take 50 samples)
    let inputD = 2 :: Int
    let outputD = 1 :: Int
    let dimensions = (inputD, outputD) :: (Int, Int)

    print (n, rate, inputD, outputD, dimensions)

    putStrLn "Teste do readLn hs, digite:"

    hs :: [Natural] <- readLn

    print hs

    rn :: OpaqueNet 2 1 <- randomONet hs [Linear]

    print rn

    initialNet  :: Network 2 '[5] 1    <- randomNet [Logistic, Linear]

    initialNet2 :: Network 2 '[5] 1    <- randomNet [Logistic, Linear]

    putStrLn $ "initialNet e initialNet2 Eq check? R: " ++ show ( initialNet == initialNet2)
    putStrLn $ "initialNet e initialNet Eq check? R: "  ++ show ( initialNet == initialNet)



    putStrLn "Showing initial Network, before training:\n\n"
    putStrLn "\n\ninitialNet:\n"
    print initialNet

    putStrLn "\n\nTraining network..."

    (_, _, netTrained, outputS) <- netTrain initialNet
                                  (fromMaybe 0.0025   rate)
                                  (fromMaybe 1000 n   )
                                  (take 100 samples)
                                   dimensions

    putStrLn $ "Are initialNet e netTrained equals? R: " ++ show ( initialNet == netTrained)

    putStrLn "\n\nNetwork after training: \n\n"
    print netTrained

    putStrLn "\n\n\nShowing network prediction of:\n"
    putStrLn $ "\n-------> Accuracy: " ++ show (checkAccuracy outputS) ++ " % <------"
    putStrLn $ renderOutput outputS
    putStrLn "\n\n\nShowing network filtered prediction, with BinaryOutput NetFilter applied:\n"
    let filteredResults = runNetFiltered netTrained (take 200 samples) dimensions BinaryOutput
    putStrLn $ "\n-------> Accuracy: " ++ show (checkAccuracy filteredResults) ++ " % <------"
    putStrLn $ renderOutput filteredResults



    putStrLn "\n\nWriting the trained Network in the file: trainedNet.tsnn ...."
    BSL.writeFile "trainedNet.tsnn" $ encode netTrained
    putStrLn "\nLoading the Network from file trainedNet.tsnn, printing it:"
    byteStringNet <- BSL.readFile "trainedNet.tsnn"
    let fileTrainedNet :: Network 2 '[5] 1 = deserializeNetwork byteStringNet
    print fileTrainedNet
    putStrLn $ "Are netTrained and fileTrainedNet equals? R: " ++ show ( netTrained == fileTrainedNet)
    putStrLn "\nNetwork succesfuly saved, shuting down the execution!"

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
