{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
--{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-missing-export-lists #-}


module Main where

--import Lib

import Control.Monad ()
import Control.Monad.Random
--import Data.Vector.Storable (basicLength)
import Data.List ( foldl' )
import GHC.Float ()
import Data.Maybe
import Numeric.LinearAlgebra
import System.Environment
import Text.Read
import Data.Kind (Type)
import GHC.Generics
import Data.Binary
import qualified Data.ByteString.Lazy as BSL

import System.Random.Shuffle

--import qualified Prelude as Numeric.LinearAlgebra

-- Using hmatrix minimization methods
--import Numeric.GSL.Minimization



data Weights = W { wBiases :: !(Vector Double)  -- n
                 , wNodes  :: !(Matrix Double)  -- n x m
                 }                              -- "m to n" layer
                  deriving (Generic)


data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double deriving (Show, Generic)


--getFunctions :: (Ord a, Floating a) => Activation -> (a -> a, a -> a)
getFunctions :: Activation -> (Vector Double -> Vector Double, Vector Double -> Vector Double)
getFunctions f = case f of
                  Linear      -> (linear, linear')
                  Logistic    -> (logistic, logistic')
                  Tangent     -> (tangent, tangent')
                  ReLu        -> (relu, relu')
                  LeakyReLu   -> (lrelu, lrelu')
                  ELU a       -> (elu a, elu' a)


data Network :: Type where
    O     :: !Activation -> !Weights
          -> Network
    (:&~) :: (Activation , Weights)
          -> !Network
          -> Network
                          deriving Generic
infixr 5 :&~


instance Show Network where        -- Implementacao de instancia de show de Network para facilitar o debug
  show :: Network -> String
  show (O f a)          =  "Nos de saida: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f
  show ((f , a) :&~ b)  =  "Nos camada: "   ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f ++ "\n" ++ show b


-- Definicao de instancias para serializar a rede:

instance Binary Weights
instance Binary Activation
instance Binary Network

-- Definicao de funcoes para serializar e desserializar a rede

-- Serializa um modelo de Rede para uma ByteString
serializeNetwork :: Network -> BSL.ByteString
serializeNetwork = encode

-- Desserializa um modelo de Rede a partir de uma ByteString
deserializeNetwork :: BSL.ByteString -> Network
deserializeNetwork = decode



-- Auxiliar definition of activation functions and it's derivatives

linear :: a -> a
linear x = x

linear' :: Floating a => a -> a
linear' _ =  1

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x


tangent :: Floating a => a -> a
tangent x = (exp x - exp (-x)) / (exp x + exp (-x))

tangent' :: Floating a => a -> a
tangent' x = 1 + tangent x * tangent x


relu :: Vector Double -> Vector Double
relu = cmap (max 0)

relu' :: Vector Double -> Vector Double
relu' = cmap (\y -> if y > 0 then 1 else 0)


lrelu :: Vector Double -> Vector Double
lrelu  = cmap (\y -> max (0.01*y) y)

lrelu' :: Vector Double -> Vector Double
lrelu' = cmap (\y -> if y > 0 then 1 else 0.01)


elu :: Double -> Vector Double -> Vector Double 
elu a  = cmap (\y -> if y >= 0 then y else a * (exp y - 1))

elu' :: Double -> Vector Double -> Vector Double 
elu' a  = cmap (\y -> if y >= 0 then 1 else a + (a * (exp y - 1)))


-- Auxiliar way to define a derivative of a function, using limits (can be very unprecise)
derive :: (Fractional a) => a -> (a -> a) -> (a -> a)
derive h f x = (f (x+h) - f x) / h


-- Definiton of Filters to output of the neural network

data Filter = BinaryOutput | SoftMax deriving Show


getFilter :: Filter -> (Vector Double -> Vector Double)
getFilter f = case f of 

                BinaryOutput   ->   binaryOutput 
                SoftMax        ->   softmaxOut


-- Auxiliar definitions to  Filters

binaryOutput :: Vector Double -> Vector Double
binaryOutput = cmap (\y -> if y > 0.5 then 1 else 0)


softmaxOut :: Vector Double -> Vector Double
softmaxOut x = cmap (/ sumElements expX) expX
              where
                  expX = cmap exp x


-- Definitions of functions to run the network itself

runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + (wN #> v)

runNet :: Network -> Vector Double -> Vector Double
runNet (O f w)      !v = let (function, _) = getFunctions f
                          in function (runLayer w v)
runNet ((f,w) :&~ n') !v = let
                              (function, _) = getFunctions f
                              v' = function (runLayer w v)
                            in  runNet n' v'


-- Definitions of functions to generate a random network

randomWeights :: MonadRandom m => Int -> Int -> m Weights
randomWeights i o = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wB = randomVector  seed1 Uniform o * 2 - 1
        wN = uniformSample seed2 o (replicate i (-1, 1))
    return $ W wB wN


randomNet :: MonadRandom m => Int -> Activation -> [(Int, Activation)] -> Int -> m Network
randomNet i f []     o =     O f <$> randomWeights i o
randomNet i f ((h,f2):hs) o = do
                                w <- randomWeights i h
                                (:&~) (f, w)  <$> randomNet h f2 hs o




-- Training function, train the network for just one iteration

train :: Double           -- ^ learning rate
      -> Vector Double    -- ^ input vector
      -> Vector Double    -- ^ target vector
      -> Network          -- ^ network to train
      -> Network
train rate x0 target = fst . go x0
  where
    go :: Vector Double    -- ^ input vector
       -> Network          -- ^ network to train
       -> (Network, Vector Double)
    -- handle the output layer
    go !x (O f w@(W wB wN))
        = let y    = runLayer w x
              (function, derivative) = getFunctions f
              o    = function y
              -- the gradient (how much y affects the error)
              dEdy = derivative y * (o - target)

              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (O f w', dWs)

    -- handle the inner layers
    go !x ((f, w@(W wB wN)) :&~ n)
        = let y          = runLayer w x
              (function, derivative) = getFunctions f
              o          = function y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = derivative y * dWs'
        
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  ((f, w') :&~ n', dWs)




-- Auxiliar functions to apply to samples
lastN :: Int -> [a] -> [a]
lastN n xs = drop (length xs - n) xs


-- atualizar para versao final de treino de rede: receber entradas E saidas, receber modelo inicial de rede construido fora da funcao de treino!
netTrain :: (MonadRandom m, MonadIO m) => Network -> Double -> Int -> [[Double]] -> (Int, Int) -> m (Network, [([Double], [Double], [Double])], Network, [([Double], [Double], [Double])])
netTrain initnet learningrate nruns samples (inputD, outputD) = do

    let inps = map (Numeric.LinearAlgebra.fromList . take inputD) samples
    let outs = map (Numeric.LinearAlgebra.fromList . lastN outputD) samples

    gen <- newStdGen

    let trained = trainNTimes initnet (inps, outs) nruns
          where
            trainNTimes :: Network -> ([Vector Double], [Vector Double]) -> Int -> Network
            trainNTimes net (i, o) n2
                | n2 <= 0 = net
                | otherwise = trainNTimes (foldl' trainEach net (zip i o)) shuffledSamples (n2 - 1)  -- Shuffle the samples at every iteration of training
                        where
                            trainEach :: Network -> (Vector Double, Vector Double) -> Network
                            trainEach nt (i2, o2) = train learningrate i2 o2 nt

                            zippedSamples = zip i o
                            shuffledSamples = unzip (shuffle' zippedSamples (length zippedSamples) gen)

        outMatInit = [( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList(runNet initnet (vector (take inputD x))))
                       | x <- samples ]

        outMat     = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList(runNet trained (vector (take inputD x))))
                       | x <- samples ]


    return (initnet, outMatInit, trained, outMat)



-- Network prediction with full responde, inputs, expected and predicted outputs
netPredict :: Network -> [[Double]] -> (Int, Int) -> [([Double], [Double], [Double])]
netPredict neuralnet samples (inputD, outputD) = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList(runNet neuralnet (vector (take inputD x)))) | x <- samples ]





runNetFiltered :: Network -> [[Double]] -> (Int, Int) -> Filter -> [([Double], [Double], [Double])]
runNetFiltered net samples (inputD, outputD) filterF = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList $ nnFilter (runNet net (vector (take inputD x)))) | x <- samples ]

                                                            where

                                                              nnFilter = getFilter filterF



renderOutput :: [([Double], [Double], [Double])] -> String
renderOutput samples = unlines $ map render samples
                          where
                            render (inputs, outputs, netResult) = "Inputs: " ++ show inputs ++ ", Expected Outputs: " ++ show outputs ++ ", Neural Network Results: " ++ show netResult


-- definir funcao para checar precisao da rede
checkAccuracy :: [([Double], [Double], [Double])] -> Double
checkAccuracy xs = 100 * foldr checkAc 0 xs / fromIntegral(length xs)
                      where

                        checkAc (_, expO, netO) acc = if expO == netO then acc + 1 else acc




stringToSamples :: String -> [[Double]]
stringToSamples x = map (map readSamples . words) (lines x)
                      where
                        readSamples y = read y :: Double




main :: IO ()
main = do
    args <- getArgs
    let n    = readMaybe =<< (args !!? 0)
        rate = readMaybe =<< (args !!? 1)
    samplesFile <- readFile "/home/kali/Downloads/UFABC/PGC/Github/TypeSafeNeuralNetwork/inputs/inputs.txt"

    -- Read input file to get all samples to train the neural network!
    let samples = stringToSamples samplesFile
    putStrLn "\n\n50 primeiras Samples do txt:"
    print (take 50 samples)
    let inputD = 2 :: Int
    let outputD = 1 :: Int
    let dimensions = (inputD, outputD) :: (Int, Int)


    initialNet <- randomNet inputD (ELU 0.5) [(5, Linear )] outputD

    putStrLn "\n\n\nImprimindo a rede inicial teste:\n"
    print initialNet


    putStrLn "\n\nTraining network..."

    (_, _, netTrained, outputS) <- netTrain initialNet
                                 (fromMaybe 0.00025   rate)  -- init v 0.0025
                                 (fromMaybe 1000 n   )   -- init value 150 log log 
                                 samples
                                 dimensions



    putStrLn "\n\n\nImprimindo predicao agora treinada:\n"
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy outputS) ++ " %"
    putStrLn $ renderOutput outputS
    putStrLn "\n\n\nImprimindo predicao treinada, agora com filtro de saida da rede:\n"
    let filteredResults = runNetFiltered netTrained samples dimensions BinaryOutput
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy filteredResults) ++ " %"
    putStrLn $ renderOutput filteredResults

    putStrLn "\n\n\nAgora imprimindo a rede final:\n"
    print netTrained


    putStrLn "\n\nSalvando rede treinada em arquivo: redetreinada.tsnn...."
    BSL.writeFile "redetreinada.tsnn" $ encode netTrained
    putStrLn "\nCarregando rede treianda do arquivo e exibindo:"
    byteStringNet <- BSL.readFile "redetreinada.tsnn"
    let fileTrainedNet = deserializeNetwork byteStringNet
    print fileTrainedNet
    putStrLn "\nRede salva com sucesso, encerrando a execucao!"

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
