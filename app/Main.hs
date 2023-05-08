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
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PolyKinds #-}


module Main where

--import Lib

import Control.Monad ()
import Control.Monad.Random
--import Data.Vector.Storable (basicLength)
import Data.List ( foldl' )
import GHC.Float ()
import Data.Maybe
--import Numeric.LinearAlgebra
import System.Environment
import Text.Read
import Data.Kind (Type)
import GHC.Generics
import Data.Binary
import qualified Data.ByteString.Lazy as BSL

import System.Random.Shuffle
import GHC.TypeLits (Nat, KnownNat)
import GHC.TypeLits.Singletons
import Numeric.LinearAlgebra
--import Numeric.LinearAlgebra.Static

import Numeric.LinearAlgebra.Static.Vector as StaticVector

import qualified Data.Vector.Storable.Sized as SV
import qualified Numeric.LinearAlgebra.Static as SA

import Data.Singletons



--import qualified Prelude as Numeric.LinearAlgebra

-- Using hmatrix minimization methods
--import Numeric.GSL.Minimization

--data family Sing (x :: k)

data SList xs where
  SNil  :: SList '[]
  SCons :: Sing x -> SList xs -> SList (x ': xs)




data Weights i o = W { wBiases  :: !(SA.R o)  -- n
                        , wNodes  :: !(SA.L o i)  -- n x m
                      }                              -- "m to n" layer
                  deriving (Generic)



data Activation = Linear | Logistic | Tangent | ReLu | LeakyReLu | ELU Double deriving (Show, Generic)


--getFunctions :: (Ord a, Floating a) => Activation -> (a -> a, a -> a)
getFunctions :: Activation -> (SA.R i -> SA.R i, SA.R i -> SA.R i)
getFunctions f = case f of
                  Linear      -> (linear, linear')
                  Logistic    -> (logistic, logistic')
                  Tangent     -> (tangent, tangent')
                  ReLu        -> (relu, relu')
                  LeakyReLu   -> (lrelu, lrelu')
                  ELU a       -> (elu a, elu' a)


data Network :: Nat -> Activation -> [(Nat, Activation)] -> Nat -> * where
    O     :: !(Weights i o) -> Activation
          -> Network i f '[] o
    (:&~) :: (KnownNat h, SingI hs)
          => (Weights i h , Activation)
          -> !(Network h f2 hs o)
          -> Network i f ((h, f2) ': hs :: [(Nat, Activation)])  o
infixr 5 :&~


instance Show (Network i f hs fs o) where        -- Implementacao de instancia de show de Network para facilitar o debug
  show :: Network i f hs fs o -> String
  show (O a f)          =  "Nos de saida: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f
  show ((a , f) :&~ b)  =  "Nos camada: "   ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f ++ "\n" ++ show b


-- Definicao de instancias para serializar a rede:

instance Binary (Weights i o)
instance Binary Activation
instance Binary (Network i f hs fs o)

-- Definicao de funcoes para serializar e desserializar a rede

-- Serializa um modelo de Rede para uma ByteString
serializeNetwork :: Network i f hs fs o -> BSL.ByteString
serializeNetwork = encode

-- Desserializa um modelo de Rede a partir de uma ByteString
deserializeNetwork :: BSL.ByteString -> Network i f hs fs o
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


--relu :: KnownNat i => SA.R i -> SA.R i
relu :: Floating a => a -> a
relu = (max 0)

--relu' :: SA.R i -> SA.R i
relu' ::Floating a => a -> a
relu' x = if x > 0 then 1 else 0


--lrelu :: SA.R i -> SA.R i
lrelu :: Floating a => a -> a
lrelu y = max (0.01*y) y

lrelu' :: SA.R i -> SA.R i
lrelu' y = if y > 0 then 1 else 0.01


--elu :: Double -> SA.R i -> SA.R i
--elu :: Floating a => a -> a -> a
--elu a  = (\y -> if y >= 0 then y else a * (exp y - 1))

elu :: (KnownNat n) => Double -> SA.R n -> SA.R n
elu a y = StaticVector.vecR $ cmap (\x -> if x >= 0 then x else a * (exp x - 1)) $ StaticVector.rVec y


--elu' :: Double -> SA.R i -> SA.R i
elu' :: (KnownNat n) => Double -> SA.R n -> SA.R n
elu' a y = StaticVector.vecR $ cmap (\x -> if x >= 0 then 1 else a + a * (exp x - 1)) $ StaticVector.rVec y


-- Auxiliar way to define a derivative of a function, using limits (can be very unprecise)
derive :: (Fractional a) => a -> (a -> a) -> (a -> a)
derive h f x = (f (x+h) - f x) / h


-- Definiton of Filters to output of the neural network

data Filter = BinaryOutput | SoftMax deriving Show


getFilter :: Filter -> (SA.R i-> SA.R i)
getFilter f = case f of

                BinaryOutput   ->   binaryOutput
                SoftMax        ->   softmaxOut


-- Auxiliar definitions to  Filters

binaryOutput :: SA.R i -> SA.R i
binaryOutput x = StaticVector.vecR $ cmap (\y -> if y > 0.5 then 1 else 0) $ StaticVector.rVec x


softmaxOut :: SA.R i -> SA.R i
softmaxOut x = StaticVector.vecR $ cmap (/ sumElements expX) expX
              where
                  expX = cmap exp $ StaticVector.rVec x


-- Definitions of functions to run the network itself

runLayer :: (KnownNat i, KnownNat o) => Weights i o -> SA.R i -> SA.R o
runLayer (W wB wN) v = wB + (wN SA.#> v)


runNet :: (KnownNat i, KnownNat o) => Network i f hs fs o -> SA.R i -> SA.R o
runNet = \case
   O w f -> \(!v)  ->          let (function, _) = getFunctions f
                                in function (runLayer w v)

   ((w, f) :&~ n') -> \(!v) -> let
                                  (function, _) = getFunctions f
                                  v' = function (runLayer w v)
                                in  runNet n' v'

-- Definitions of functions to generate a random network

randomWeights :: (MonadRandom m, KnownNat i, KnownNat o) => m (Weights i o)
randomWeights = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wB = SA.randomVector  seed1 Uniform * 2 - 1
        wN = SA.uniformSample seed2 (-1) 1
    return $ W wB wN


randomNet :: forall m i f hs o. (MonadRandom m, KnownNat i, SingI hs, KnownNat o) => Activation -> Sing (hs :: [(Nat, Activation)]) -> m (Network i f hs o)
randomNet activation = go activation sing
  where
    go :: forall h f' hs' fs'. KnownNat h
       => Activation -> Sing hs'
       -> m (Network h f' hs' fs' o)
    go f' SNil              =  O f'   <$> randomWeights
    go f' (SNat `SCons` ss) = f' (:&~)  <$> randomWeights <*> go ss



-- Training function, train the network for just one iteration

train :: Double           -- ^ learning rate
      -> Vector Double    -- ^ input vector
      -> Vector Double    -- ^ target vector
      -> Network i f hs fs o          -- ^ network to train
      -> Network i f hs fs o
train rate x0 target = fst . go x0
  where
    go :: Vector Double    -- ^ input vector
       -> Network i f hs fs o          -- ^ network to train
       -> (Network i f hs fs o, Vector Double)
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
netTrain :: (MonadRandom m, MonadIO m) =>  Network i f hs fs o -> Double -> Int -> [[Double]] -> (Int, Int) -> m (Network i f hs fs o, [([Double], [Double], [Double])], Network i f hs fs o, [([Double], [Double], [Double])])
netTrain initnet learningrate nruns samples (inputD, outputD) = do

    let inps = map (Numeric.LinearAlgebra.fromList . take inputD) samples
    let outs = map (Numeric.LinearAlgebra.fromList . lastN outputD) samples

    gen <- newStdGen

    let trained = trainNTimes initnet (inps, outs) nruns
          where
            trainNTimes :: Network i f hs fs o -> ([Vector Double], [Vector Double]) -> Int -> Network i f hs fs o
            trainNTimes net (i, o) n2
                | n2 <= 0 = net
                | otherwise = trainNTimes (foldl' trainEach net (zip i o)) shuffledSamples (n2 - 1)  -- Shuffle the samples at every iteration of training
                        where
                            trainEach :: Network i f hs fs o -> (Vector Double, Vector Double) -> Network i f hs fs o
                            trainEach nt (i2, o2) = train learningrate i2 o2 nt

                            zippedSamples = zip i o
                            shuffledSamples = unzip (shuffle' zippedSamples (length zippedSamples) gen)

        outMatInit = [( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList $ SA.extract $ (runNet initnet (SA.vector (take inputD x))))
                       | x <- samples ]

        outMat     = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList $ SA.extract $ (runNet trained (SA.vector (take inputD x))))
                       | x <- samples ]


    return (initnet, outMatInit, trained, outMat)



-- Network prediction with full responde, inputs, expected and predicted outputs
netPredict :: Network i f hs fs o -> [[Double]] -> (Int, Int) -> [([Double], [Double], [Double])]
netPredict neuralnet samples (inputD, outputD) = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList $ SA.extract $ (runNet neuralnet (SA.vector (take inputD x)))) | x <- samples ]





runNetFiltered :: Network i f hs fs o -> [[Double]] -> (Int, Int) -> Filter -> [([Double], [Double], [Double])]
runNetFiltered net samples (inputD, outputD) filterF = [ ( take inputD x, lastN outputD x, Numeric.LinearAlgebra.toList $ SA.extract $ nnFilter (runNet net ( SA.vector (take inputD x)))) | x <- samples ]

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
    samplesFile <- readFile "/home/kali/Downloads/UFABC/PGC/Github/TypeSafeNeuralNetwork/inputs/inputs30K.txt"

    -- Read input file to get all samples to train the neural network!
    let samples = stringToSamples samplesFile
    putStrLn "\n\n50 primeiras Samples do txt:"
    print (take 50 samples)
    let inputD = 2 :: Int
    let outputD = 1 :: Int
    let dimensions = (inputD, outputD) :: (Int, Int)

    -- Exemplo que mostra que a rede nao esta segura contra formas incoerentes...
    testNet <- randomNet (-1) Linear [(-2, Linear)] (-3)

    initialNet <- randomNet inputD (ELU 0.5) [(5, Linear )] outputD

    putStrLn "\n\n\nImprimindo a rede inicial teste:\n"
    print initialNet


    putStrLn "\n\nTraining network..."

    (_, _, netTrained, outputS) <- netTrain initialNet
                                 (fromMaybe 0.00025   rate)  -- init v 0.0025
                                 (fromMaybe 1000 n   )   -- init value 150 log log 
                                 (take 20000 samples)
                                 dimensions



    putStrLn "\n\n\nImprimindo predicao agora treinada:\n"
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy outputS) ++ " %"
    --putStrLn $ renderOutput outputS
    putStrLn "\n\n\nImprimindo predicao treinada, agora com filtro de saida da rede:\n"
    let filteredResults = runNetFiltered netTrained (take 20000 samples) dimensions BinaryOutput
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy filteredResults) ++ " %"
    --putStrLn $ renderOutput filteredResults


    putStrLn "\n\n\nVerificando agora a performance em samples que nao foram usadas no treino:"
    let filteredResultsFinal = runNetFiltered netTrained (drop 20000 samples) dimensions BinaryOutput
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy filteredResultsFinal) ++ " %"
    --putStrLn $ renderOutput filteredResultsFinal

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
