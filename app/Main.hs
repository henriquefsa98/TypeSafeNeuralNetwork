{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
--{-# LANGUAGE KindSignatures      #-}
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
--{-# LANGUAGE StandaloneKindSignatures #-}
--{-# LANGUAGE TypeApplications #-}
--{-# LANGUAGE TypeFamilies #-}
--{-# LANGUAGE StandaloneDeriving  #-}
--{-# LANGUAGE AllowAmbiguousTypes #-}

module Main where

--import Lib

import Control.Monad ()
import Control.Monad.Random
--import Data.Vector.Storable (basicLength)
import GHC.Float ()
import Data.Maybe
--import Numeric.LinearAlgebra
import System.Environment
import Text.Read
import Data.Kind (Type)
import GHC.Generics

import GHC.TypeLits.Singletons
import Numeric.LinearAlgebra as NonStatic


import qualified Numeric.LinearAlgebra.Static as SA

import Data.Singletons
import Data.List.Singletons
import qualified Numeric.LinearAlgebra.Static.Vector as SA
import Data.Vector.Storable.Sized as VecSized (toList, fromList) -- Usar isso para converter SA.rVec para listas!
import Data.Binary
import qualified Data.ByteString.Lazy as BSL
import qualified GHC.Exts as SV 
import Data.List (foldl')
import System.Random.Shuffle (shuffle')


--import qualified Prelude as Numeric.LinearAlgebra

-- Using hmatrix minimization methods
--import Numeric.GSL.Minimization

--data family Sing (x :: k)

{-
data SList xs where
  SNil  :: SList '[]
  SCons :: Sing x -> SList xs -> SList (x ': xs)
-}



data Weights i o = W { wBiases  :: !(SA.R o)  -- n
                        , wNodes  :: !(SA.L o i)  -- n x m
                      }                              -- "m to n" layer
                  deriving (Show, Generic)



data Activation = Linear | Logistic | Tangent | ReLu {-| LeakyReLu | ELU Double-} deriving (Show, Generic)

{-
-- Define the promoted version of Activation
data SActivation :: Activation -> Type where
  SLinear :: SActivation 'Linear
  SLogistic :: SActivation 'Logistic
  STangent :: SActivation 'Tangent
  SReLu :: SActivation 'ReLu
  SLeakyReLu :: SActivation 'LeakyReLu
  SELU :: Double -> SActivation ('ELU a)
-}

-- Generate the singleton instances for Activation
-- $(TH.genSingletons [''Activation])


--getFunctions :: (Floating a) => Activation -> (a -> a, a -> a)
--getFunctions :: (KnownNat i) => Activation -> (SA.R i -> SA.R i, SA.R i -> SA.R i)
getFunctions :: (KnownNat i) => Activation -> (SA.R i -> SA.R i, SA.R i -> SA.R i)
getFunctions f = case f of
                  Linear      -> (linear, linear')
                  Logistic    -> (logistic, logistic')
                  Tangent     -> (tangent, tangent')
                  ReLu        -> (relu, relu')
                  --LeakyReLu   -> (lrelu, lrelu')
                  --ELU a       -> (elu a, elu' a)


data Network :: Nat -> [Nat] -> Nat -> Type where
    O     :: !(Weights i o) -> Activation
          -> Network i '[] o
    (:&~) :: (KnownNat h) => Weights i h -> Activation
          -> !(Network h hs o)
          -> Network i (h ': hs)  o
                                    --deriving (Generic)
infixr 5 :&~



instance (KnownNat i, KnownNat o) => Show (Network i hs o) where        -- Implementacao de instancia de show de Network para facilitar o debug
  show :: Network i hs o -> String
  show (O a f)          =  "Nos de saida: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f
  show ((:&~) a f b)  =  "Nos camada: "   ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f ++ "\n" ++ show b




{-
-- Definicao de instancias para serializar a rede:

instance (KnownNat i, KnownNat o) => Binary (Weights i o)
instance Binary Activation
instance (KnownNat i, KnownNat o) => Binary (Network i hs o)

-- Definicao de funcoes para serializar e desserializar a rede

-- Serializa um modelo de Rede para uma ByteString
serializeNetwork :: Network i hs o -> BSL.ByteString
serializeNetwork = encode

-- Desserializa um modelo de Rede a partir de uma ByteString
deserializeNetwork :: BSL.ByteString -> Network i hs o
deserializeNetwork = decode

-}


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


relu ::  KnownNat i =>  SA.R i -> SA.R i
--relu :: (Floating a) => a -> a
relu x = SA.vecR $ (max 0) $ SA.rVec x

relu' :: KnownNat i => SA.R i -> SA.R i
--relu' :: (Floating a) => a -> a
relu' x = SA.vecR $ max 0 $ SA.rVec x

{-
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



-}


-- Auxiliar way to define a derivative of a function, using limits (can be very unprecise)
derive :: (Fractional a) => a -> (a -> a) -> (a -> a)
derive h f x = (f (x+h) - f x) / h


-- Definiton of NetFilters to output of the neural network

data NetFilter = BinaryOutput {-| SoftMax-} deriving Show


getFilter :: (KnownNat i) => NetFilter -> (SA.R i-> SA.R i)
getFilter f = case f of

                BinaryOutput   ->   binaryOutput
                --SoftMax        ->   softmaxOut


-- Auxiliar definitions to  Filters

binaryOutput :: (KnownNat i) => SA.R i -> SA.R i
binaryOutput x = SA.dvmap (\y -> if y > 0.5 then 1 else 0) x

{-
softmaxOut :: (KnownNat i) => SA.R i -> SA.R i
softmaxOut x = SA.dvmap (/ total) SA.extract x
              where
                  total = sumElements $ SA.rVec x
-}

-- Definitions of functions to run the network itself

runLayer :: (KnownNat i, KnownNat o) => Weights i o -> SA.R i -> SA.R o
runLayer (W wB wN) v = wB + (wN SA.#> v)


runNet :: (KnownNat i, KnownNat o) => Network i hs o -> SA.R i -> SA.R o
runNet = \case
   O w f -> \(!v)  ->          let (function, _) = getFunctions f
                                in function (runLayer w v)

   ((:&~) w f n') -> \(!v) -> let
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


getAct :: [Activation] -> Activation
getAct (a:_) = a
getAct []     = Linear


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


-- Training function, train the network for just one iteration

train :: forall i hs o. (KnownNat i, KnownNat o)
      => Double           -- ^ learning rate
      -> SA.R i    -- ^ input vector
      -> SA.R o    -- ^ target vector
      -> Network i hs o          -- ^ network to train
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
              wN'  = wN - SA.konst rate * (SA.outer dEdy x)
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
              wN'  = wN - SA.konst rate * (SA.outer dEdy  x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN SA.#> dEdy
          in  ((:&~) w' f n', dWs)




-- Auxiliar functions to apply to samples
lastN :: Int -> [a] -> [a]
lastN n xs = drop (length xs - n) xs


-- atualizar para versao final de treino de rede: receber entradas E saidas, receber modelo inicial de rede construido fora da funcao de treino!
netTrain :: (MonadRandom m, MonadIO m, KnownNat i, KnownNat o, SingI hs) =>  Network i hs o -> Double -> Int -> [[Double]] -> (Int, Int) -> m (Network i hs o, [(SA.R i, SA.R 0, SA.R o)], Network i hs o, [(SA.R i, SA.R o, SA.R o)])
netTrain initnet learningrate nruns samples (inputD, outputD) = do

    let inps = map (SA.vector . take inputD) samples
    let outs = map (SA.vector . lastN outputD) samples

    gen <- newStdGen

    let trained = trainNTimes initnet (inps, outs) nruns
          where
            trainNTimes :: (KnownNat i, SingI hs, KnownNat o) => Network i hs o -> ([SA.R i], [SA.R o]) -> Int -> Network i hs o
            trainNTimes net (i, o) n2
                | n2 <= 0 = net
                | otherwise = trainNTimes (foldl' trainEach net (zip i o)) shuffledSamples (n2 - 1)  -- Shuffle the samples at every iteration of training
                        where
                            trainEach :: (KnownNat i, SingI hs, KnownNat o) => Network i hs o -> (SA.R i, SA.R o) -> Network i hs o
                            trainEach nt (i2, o2) = train learningrate i2 o2 nt

                            zippedSamples = zip i o
                            shuffledSamples = unzip (shuffle' zippedSamples (length zippedSamples) gen)

        outMatInit = [( SA.vector $ take inputD x, SA.vector $ lastN outputD x, (runNet initnet (SA.vector (take inputD x))))
                       | x <- samples ]

        outMat     = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, (runNet trained (SA.vector (take inputD x))))
                       | x <- samples ]


    return (initnet, outMatInit, trained, outMat)



-- Network prediction with full responde, inputs, expected and predicted outputs
netPredict :: (KnownNat i, KnownNat o) => Network i hs o -> [[Double]] -> (Int, Int) -> [(SA.R i, SA.R o, SA.R o)]
netPredict neuralnet samples (inputD, outputD) = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, (runNet neuralnet (SA.vector (take inputD x)))) | x <- samples ]





runNetFiltered :: (KnownNat i, KnownNat o) => Network i hs o -> [[Double]] -> (Int, Int) -> NetFilter -> [(SA.R i, SA.R o, SA.R o)]
runNetFiltered net samples (inputD, outputD) filterF = [ ( SA.vector $ take inputD x, SA.vector $ lastN outputD x, nnFilter (runNet net ( SA.vector (take inputD x)))) | x <- samples ]

                                                            where

                                                              nnFilter = getFilter filterF



renderOutput :: (KnownNat i, KnownNat o) => [(SA.R i, SA.R o, SA.R o)] -> String
renderOutput samples = unlines $ map render samples
                          where
                            render (inputs, outputs, netResult) = "Inputs: " ++ show inputs ++ ", Expected Outputs: " ++ show outputs ++ ", Neural Network Results: " ++ show netResult




-- definir funcao para checar precisao da rede
checkAccuracy :: (KnownNat o) =>[(SA.R i, SA.R o, SA.R o)] -> Double
checkAccuracy xs = 100 * foldr checkAc 0 xs / fromIntegral(length xs)
                      where

                        checkAc (_, expO, netO) acc = if VecSized.toList (SA.rVec expO) == VecSized.toList (SA.rVec netO) then acc + 1 else acc




stringToSamples :: String -> [[Double]]
stringToSamples x = map (map readSamples . words) (lines x)
                      where
                        readSamples y = read y :: Double




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

    -- Exemplo que mostra que a rede nao esta segura contra formas incoerentes...
    testNet :: Network 2 '[2, 3] 1 <- randomNet [Logistic, Logistic, Logistic]

    initialNet :: Network 2 '[5] 1 <- randomNet [(Logistic), Linear]

    --putStrLn "\n\n\nImprimindo a rede inicial teste:\n"
    --print initialNet

    putStrLn "Imprimindo testNet e initialNet:\n\n"
    putStrLn "testNet:\n"
    print testNet
    putStrLn "\n\ninitialNet:\n"
    print initialNet

    putStrLn "\n\nTraining network..."

    (_, _, netTrained, outputS) <- netTrain initialNet
                                  (fromMaybe 0.00025   rate)  -- init v 0.0025
                                  (fromMaybe 1000 n   )   -- init value 150 log log 
                                  (take 100 samples)
                                   dimensions

    putStrLn "\n\nRede atualizada apos treino: \n\n"
    print netTrained

    putStrLn "\n\n\nImprimindo predicao agora treinada:\n"
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy outputS) ++ " %"
    putStrLn $ renderOutput outputS
    putStrLn "\n\n\nImprimindo predicao treinada, agora com filtro de saida da rede:\n"
    let filteredResults = runNetFiltered netTrained (take 200 samples) dimensions BinaryOutput
    putStrLn $ "\nAcuracia: " ++ show (checkAccuracy filteredResults) ++ " %"
    putStrLn $ renderOutput filteredResults


    --putStrLn "\n\n\nVerificando agora a performance em samples que nao foram usadas no treino:"
    --let filteredResultsFinal = runNetFiltered netTrained (drop 20000 samples) dimensions BinaryOutput
    --putStrLn $ "\nAcuracia: " ++ show (checkAccuracy filteredResultsFinal) ++ " %"
    --putStrLn $ renderOutput filteredResultsFinal

    --putStrLn "\n\n\nAgora imprimindo a rede final:\n"
    --print netTrained


    --putStrLn "\n\nSalvando rede treinada em arquivo: redetreinada.tsnn...."
    --BSL.writeFile "redetreinada.tsnn" $ encode netTrained
    --putStrLn "\nCarregando rede treianda do arquivo e exibindo:"
    --byteStringNet <- BSL.readFile "redetreinada.tsnn"
    --let fileTrainedNet = deserializeNetwork byteStringNet
    --print fileTrainedNet
    --putStrLn "\nRede salva com sucesso, encerrando a execucao!"

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
