{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}


module Main where

--import Lib

import Control.Monad
import Control.Monad.Random
--import Data.Vector.Storable (basicLength)
import Data.List
import GHC.Float
import Data.Maybe
import Numeric.LinearAlgebra
import System.Environment
import Text.Read

data Weights = W { wBiases :: !(Vector Double)  -- n
                 , wNodes  :: !(Matrix Double)  -- n x m
                 }                              -- "m to n" layer


data Activation = Linear | Logistic | Tangent deriving Show

getFunctions :: Floating a => Activation -> ((a -> a), (a -> a))
getFunctions f = case f of
                  Linear   -> (linear, linear')
                  Logistic -> (logistic, logistic')
                  Tangent  -> (tangent, tangent')

{-}
data Network :: * where
    O     :: !Weights
          -> Network
    (:&~) :: !Weights
          -> !Network
          -> Network
infixr 5 :&~
-}

data Network :: * where
    O     :: !Activation -> !Weights
          -> Network
    (:&~) :: (Activation , Weights)
          -> !Network
          -> Network
infixr 5 :&~


instance Show Network where        -- Implementacao de instancia de show de Network para facilitar o debug
  show (O f a)          =  "Nos de saida: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f
  show ((f , a) :&~ b)  =  "Nos camada: "   ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ ", Funcao de Ativacao: " ++ show f ++ "\n" ++ show b


-- Auxiliar definition of activation functions and it's derivatives

linear :: Floating a => a -> a   
linear x = x

linear' :: Floating a => a -> a
linear' _ =  1

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

linear2 :: (Floating a, Eq a) => a -> a    -- tentativa de criar uma activacao linear
linear2 x = if y == (1/0) then 0 else y
            where
             y = (x * x) + 3

linear2' :: (Floating a, Eq a) => a -> a
linear2' x = if y == (1/0) then 0 else y
            where
              y = 2 * x

tangent :: Floating a => a -> a
tangent x = ((exp x) - (exp (-x))) / (((exp x) + (exp (-x))))

tangent' :: Floating a => a -> a
tangent' x = 1 + ((tangent x) * (tangent x))








runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + wN #> v

runNet :: Network -> Vector Double -> Vector Double
runNet (O f w)      !v = let (function, _) = getFunctions f
                          in function (runLayer w v)
runNet ((f,w) :&~ n') !v = let 
                              (function, _) = getFunctions f
                              v' = function (runLayer w v)
                            in  runNet n' v'

arredonda :: (Ord a1, Fractional a1, Fractional a2) => a1 -> a2
arredonda x
  | x > 0.5   = fromRational 1
  | otherwise = fromRational 0

derive :: (Fractional a) => a -> (a -> a) -> (a -> a)
derive h f x = (f (x+h) - f x) / h 
                                      

vectorRandomico :: MonadRandom m => Int -> m (Vector Double)
vectorRandomico x = do
                      seed :: Int <- getRandom
                      let vec = randomVector seed Uniform x
                      return vec

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

{-
randomNet2 :: MonadRandom m => Int -> Activation -> [(Int, Activation)] -> Int -> m Network    --  randomNet2 2 linear [(3, linear)] 1
randomNet2 i f []          o =  O2 f <$> randomWeights i o     --O2     <$> randomWeights i o $ f
randomNet2 i f ((h,f2):hs) o =  (:&~~) f <$> randomWeights i h <*> randomNet2 h f2 hs o
-}

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
              --   (logistic' is the derivative of logistic)
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


imc3 :: Fractional a => p -> a
imc3 _ = fromRational 1

imc2 :: (Container c a, Ord a, Fractional a, Num (IndexOf c)) => c a -> Vector Double
imc2 vec = if (atIndex vec 0 / (a * a)) >= 25.0 then Numeric.LinearAlgebra.fromList [1.0 :: Double] else Numeric.LinearAlgebra.fromList [0.0 :: Double] -- retorna 1 se acima do peso 
            where
              a = atIndex vec 1

imc :: (Container c a1, Ord a1, Fractional a1, Fractional a2, Num (IndexOf c)) => c a1 -> a2
imc vec = if (atIndex vec 0 / (a * a)) >= 25.0 then fromRational 1 else fromRational 0 -- retorna 1 se acima do peso 
            where
              a = atIndex vec 1


randomNumber :: (Random a, RandomGen g) => a -> a -> g -> a
randomNumber lo hi gen = head $ randomRs (lo, hi) gen

randomNumberPrint :: MonadIO m => Double -> Double -> m Double
randomNumberPrint lo hi = do
                g <- newStdGen
                let result :: Double = randomNumber lo hi g
                return result

-- Auxiliar functions to apply to samples
lastN :: Int -> [a] -> [a]
lastN n xs = drop (length xs - n) xs


-- atualizar para versao final de treino de rede: receber entradas E saidas, receber modelo inicial de rede construido fora da funcao de treino!
netTest4 :: (MonadRandom m, MonadIO m) => Network -> Double -> Int -> [[Double]] -> (Int, Int) -> m (Network, String, Network, String)     -- Tentativa de resolver IMC  -- agora com input de arquivo txt como entrada de samples!
netTest4 initnet learningrate nruns samples (inputD, outputD) = do

    let inps = map Numeric.LinearAlgebra.fromList $ map (take inputD) samples
    --let outs = {-map Numeric.LinearAlgebra.fromList $-} map imc inps
    let outs = map Numeric.LinearAlgebra.fromList $ map (lastN outputD) samples

    -- net0 <- randomNet 2 [20] 1 -- params originais: 2 [16,8] 1 -- aparentemente tem problemas em ter mais de um output  -- nao tinha problemas com mais de um output, mas sim com a func de ativi

    --let trained = foldl' trainEach net0 (zip inps outs)
    let trained = trainNTimes initnet (inps, outs) nruns
          where
            trainNTimes :: Network -> ([Vector Double], [Vector Double]) -> Int -> Network
            trainNTimes net (i, o) n2
                | n2 == 0 = net
                | otherwise = trainNTimes (foldl' trainEach net (zip i o)) (i, o) (n2 - 1)
                        where
                            trainEach :: Network -> (Vector Double, Vector Double) -> Network
                            trainEach nt (i2, o2) = train learningrate i2 o2 nt

        outMat = [ [ render (( (take inputD x), (lastN outputD x), (runNet trained (vector (take inputD x)))))     -- usando runNetCustom para poder passar func activ custom pra rodar a net
                   | x <- samples ] ]   -- init v 50    -- pesos    -- [45,46 .. 150]
                 -- | y <- (map (head . tail) samples)]      -- init v 20    -- alturas   -- [1.00, 1.05 .. 2.15]
        outMatInit = [ [ render (( (take inputD x), (lastN outputD x), (runNet initnet (vector ((take inputD x))))))
                       | x <- samples ] ]

        render (inputs, outputs, netResult) = "Inputs: " ++ show inputs ++ ", Expected Outputs: " ++ show outputs ++ ", Neural Network Results: " ++ show netResult

    return $ (initnet, unlines $ map unlines outMatInit, trained, unlines $ map unlines outMat)


stringToSamples :: String -> [[Double]]
stringToSamples x = map (map readSamples) $ map words $ lines x
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
    putStrLn ( show $ take 50 samples)
    let inputD = 2 :: Int
    let outputD = 1 :: Int
    let dimensions = (inputD, outputD) :: (Int, Int)

    testNet <- randomNet inputD Logistic [(20, Logistic), (5, Linear), (10, Logistic)] outputD

    initialNet <- randomNet inputD Logistic [(20, Logistic)] outputD

    --let outputs = map imc samples
    putStrLn "\n\nTraining network..."
    {-(outputS, netTrained) <- (netTest2 (fromMaybe 0.25   rate)
                                     (fromMaybe 2 n   )   -- init value 500000
                            )-}
    {-(netInit, outputInit, netTrained, outputS) <- (netTest3 (fromMaybe 0.000025   rate)
                                     (fromMaybe 100000 n   )   -- init value 500000
                                     samples
                            )-}
    (netInit, outputInit, netTrained, outputS) <- netTest4 initialNet
                                 (fromMaybe 0.00025   rate)
                                 (fromMaybe 10000 n   )   -- init value 500000
                                 samples
                                 dimensions

    putStrLn "\n\n\nImprimindo predicao nao treinada:\n"
    putStrLn outputInit
    putStrLn "\n\n\nImprimindo predicao agora treinada:\n"
    putStrLn outputS
    putStrLn "\n\n\nImprimindo a rede inicial:\n"
    putStrLn $ show netInit
    putStrLn "\n\n\nAgora imprimindo a rede final:\n"
    putStrLn $ show netTrained
    {-putStrLn =<< evalRandIO (netTest2 (fromMaybe 0.25   rate)
                                     (fromMaybe 500000 n   )   -- init value 500000
                            )-}

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
