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

data Network :: * where
    O     :: !Weights
          -> Network
    (:&~) :: !Weights
          -> !Network
          -> Network
infixr 5 :&~

instance Show Network where        -- Implementacao de instancia de show de Network para facilitar o debug
  show (O a) = "Nos: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a)
  show (a :&~ b) = "Nos: " ++ show (wNodes a) ++ ", Pesos: " ++ show (wBiases a) ++ "\n" ++ show b

linear :: Floating a => a -> a   -- tende a explodir o valor maximo cabivel de um double...
linear x = x

linear' :: Floating (Vector Double) => Vector Double -> Vector Double
linear' x = Numeric.LinearAlgebra.fromList $ replicate (size  x) 1

tangente :: Floating a => a -> a 
tangente x = ((exp x) - (exp (-x))) / (((exp x) + (exp (-x))))

tangente' :: Floating a => a -> a 
tangente' x = 1 + ((tangente x) * (tangente x))

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + wN #> v

runNet :: Network -> Vector Double -> Vector Double
runNet (O w)      !v = logistic (runLayer w v)
runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
                       in  runNet n' v'



runNetCustom :: Network -> (Floating (Vector Double) => Vector Double -> Vector Double) -> Vector Double -> Vector Double  -- para usar funcs de ativacao customizadas
runNetCustom (O w)      f !v = f (runLayer w v)  -- era f, agora forcando que a ultima camada seja logistica!
runNetCustom (w :&~ n') f !v = let v' = f (runLayer w v)
                       in  runNetCustom n' f v'   -- mudar para runNetCustom


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

randomNet :: MonadRandom m => Int -> [Int] -> Int -> m Network
randomNet i []     o =     O <$> randomWeights i o
randomNet i (h:hs) o = (:&~) <$> randomWeights i h <*> randomNet h hs o

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
    go !x (O w@(W wB wN))
        = let y    = runLayer w x
              o    = logistic y
              -- the gradient (how much y affects the error)
              --   (logistic' is the derivative of logistic)
              dEdy = logistic' y * (o - target)
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (O w', dWs)
    -- handle the inner layers
    go !x (w@(W wB wN) :&~ n)
        = let y          = runLayer w x
              o          = logistic y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = logistic' y * dWs'
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (w' :&~ n', dWs)


trainCustom :: Double           -- ^ learning rate
      -> Vector Double    -- ^ input vector
      -> Vector Double    -- ^ target vector
      -> Network          -- ^ network to train
      -> (Vector Double -> Vector Double)               --  activation function!
      -> (Vector Double -> Vector Double)         -- derivative of activation function
      -> Network
trainCustom rate x0 target net f f' = fst $ go x0 net
  where
    go :: Vector Double    -- ^ input vector
       -> Network          -- ^ network to train
       -> (Network, Vector Double)
    -- handle the output layer
    go !x (O w@(W wB wN))
        = let y    = runLayer w x
              o    = f y     -- era f, tentando for'car agora que a ultima camada apenas seja logistica, e as demais possam ser F
              -- the gradient (how much y affects the error)
              --   (logistic' is the derivative of logistic)
              dEdy = f' y * (o - target)
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (O w', dWs)
    -- handle the inner layers
    go !x (w@(W wB wN) :&~ n)
        = let y          = runLayer w x
              o          = f y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = f' y * dWs'
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (w' :&~ n', dWs)

netTest :: MonadRandom m => Double -> Int -> m String
netTest rate n = do
    inps <- replicateM n $ do
      s <- getRandom
      return $ randomVector s Uniform 2 * 2 - 1
    let outs = flip map inps $ \v ->
                 if v `inCircle` (fromRational 0.33, 0.33)
                      || v `inCircle` (fromRational (-0.33), 0.33)
                   then fromRational 1
                   else fromRational 0
    net0 <- randomNet 2 [4,3] 1     -- params originais: 2 [16,8] 1
    let trained = foldl' trainEach net0 (zip inps outs)
          where
            trainEach :: Network -> (Vector Double, Vector Double) -> Network
            trainEach nt (i, o) =  trainCustom rate i o nt logistic logistic'  -- train rate i o nt

        outMat = [ [ render (norm_2 (runNet trained (vector [x / 25 - 1,y / 10 - 1])))   
                   | x <- [0..50] ]    -- init v 50
                 | y <- [0..20] ]      -- init v 20
        render r | r <= 0.2  = ' '
                 | r <= 0.4  = '.'
                 | r <= 0.6  = '-'
                 | r <= 0.8  = '='
                 | otherwise = '#'

    return $ unlines outMat
  where
    inCircle :: Vector Double -> (Vector Double, Double) -> Bool
    v `inCircle` (o, r) = norm_2 (v - o) <= r


imc3 _ = fromRational 1

imc2 vec = if (atIndex vec 0 / (a * a)) >= 25.0 then Numeric.LinearAlgebra.fromList [1.0 :: Double] else Numeric.LinearAlgebra.fromList [0.0 :: Double] -- retorna 1 se acima do peso 
            where
              a = atIndex vec 1

imc vec = if (atIndex vec 0 / (a * a)) >= 25.0 then fromRational 1 else fromRational 0 -- retorna 1 se acima do peso 
            where
              a = atIndex vec 1


randomNumber lo hi gen = head $ randomRs (lo, hi) gen

randomNumberPrint lo hi = do
                g <- newStdGen
                let result :: Double = randomNumber lo hi g
                return result


netTest2 :: (MonadRandom m, MonadIO m) => Double -> Int -> m (String, Network)     -- Tentativa de resolver IMC
netTest2 rate n = do
    inps <- replicateM n $ do
      --s <- getRandom
      g <- newStdGen
      let randomWeight :: Double = randomNumber 35.0 250.0 g
      let randomHeight :: Double = randomNumber 1.0 2.5 g
      --return $ randomVector s (enumFromThen initLimit endLimit)  2 * 2 - 2
      return $ Numeric.LinearAlgebra.fromList $ [randomWeight, randomHeight]

    --let outs = {-map Numeric.LinearAlgebra.fromList $-} map ((\x -> [x]) . imc) inps
    let outs = {-map Numeric.LinearAlgebra.fromList $-} map imc inps

    net0 <- randomNet 2 [16, 8, 4] 1 -- params originais: 2 [16,8] 1 -- aparentemente tem problemas em ter mais de um output  -- nao tinha problemas com mais de um output, mas sim com a func de ativi

    let trained = foldl' trainEach net0 (zip inps outs)
          where
            trainEach :: Network -> (Vector Double, Vector Double) -> Network
            trainEach nt (i, o) = trainCustom rate i o nt logistic logistic'

        outMat = [ [ render (( {-norm_2-}  (runNetCustom trained logistic (vector [x,y]))), x, y)     -- usando runNetCustom para poder passar func activ custom pra rodar a net
                   | x <- ([45,46 .. 150]) ]    -- init v 50    -- pesos
                 | y <- ([1.00, 1.05 .. 2.15])]      -- init v 20    -- alturas
        render (result, p, a) = "peso: " ++ show p ++ ", altura: " ++ show a ++ ", imc: " ++ show (imc (Numeric.LinearAlgebra.fromList[p,a])) ++ ", AI Result: " ++ show result

    return $ (unlines $ map unlines outMat, trained)


main :: IO ()
main = do
    args <- getArgs
    let n    = readMaybe =<< (args !!? 0)
        rate = readMaybe =<< (args !!? 1)
    putStrLn "Training network..."
    (outputS, netTrained) <- (netTest2 (fromMaybe 0.25   rate)
                                     (fromMaybe 500000 n   )   -- init value 500000
                            )
    putStrLn outputS
    putStrLn "\n\n\nAgora imprimindo a rede final:\n"
    putStrLn $ show netTrained
    {-putStrLn =<< evalRandIO (netTest2 (fromMaybe 0.25   rate)
                                     (fromMaybe 500000 n   )   -- init value 500000
                            )-}

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
