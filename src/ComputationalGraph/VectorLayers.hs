-----------------------------------------------------------------------------
--
-- Module      :  ComputationalGraph.VectorLayers
-- Copyright   :
-- License     :  AllRightsReserved
--
-- Maintainer  :
-- Stability   :
-- Portability :
--
-- |
--
-----------------------------------------------------------------------------

module ComputationalGraph.VectorLayers
( distribute
, distributeGrad
, sumLayer
, sumGrad
, reluLayer
, reluGrad
) where

import qualified Data.Vector as V
import Data.Vector ((!), Vector)

-- | Distributive layer using vectors
-- >>> distribute (V.fromList [0,1,1,0]) (V.fromList [5,10])
-- [5,10,10,5]
distribute :: V.Vector Int -> V.Vector a -> V.Vector a
distribute inList x = V.backpermute x inList

-- >>> distributeGrad (V.fromList [[0,3],[1,2]]) (V.fromList [1,2,3,4.1]) 7
-- [5.1,5.0]
distributeGrad :: Num a => V.Vector [Int] -> V.Vector a -> t -> V.Vector a
distributeGrad outList dy _ = V.generate l df where
    l = length outList
    df i = sum [dy!j | j <- outList!i]

-- | Sum layer using vectors
-- forward layer is just sum
sumLayer :: Num a => Vector a -> Vector a
sumLayer = V.singleton . V.sum
-- backward layer just distributes the input gradient (must be singleton)
sumGrad :: Num g => Int -> Vector g -> a -> Vector g
sumGrad n dy _ = V.replicate n (dy!0)

-- | Relu
reluLayer :: (Num a, Ord a) => Vector a -> Vector a
reluLayer = V.map (max 0)

reluGrad :: (Num g, Num a, Ord a) => Vector g -> Vector a -> Vector g
reluGrad = V.zipWith (\grad act -> if act > 0 then grad else 0)

-- | Pointwise multiply
vectorMultiply :: Num a => Vector a -> Vector a -> Vector a
vectorMultiply = V.zipWith (*)

pwMultiplyGrad :: Num g => Vector g -> Vector g -> a -> Vector g
pwMultiplyGrad c dy _ = vectorMultiply c dy

-- | Multiply fixed scalar
scaleLayer :: Num a => a -> Vector a -> Vector a
scaleLayer s = V.map (* s)

scaleGrad :: Num g => g -> Vector g -> a -> Vector g
scaleGrad s dy _ = V.map (* s) dy

-- | Add fixed scalar
biasLayer :: Num a => a -> Vector a -> Vector a
biasLayer b = V.map (+ b)

biasGrad :: g -> a -> g
biasGrad dy _ = dy

-- | Block-heterogeneous vector
-- wrapper for a blockwise heterogenous layer
blockActivationLayer :: [Int] -> [Vector a -> Vector b] -> Vector a -> Vector b
blockActivationLayer inds blocks x = V.concat ys where
    ys = [f act | (f, act) <- zip blocks xs]
    xs = multiSlice inds x

blockGradientLayer :: [Int] -> [Vector a -> Vector b -> Vector c] -> Vector a -> Vector b -> Vector c
blockGradientLayer inds blocks dy x = V.concat dxs where
    dxs = [df dy act | (df, dy, act) <- zip3 blocks dys xs]
    dys = multiSlice inds dy
    xs = multiSlice inds x

-- | Slice a list in multiple places
-- >>> multiSlice [4,8,9] $ V.fromList [0.1,0.2,0.4,0.11,0.21,11,22,33,44]
-- [[0.1,0.2,0.4,0.11],[0.21,11.0,22.0,33.0],[44.0]]
multiSlice :: [Int] -> Vector a -> [Vector a]
multiSlice inds x = [V.slice i n x | (i, n) <- slices] where
    iinds = init $ 0:inds
    slices = zip iinds $ zipWith (-) inds iinds
