-----------------------------------------------------------------------------
--
-- Module      :  Optimizer
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

module Optimizer (

) where

import Data.Vector (Vector, (!))
import qualified Data.Vector as V

---- Gradient update methods ---

-- | Standard Gradient
gradientUpdate :: Num a => a -> Vector a -> Vector a
gradientUpdate rate = V.map (rate *)

-- | Nesterov Gradient
nesterovUpdate ::
    Num a =>
    a -> a -> Vector a -> Vector a -> Vector a -> (Vector a, Vector a)
nesterovUpdate rate mu v0 dx = (v, update) where
    v = V.zipWith (-) (V.map (mu *) v) (V.map (rate *) dx)
    update =
        V.zipWith (+) (V.map ((-mu) *) v0) (V.map ((1 + mu) *) v)

-- | Adam-inspired Update
eveUpdate scale b1 b2 m v dx = (m', v', update) where
    update = V.map (scale *) $ V.zipWith (/) m' (V.map sqrt v')
    m' = V.zipWith (+) (V.map (b1 *) m) (V.map ((1 - b1) *) dx)
    v' = V.zipWith (+) (V.map (b2 *) v) (V.map (\g -> (1 - b2) * g^2) dx)
