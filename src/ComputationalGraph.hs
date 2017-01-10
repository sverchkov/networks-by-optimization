-----------------------------------------------------------------------------
--
-- Module      :  ComputationalGraph
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

module ComputationalGraph
( forwardBackward
, generalForwardBackward
) where

-- | Forward-backward pass
--
-- Takes in the activations at the bottom of the graph, and returns the
-- gradient, where we assume the top of the graph always comes together to a
-- single number.
--
-- A graph is a list of layers, and each layer is a pair of functions,
-- f : x -> y
-- df : dy -> x -> dx (at x)
-- where x, y, dx, dy are (conceptually) vectors
--
-- To have a maximally general implementation, "vector" is generalized to
-- a monad (v), and gradients are greneralized to a numeric (g). We don't
-- restict the activations (a) at all.
forwardBackward :: (Num g, Monad v) => [(a -> a, v g -> a -> v g)] -> a -> v g
forwardBackward = generalForwardBackward (return 1)

-- | The general forward-backward pass
-- This implementation takes a parameter that corresponds to the "unity" gradient
-- that a final layer would return. This makes it easier to plug in different
-- types.
generalForwardBackward :: g -> [(a -> a, g -> a -> g)] -> a -> g
generalForwardBackward one [] _ = one
generalForwardBackward one ((f, df):subgraph) x = df dy x where
    dy = generalForwardBackward one subgraph y
    y = f x
