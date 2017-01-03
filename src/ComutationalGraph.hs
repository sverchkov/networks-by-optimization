-----------------------------------------------------------------------------
--
-- Module      :  ComutationalGraph
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

module ComutationalGraph (
    forwardBackward
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
forwardBackward [] _ = return 1 -- Most general implementation
forwardBackward ((f, df):subgraph) x = df dy x where
    dy = forwardBackward subgraph y
    y = f x
