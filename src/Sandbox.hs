---- Just Messing Around ----

import qualified Data.Vector as V
import Data.Vector (Vector)
import Control.Monad
import ComputationalGraph
import ComputationalGraph.VectorLayers

-- We want to print the likelihood layer as it is computed
-- For this we can wrap the "action" pass of layers in IO
-- Particularly, we want a printing sum layer
printingSum :: (Num a, Show a) => IO (Vector a) -> IO (Vector a)
printingSum x = do
    x' <- x
    y <- return $ V.singleton $ V.sum x'
    putStrLn $ show y
    return y

printingGrad :: (Show a) => IO g -> IO a -> IO g
printingGrad dy x = do
    x' <- x
    putStrLn $ show x'
    dy

-- Now that we've introduced this IO mess we need to make a printerles wrapper for other layers,
-- that's easy! It's just liftM from Control.Monad

-- Now let's construct a simple graph to be minimized:
-- Gradients are going to be vectors, activations are going to be IO vectors

-- Let's say the net is Relu into a sum
layers :: [(IO (Vector Double) -> IO (Vector Double), IO (Vector Double) -> IO (Vector Double) -> IO (Vector Double))]
layers = [(liftM reluLayer, liftM2 reluGrad), (liftM sumLayer, liftM2 $ sumGrad 4), (id, printingGrad)]

-- Let's run a test
x = return $ V.fromList [100,10,1,0.1] :: IO (Vector Double)

-- Computing this dx actually doesn't print the sum because, being last, the sum layer's result
-- never needs to be computed. We could try to add a do-nothing layer maybe?
getGrad = generalForwardBackward (return $ V.singleton 1) layers


-- And then an opt step
-- eveUpdate scale b1 b2 m v dx = change
