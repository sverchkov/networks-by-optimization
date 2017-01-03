---- Just Messing Around ----

import qualified Data.Vector as V
import Data.Vector (Vector)

-- We want to print the likelihood layer as it is computed
-- For this we can wrap the "action" pass of layers in IO
-- Particularly, we want a printing sum layer
printingSum :: (Num a, Show a) => IO (Vector a) -> IO (Vector a)
printingSum x = let
    in do
    x' <- x
    y <- return $ V.singleton $ V.sum x'
    putStrLn $ show y
    return y

-- Now that we've introduced this IO mess we need to make a printerles wrapper for other layers
ioWrapper :: (a -> a) -> IO a -> IO a
ioWrapper f x = do
    x' <- x
    return $ f x'
