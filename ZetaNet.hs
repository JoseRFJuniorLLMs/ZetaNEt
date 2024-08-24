import System.Random
import Data.List (foldl')
import Text.CSV
import System.IO
import Control.Monad (replicateM)

-- Tipos de dados para nossa rede neural
type Neuron = [Double]
type Layer = [Neuron]
type Network = [Layer]

-- Função de ativação (ReLU)
relu :: Double -> Double
relu x = max 0 x

-- Função para aplicar um neurônio a uma entrada
applyNeuron :: Neuron -> [Double] -> Double
applyNeuron weights inputs = relu $ sum $ zipWith (*) weights inputs

-- Função para aplicar uma camada a uma entrada
applyLayer :: Layer -> [Double] -> [Double]
applyLayer layer inputs = map (`applyNeuron` (1:inputs)) layer

-- Função para fazer uma previsão com a rede
predict :: Network -> [Double] -> [Double]
predict = foldl' (flip applyLayer)

-- Função de erro (MSE)
mse :: [Double] -> [Double] -> Double
mse predicted actual = (sum $ zipWith (\p a -> (p - a) ^ 2) predicted actual) / fromIntegral (length predicted)

-- Função para atualizar um peso
updateWeight :: Double -> Double -> Double -> Double -> Double
updateWeight rate input error weight = weight - rate * error * input

-- Função para treinar um neurônio
trainNeuron :: Double -> [Double] -> Double -> Neuron -> Neuron
trainNeuron rate inputs error weights = 
    zipWith (updateWeight rate) (1:inputs) (repeat error) weights

-- Função para treinar uma camada
trainLayer :: Double -> [Double] -> [Double] -> Layer -> Layer
trainLayer rate inputs errors layer =
    zipWith (trainNeuron rate inputs) errors layer

-- Função para treinar a rede
trainNetwork :: Double -> [Double] -> [Double] -> Network -> Network
trainNetwork rate inputs target network =
    let outputs = scanl (flip applyLayer) inputs network
        errors = zipWith (-) (last outputs) target
        deltas = foldl' (\acc layer -> 
            let newErrors = zipWith (*) acc [sum $ zipWith (*) neuron acc | neuron <- layer]
            in newErrors) errors (tail $ reverse network)
    in zipWith3 (trainLayer rate) (init outputs) (reverse deltas) network

-- Função para criar uma rede aleatória
createRandomNetwork :: [Int] -> IO Network
createRandomNetwork sizes = mapM (\(inputs, outputs) -> replicateM outputs $ replicateM (inputs + 1) randomIO) 
                            $ zip (init sizes) (tail sizes)

-- Função principal para treinar e avaliar a rede
main :: IO ()
main = do
    -- Carregar dados
    csv <- parseCSVFromFile "zeta_dataset.csv"
    let dataset = case csv of
            Right ((_:dataPts)) -> map (\[s, z] -> (read s, read z)) dataPts
            Left err -> error $ "Erro ao ler CSV: " ++ show err
    
    -- Criar rede
    network <- createRandomNetwork [1, 64, 64, 32, 1]
    
    -- Treinar rede
    let trainedNetwork = foldl' (\net (s, z) -> trainNetwork 0.001 [s] [z] net) network (take 80 dataset)
    
    -- Avaliar rede
    let testSet = drop 80 dataset
        predictions = map (\(s, _) -> head $ predict trainedNetwork [s]) testSet
        actuals = map snd testSet
        mseError = mse predictions actuals
    
    putStrLn $ "MSE: " ++ show mseError
    
    -- Salvar resultados
    writeFile "riemann_zeta_results_haskell.csv" $ unlines $
        "s,actual,predicted" : 
        zipWith (\(s, z) p -> show s ++ "," ++ show z ++ "," ++ show p) testSet predictions

    putStrLn "Resultados salvos em riemann_zeta_results_haskell.csv"