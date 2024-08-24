import System.IO
import Text.Printf
import Data.List (intercalate)

-- Função para calcular zeta(s) usando a série infinita
zeta :: Double -> Double
zeta s = sum $ take 1000 [1 / (fromIntegral n ** s) | n <- [1..]]

-- Função para gerar o dataset
generateZetaDataset :: Double -> Double -> Double -> String -> IO ()
generateZetaDataset start end step filename = do
    let sValues = [start, start + step .. end]
        zetaValues = map zeta sValues
        dataPoints = zip sValues zetaValues
        csvContent = "s,zeta(s)\n" ++ 
                     intercalate "\n" [printf "%.6f,%.6f" s z | (s, z) <- dataPoints]
    
    writeFile filename csvContent
    putStrLn $ "Dataset salvo como " ++ filename

main :: IO ()
main = do
    let start_s = 1.0  -- Valor inicial de s
        end_s = 2.0    -- Valor final de s
        step_s = 0.01  -- Incremento de s
        filename = "zeta_dataset.csv"
    
    generateZetaDataset start_s end_s step_s filename