from python import Python
from algorithm import parallelize
from math import sqrt, pow
from sys.info import num_cores
from time import now

struct ZetaParams:
    var start: Float64
    var end: Float64
    var step: Float64

fn zeta(s: Float64) -> Float64:
    var sum: Float64 = 0.0
    var n: Int = 1
    var term: Float64 = 1.0 / pow(n.cast[Float64](), s)
    
    while term > 1e-10:
        sum += term
        n += 1
        term = 1.0 / pow(n.cast[Float64](), s)
    
    return sum

fn generate_zeta_chunk(params: ZetaParams, start_idx: Int, end_idx: Int) -> PythonObject:
    let np = Python.import_module("numpy")
    let pd = Python.import_module("pandas")
    
    var s_values = np.arange(params.start + start_idx * params.step, 
                             params.start + end_idx * params.step, 
                             params.step)
    var zeta_values = np.array([zeta(s) for s in s_values])
    
    return pd.DataFrame({"s": s_values, "zeta(s)": zeta_values})

fn generate_zeta_dataset(start: Float64, end: Float64, step: Float64, filename: String):
    let params = ZetaParams(start, end, step)
    let num_points = ((end - start) / step).cast[Int]()
    let chunk_size = (num_points + num_cores() - 1) // num_cores()
    
    var results = parallelize[generate_zeta_chunk, PythonObject](params, num_points, chunk_size)
    
    let pd = Python.import_module("pandas")
    var df = pd.concat(results)
    df.to_csv(filename, index=False)
    print("Dataset saved as", filename)

fn main():
    let start_time = now()
    
    let start_s: Float64 = 1.0
    let end_s: Float64 = 2.0
    let step_s: Float64 = 0.01
    let filename = "zeta_dataset.csv"
    
    generate_zeta_dataset(start_s, end_s, step_s, filename)
    
    let end_time = now()
    print("Execution time:", (end_time - start_time) / 1e9, "seconds")

main()