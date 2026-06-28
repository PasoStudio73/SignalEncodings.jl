using Test
using Binnings

using DataFrames, MLJ, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = MLJ.@load_iris
Xc = DataFrame(Xc)

config = Binnings.SampledQuantile()

X_bin, edges = bin(config, Matrix(Xc))
