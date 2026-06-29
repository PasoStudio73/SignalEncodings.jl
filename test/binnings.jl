using Test
using Binnings

using DataFrames, MLJ, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = MLJ.@load_iris
Xc = Matrix(DataFrame(Xc))

config = Binnings.Uniform()
X_bin, edges = bin(config, Matrix(Xc))

config = Binnings.Quantile()
X_bin, edges = bin(config, Matrix(Xc))

config = Binnings.Jenks()