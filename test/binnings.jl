using Test
using Binnings

using DataFrames, MLJ, Random

using Jenks

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = MLJ.@load_iris
Xc = Matrix(DataFrame(Xc))

config = Binnings.Uniform(;nbins=16)
X_bin, edges = bin(config, Matrix(Xc))

config = Binnings.Quantile()
X_bin, edges = bin(config, Matrix(Xc))

config = Binnings.Jenks(;nbins=16, float_type=Float32)
a=bin(config, Vector(Xc[:,1]))

# for g in eachcol(Xc)
R = JenksClassification(16,Vector(Xc[:,1]))
@show R.breaks
# end

# breaks = [1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 109, 118, 127, 136, 151]
R.breaks = [1, 8, 13, 21, 36, 46, 55, 64, 76, 84, 96, 108, 120, 135, 141, 145, 151]
