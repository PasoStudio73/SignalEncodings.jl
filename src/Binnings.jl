module Binnings

using Random: AbstractRNG, Xoshiro
using StatsBase: mean, sample
using Statistics: quantile

# abstract type AbstractBinning end
abstract type AbstractBinningConfig end

# abstract type AbstractAlphaBetaConfig <: AbstractBinningConfig end

export Uniform, Quantile, Jenks
include("jenks.jl")
include("structs.jl")

# export bin
export bin
include("binning.jl")

end
