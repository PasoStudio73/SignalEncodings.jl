module Binnings

using Random: AbstractRNG, Xoshiro
using StatsBase: sample, quantile

# abstract type AbstractBinning end
abstract type AbstractBinningConfig end

export SampledQuantile
include("configs.jl")

export bin
include("binning.jl")

end
