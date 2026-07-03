module SignalEncodings

using Random: AbstractRNG, Xoshiro
using StatsBase: mean, sample
using Statistics: quantile

abstract type AbstractEncodingConfig end

"""
    lin_deviation(x::AbstractVector{T}) where {T<:Real}

Return the **sum of absolute deviations from the mean** of `x`:

`∑ᵢ |xᵢ - μ|`, where `μ = mean(x)`.

Useful as an L1 spread measure
(less sensitive to outliers than squared deviation).
"""
function lin_deviation(x::AbstractVector{T}) where {T<:Real}
    μ = mean(x)
    return sum(xi -> abs(xi - μ), x)
end

"""
    sq_deviation(x::AbstractVector{T}) where {T<:Real}

Return the **sum of squared deviations from the mean** of `x`:

`∑ᵢ (xᵢ - μ)^2`, where `μ = mean(x)`.

Equivalent to `length(x) * var(x)` when variance is computed with
population normalization.
"""
function sq_deviation(x::AbstractVector{T}) where {T<:Real}
    μ = mean(x)
    return sum(xi -> (xi - μ)^2, x) 
end

export Uniform, Quantile, Jenks
include("structs.jl")

export encode
include("encode.jl")

end
