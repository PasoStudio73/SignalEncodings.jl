"""
max_nobs::Int determine how many rows to sample for the quantile estimation.
The heuristic 1000 * nbins means “allow roughly 1000 observations per bin”.
Sampling more than that rarely improves the quality of the bin edges,
so it caps the cost.
The min ensures that if the dataset is smaller than that cap,
all rows are used (no sampling losses).
"""
struct SampledQuantile <: AbstractBinningConfig
    nbins::Int
    max_nobs::Int
    rng::AbstractRNG

    function SampledQuantile(
        nbins::Int=64,
        max_nobs::Int=1000,
        rng::AbstractRNG=Xoshiro(42)
    )
        check_parameters(nbins, max_nobs)
        new(nbins, max_nobs, rng)
    end
end

function check_parameters(nbins::Int, max_nobs::Int)
    @assert 2 ≤ nbins ≤ 255 "nbins must be in [2, 255], got $nbins"
    @assert max_nobs ≥ 1 "max_nobs must be ≥ 1, got $max_nobs"
end

get_nbins(b::SampledQuantile) = b.nbins
get_max_nobs(b::SampledQuantile) = b.max_nobs
get_rng(b::SampledQuantile) = b.rng