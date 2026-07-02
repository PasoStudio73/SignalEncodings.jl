using Test
using SignalEncodings

using SoleData
using DataFrames, StableRNGs, MLJ

# ---------------------------------------------------------------------------- #
#                                  utilities                                   #
# ---------------------------------------------------------------------------- #
function _collect_numbers(x, acc = Float64[])
    if x isa Number
        push!(acc, float(x))
    elseif x isa AbstractArray || x isa Tuple
        for v in x
            _collect_numbers(v, acc)
        end
    end
    return acc
end

function _count_numbers(x)
    if x isa Number
        return 1
    elseif x isa AbstractArray || x isa Tuple
        s = 0
        for v in x
            s += _count_numbers(v)
        end
        return s
    end
    return 0
end

function _assert_encoding_output(X_bin, edges, X_in, nbins::Int)
    # Some backends return feature-wise containers instead of preserving outer shape.
    @test _count_numbers(X_bin) == _count_numbers(X_in)

    bvals = _collect_numbers(X_bin)
    @test !isempty(bvals)
    @test all(isfinite, bvals)

    bints = Int.(round.(bvals))
    @test all(==(0), abs.(bvals .- bints))  # bins should be integral ids

    bmin, bmax = extrema(bints)
    @test bmin >= 0
    @test bmax <= nbins + 1
    @test length(unique(bints)) <= nbins + 1

    evals = _collect_numbers(edges)
    @test !isempty(evals)
    @test all(isfinite, evals)
end

# ---------------------------------------------------------------------------- #
#                                   tabular                                    #
# ---------------------------------------------------------------------------- #
@testset "encoding: tabular" begin
    Xc, yc = @load_iris
    Xc = Matrix(DataFrame(Xc))

    for config in (
        SignalEncodings.Uniform(; nbins = 16),
        SignalEncodings.Quantile(; nbins = 16),
        SignalEncodings.Jenks(; nbins = 16),
    )
        X_bin, edges = encode(config, Xc)
        _assert_encoding_output(X_bin, edges, Xc, 16)
    end
end

# ---------------------------------------------------------------------------- #
#                                time series                                   #
# ---------------------------------------------------------------------------- #
@testset "encoding: time series" begin
    natopsloader = SoleData.Artifacts.NatopsLoader()
    Xts, yts = SoleData.Artifacts.load(natopsloader)
    Xts = Matrix(Xts)

    for config in (
        SignalEncodings.Uniform(; nbins = 32),
        SignalEncodings.Quantile(; nbins = 32),
        SignalEncodings.Jenks(; nbins = 32),
    )
        X_bin, edges = encode(config, Xts)
        _assert_encoding_output(X_bin, edges, Xts, 32)
    end
end

# ---------------------------------------------------------------------------- #
#                                   images                                     #
# ---------------------------------------------------------------------------- #
@testset "encoding: images" begin
    rng = StableRNG(42)
    X = [round.(rand(rng, Float32, 2, 2); digits = 2) for _ in 1:3, _ in 1:2]

    for config in (
        SignalEncodings.Uniform(; nbins = 3),
        SignalEncodings.Quantile(; nbins = 3),
        SignalEncodings.Jenks(; nbins = 3),
    )
        X_bin, edges = encode(config, X)
        _assert_encoding_output(X_bin, edges, X, 3)
    end
end

# ---------------------------------------------------------------------------- #
#                             multi dimensional                                #
# ---------------------------------------------------------------------------- #
@testset "encoding: multi-dimensional tensors" begin
    rng = StableRNG(42)
    X4 = [round.(rand(rng, Float32, 3, 2, 2); digits = 2)
        for _ in 1:3, _ in 1:2]

    for config in (
        SignalEncodings.Uniform(; nbins = 3),
        SignalEncodings.Quantile(; nbins = 3),
        SignalEncodings.Jenks(; nbins = 3),
    )
        X_bin, edges = encode(config, X4)
        _assert_encoding_output(X_bin, edges, X4, 3)
    end
end
