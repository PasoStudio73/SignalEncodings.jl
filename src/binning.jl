function bin(config::SampledQuantile, X::Matrix{T}) where {T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)

    nsamples, nfeats = size(X)
    nobs = min(nsamples, max_nobs * nbins)
    idxs = nobs < nsamples ?
        sample(rng, 1:nsamples, nobs, replace=false, ordered=true) :
        collect(1:nsamples)

    # edges = Vector{Vector{T}}(undef, nfeats)
    edges = Vector{Vector{T}}(undef, nfeats)
    # featbins = Vector{UInt8}(undef, nfeats)
    # feattypes = trues(nfeats) # forse non serve, sono tutti uni
    X_bin = Matrix{UInt8}(undef, nsamples, nfeats)

    Threads.@threads for j in 1:nfeats
        edges[j] = quantile(view(X, idxs, j), (1:nbins-1) / nbins)
        length(edges[j]) == 1 && (edges[j] = [minimum(view(X, idxs, j))])
        # featbins[j] = length(edges[j]) + 1
        X_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end

    # return edges, featbins, feattypes
    return X_bin, edges
end
