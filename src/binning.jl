function get_idxs(
    x::AbstractVector{T},
    max_nobs::Int,
    nbins::UInt8,
    rng::AbstractRNG
) where {T<:Real}
    nsamples = length(x)
    nobs = min(nsamples, max_nobs * nbins)
    return nobs < nsamples ?
        sample(rng, 1:nsamples, nobs, replace=false, ordered=true) :
        collect(1:nsamples)
end

function lin_deviation(x::AbstractVector{T}, ::Type{S}) where {S<:AbstractFloat,T<:Real}
    μ = mean(x)
    return sum(xi -> S(abs(xi - μ)), x)
end

function sq_deviation(x::AbstractVector{T}, ::Type{S}) where {S<:AbstractFloat,T<:Real}
    μ = mean(x)
    return sum(xi -> S((xi - μ)^2), x) 
end

function bin(config::Uniform{S}, x::AbstractVector{T}) where {S<:Float,T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = collect(range(minimum(x), maximum(x); length=nbins))
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = searchsortedfirst.(Ref(edges), x)

    return S.(edges), x_bin
end

function bin(config::Quantile{S}, x::AbstractVector{T}) where {S<:Float,T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)
    alpha, beta = get_alpha(config), get_beta(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = quantile(view(x, idxs), (1:nbins-1) / nbins; alpha, beta)
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = searchsortedfirst.(Ref(edges), x)

    return S.(edges), x_bin
end

function bin(config::Jenks{S}, x::AbstractVector{T}) where {S<:Float,T<:Real}
    nbins, maxiter = get_nbins(config), get_maxiter(config)
    fluxadjust_bothways = get_fluxadjust_bothways(config)
    fluxadjust = get_fluxadjust(config)
    deviation = get_deviation(config)

    sort!(x)
    ndata = length(x)

    np = ndata ÷ nbins
    breaks = [1:np:np*nbins; ndata+1]
    
    fluxes = fill(get_flux(config),nbins-1)
    devs = Vector{S}(undef,nbins)
    devs_pre = Vector{S}(undef,nbins)

    for iter in 1:maxiter
        devs_pre .= devs

        for iclass in 1:nbins
            devs[iclass] =
                deviation(
                    @view(x[breaks[iclass]:breaks[iclass+1]-1]), S
                )
        end

        for iclass in 1:nbins-1
            a, b, c = breaks[iclass:iclass+2]
            if devs[iclass] < devs[iclass+1]
                if iter > 1
                    if devs_pre[iclass] >= devs_pre[iclass+1]
                        fluxes[iclass] /= fluxadjust
                    elseif fluxadjust_bothways
                        fluxes[iclass] *= fluxadjust
                    end
                end

                cs_factor = Int(round((b-a)*fluxes[iclass]))
                newbreak = breaks[iclass+1]+cs_factor
                breaks[iclass+1] = min(newbreak,breaks[iclass+2]-2)
            else

                if iter > 1
                    if devs_pre[iclass] <= devs_pre[iclass+1]
                        fluxes[iclass] /= fluxadjust
                    elseif fluxadjust_bothways
                        fluxes[iclass] *= fluxadjust
                    end
                end

                cs_factor = Int(round((c-b)*fluxes[iclass]))
                newbreak = breaks[iclass+1]-cs_factor
                breaks[iclass+1] = max(newbreak,breaks[iclass]+2)
            end
        end
    end

    breaks
    # edges = collect(range(minimum(x), maximum(x); length=nbins))
    # length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    # x_bin = searchsortedfirst.(Ref(edges), x)

    # return S.(edges), x_bin
end

function bin(
    config::BinningConfig{S},
    X::AbstractArray{T}
) where {S<:Float,T<:Real}
    nfeats = size(X, 2)
    edges = Vector{Vector{S}}(undef, nfeats)
    X_bin = Vector{Vector{UInt8}}(undef, nfeats)

    Threads.@threads for j in 1:nfeats
        edges[j], X_bin[j] = bin(config, view(X, :, j))
    end

    return X_bin, edges
end

function bin(
    config::BinningConfig{S},
    X::Matrix{<:AbstractArray{T}}
) where {S<:Float,T<:Real}

end