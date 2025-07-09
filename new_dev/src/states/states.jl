module States

export SiteInfo, productMPS, random_psi, boson_excitation

using ..Types: MPS

struct SiteInfo{T}
    labels::Vector{Symbol}
    basis::Symbol
  end
  
function SiteInfo{T}(N::Int; basis::Symbol=:Z, default::Symbol=:up) where T
    labels = fill(default, N)
    return SiteInfo{T}(labels, basis)
end

function _local_vec(lbl::Symbol, basis::Symbol, ::Type{T}) where T
    if basis === :Z
      return lbl===:up ? T[1.0 0.0] : T[0.0 1.0]
    elseif basis === :X
      a = one(T)/sqrt(T(2))
      return lbl===:up ? T[a a] : T[a -a]
    elseif basis === :Y
      a = one(T)/sqrt(T(2))
      return lbl===:up ? T[a -im*a] : T[a im*a]
    else
      error("Unsupported basis: $basis")
    end
end

function productMPS(st::SiteInfo{T}) where T
    N = length(st.labels)
    d = 2  # spin-½
    tensors = Vector{Array{T,3}}(undef, N)
    for i in 1:N
      v = _local_vec(st.labels[i], st.basis, T)
      # reshape into (l=1, phys=d, r=1)
      tensors[i] = reshape(v, (1, length(v), 1))
    end
    return MPS{T}(tensors)
end

function random_psi(N::Integer, chi::Integer, d::Integer, T::Type=Float64)
    tensors = Vector{Array{T,3}}(undef, N)
    tensors[1]   = rand(T, 1, d, chi)
    tensors[N]   = rand(T, chi, d, 1)
    @inbounds for i in 2:N-1
        tensors[i] = rand(T, chi, d, chi)
    end
    return MPS{T}(tensors)
end

function boson_excitation(nmax::Integer, n::Integer,T::Type=Float64)
    dB = nmax + 1
    v = zeros(T, dB)
    v[n+1] = one(T)
    return MPS{T}([reshape(v, (1, dB, 1))])
end

end #module


