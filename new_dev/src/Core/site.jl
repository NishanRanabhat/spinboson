module TNCodebase.Core.Site

using LinearAlgebra 

export spin_ops, boson_ops, AxisSpinSite, BosonSite

function spin_ops(d::Integer)
    @assert d ≥ 1 "d must be ≥ 1"
    # total spin S and its m‐values
    S = (d - 1)/2
    m_vals = collect(S:-1:-S)   # [S, S-1, …, -S]

    # Sz is just diagonal of m_vals
    Sz = Diagonal(m_vals)

    # Build S+ and S– by placing coef on the super/sub‐diagonal
    Sp = zeros(Float64, d, d)
    @inbounds for i in 1:d-1
        m_lower = m_vals[i+1]   # THIS is the m of the state being raised
        coef = sqrt((S - m_lower)*(S + m_lower + 1))
        Sp[i, i+1] = coef
    end
    Sm = Sp'  # adjoint

    # Now the cartesian components
    Sx = (Sp + Sm)/2
    Sy = (Sp - Sm) / (2im)

    return Dict(:X => Sx,
                :Y => Sy, 
                :Z => Sz,
                :Sp => Sp,
                :Sm => Sm, 
                :I => Matrix{Float64}(I, d, d))
end

function _boson_annihilator(nmax::Integer)
    @assert nmax ≥ 0 "nmax must be non-negative"
    dB = nmax + 1
    A = zeros(Float64, dB, dB)
    @inbounds for k in 1:nmax                 # super-diagonal entries
        A[k, k+1] = sqrt(k)              # √k = √(n) with n=k
    end
    return A
end

function _boson_identity(nmax::Integer)
    dB = nmax + 1
    I = zeros(Float64, dB, dB)          # or zeros(n,n) if Float64 is fine
    @inbounds for k in 1:dB                    # i ↔ n in the formula above
        I[k, k] = 1.0          # diagonal entry
    end
    return I
end 

function boson_ops(nmax::Integer)
    a    = _boson_annihilator(nmax)
    adag = a'
    return Dict(
      :a    => a,
      :adag => adag,
      :Bn    => adag * a,
      :Ib   => _boson_identity(nmax),
    )
end

"""
    AxisSpinSite(axis, d; T=Float64)

A spin‐(d−1)/2 site oriented along `axis` ∈ (:X,:Y,:Z).  
Precomputes its `up`/`down` eigenvectors.
Fields:
- `d::Int`
- `axis::Symbol`
- `up::Vector{T}`, `down::Vector{T}`
"""

struct AxisSpinSite{T}
    d::Int
    axis::Symbol
    up::Vector{T}
    down::Vector{T}
end

function AxisSpinSite(axis::Symbol, d::Int; T=Float64)
    @assert axis in (:X,:Y,:Z) "axis must be one of :X, :Y, :Z"
    ops = spin_ops(d)
    M = ops[axis]
    E = eigen(M)
    up_vec   = E.vectors[:, argmax(E.values)]
    down_vec = E.vectors[:, argmin(E.values)]
    return AxisSpinSite{T}(d, axis, up_vec, down_vec)
end

"""
    BosonSite(nmax,n; T=Float64)

A single boson site with max cotoff nmax,
returns the excited state corresponding to the excitation number n
Fields:
- `nmax::Int`
- `n::Int`
- `Bvec::Vector{T}`
"""

struct BosonSite{T}
    nmax::Int
    n::Int
    Bvec::Vector{T}
end

function BosonSite(nmax::Int,n::Int,T=Float64)
    ops = boson_ops(nmax)
    M = ops[:Bn]
    E = eigen(M)
    return BosonSite{T}(nmax,n,E.vectors[:,n+1])
end

# BosonSite already carries its |n⟩ vector
function state_tensor(site::BosonSite{T}, ::Int) where T
    reshape(site.Bvec,1,site.nmax+1,1)
end

# AxisSpinSite: use precomputed up/down
function state_tensor(site::AxisSpinSite{T}, lbl::Symbol) where T
    v = lbl==:up   ? site.up   :
        lbl==:down ? site.down :
        throw(ArgumentError("AxisSpinSite label must be :up/:down"))
    reshape(v,1,site.d,1)
end 

end #module
