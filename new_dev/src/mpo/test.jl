using LinearAlgebra 

include(joinpath(@__DIR__, "fsm.jl"))

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

    return Dict("X" => Sx,
                "Y" => Sy, 
                "Z" => Sz, 
                "I" => Matrix{Float64}(I, d, d))
end

function boson_annihilator(nmax::Integer)
    @assert nmax ≥ 0 "nmax must be non-negative"
    dB = nmax + 1
    A = zeros(Float64, dB, dB)
    @inbounds for k in 1:nmax                 # super-diagonal entries
        A[k, k+1] = sqrt(k)              # √k = √(n) with n=k
    end
    return A
end

function boson_identity(nmax::Integer)
    dB = nmax + 1
    I = zeros(Float64, dB, dB)          # or zeros(n,n) if Float64 is fine
    @inbounds for k in 1:dB                    # i ↔ n in the formula above
        I[k, k] = 1.0          # diagonal entry
    end
    return I
end 

function boson_ops(nmax::Integer)
    a    = boson_annihilator(nmax)
    adag = a'
    return Dict(
      "a"    => a,
      "adag" => adag,
      "n"    => adag * a,
      "Ib"   => boson_identity(nmax),
    )
end

# Your MPO type from before
abstract type TensorNetwork{T} end
struct MPO{T} <: TensorNetwork{T}
    tensors::Vector{Array{T,4}}
end

# ————————————————————————————————————————————————————————————————
# 1) Pure-spin MPO, now takes `N` sites
# ————————————————————————————————————————————————————————————————
function build_mpo(
    fsm::SpinFSMPath;
    N::Integer,
    d::Integer = 2,
    T::Type   = Float64,
)
    # operator factory
    phys_ops = spin_ops(d)
    chi        = fsm.chi

    # build the bulk
    bulk = zeros(T, chi, chi, d, d)
    for (row,col,opname,w) in fsm.transitions 
        op_mat = phys_ops[opname]
        bulk[row==0 ? chi : row,col == 0 ? chi : col,:,:] += w*op_mat 
    end 

    # left / right boundaries
    L = reshape(bulk[chi, :, :, :], (1, chi, d, d))
    R = reshape(bulk[:, 1, :, :], (chi, 1, d, d))

    # assemble N‐site MPO: [L, bulk, bulk, …, bulk, R]
    mids = fill(bulk, N-2)        # N-2 copies of the central tensor
    tensors = [L, mids..., R]     # vector of length N

    return MPO{T}(tensors)
end

#build the mpo builder for spin boson
function build_mpo(fsm::SpinBosonFSMPath;d::Integer=2,nmax::Integer=4,T::Type=Float64)
    chi = fsm.chi
    grid_bulk = zeros(T,chi-1,chi-1,d,d)
    grid_L = zeros(T,1,chi-1,nmax+1,nmax+1)
    phys_ops = merge(spin_ops(d),boson_ops(nmax))
    for (row,col,opname,w) in fsm.transitions 
        op_mat = phys_ops[opname]
        if row == 0
            grid_L[1,col,:,:] += w*op_mat
        else
            grid_bulk[row,col,:,:] += w*op_mat
        end 
    end
    return grid_L, grid_bulk, reshape(grid_bulk[:,1,:,:],(chi-1,1,d,d))
end

# ————————————————————————————————————————————————————————————————
# 2) Spin-boson MPO,
# ————————————————————————————————————————————————————————————————
function build_mpo1(
    fsm::SpinBosonFSMPath;
    N::Integer,
    d::Integer    = 2,
    nmax::Integer = 4,
    T::Type       = Float64,
)
    chi   = fsm.chi
    db  = nmax + 1
    phys_ops = merge(spin_ops(d), boson_ops(nmax))
    # build the “bulk” and left boundary
    bulk  = zeros(T, chi-1, chi-1, d, d)
    L     = zeros(T, 1, chi-1, db, db)
    for (row,col,opname,w) in fsm.transitions 
        op_mat = phys_ops[opname]
        if row == 0
            L[1,col,:,:] .+= w*op_mat
        else
            bulk[row,col,:,:] .+= w*op_mat
        end 
    end

    # right boundary
    R = reshape(bulk[:, 1, :, :], (chi-1, 1, d, d))

    # assemble N‐site MPO
    mids    = fill(bulk, N-2)
    tensors = [L, mids..., R]

    return MPO{T}(tensors)
end

spin1 = FiniteRangeCoupling("X", "X", 1,1.0)
spin2 = ExpChannelCoupling("Z", "Z",0.8, 1.0)
spin3 = Field("Z",1.0)
spin4 = Field("X",1.0)
channel_spin = [spin1,spin3]
channel_boson = [FiniteRangeCouplingSB(spin1,"Ib",1.0),FieldSB(spin3,"a",1.0),BosonOnlySB("n",1.0)]

fsm1 = spin_FSM(channel_spin)
fsm2 = spinboson_FSM(channel_boson)

L,M,R = build_mpo(fsm2,d=2,nmax=4)

mpo = build_mpo1(fsm2,N=6,d=2,nmax=4)




