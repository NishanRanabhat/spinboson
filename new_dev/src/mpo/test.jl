using LinearAlgebra

# 1) A tiny abstract hierarchy
abstract type Channel end
abstract type TwoSite   <: Channel end
abstract type SingleSite<: Channel end

#Pure spin Structs

# Finite‑range coupling between i and i+dx
struct FiniteRangeCoupling <: TwoSite
    op1::String
    op2::String
    dx::Int
    weight::Float64
end

# Exponential coupling between two spins at r distance : A*e^{-b r} == amplitude*decay^r coupling
struct ExpChannelCoupling <: TwoSite
    op1::String
    op2::String
    amplitude::Float64
    decay::Float64
end

#single site field 
struct Field <: SingleSite
    op::String
    weight::Float64
end 

#Spin-Boson Structs, all the spin degrees of freedom are coupled with a single Boson 

# Finite‑range coupling between spins at i and i+dx
struct FiniteRangeCouplingSB <: TwoSite
    op1::String
    op2::String
    boson_op::String
    dx::Int
    weight_spin::Float64
    weight_spinboson::Float64
end

# Exponential coupling between two spins at r distance : A*e^{-b r} == amplitude*decay^r coupling
struct ExpChannelCouplingSB <: TwoSite
    op1::String
    op2::String
    boson_op::String
    amplitude::Float64
    decay::Float64
    weight_spinboson::Float64
end

#single site field 
struct FieldSB <: SingleSite
    op::String
    boson_op::String
    weight_spin::Float64
    weight_spinboson::Float64
end

#Boson only
struct BosonOnlySB <: SingleSite
    op::String
    weight::Float64
end


function power_law_to_exp(a::Float64,n::Integer,N::Integer)

    """
    function gives (x_i,lambda_i) such that
    1/r^a = Sum_{i=1-->n} x_i * (lambda_i)^r + errors
    a : interaction strength, a -> infinity is nearest neighbor Ising
        a -> 0 is fully connected Ising.
    n : number of exponential sums. Refer to SciPostPhys.12.4.126 appendix C
        for further details.
    N : lattice size.
    """

    F = Array{Float64,1}(undef,N)
    @inbounds for k in 1:N
        F[k] = 1/k^a
    end

    M = zeros(N-n+1,n)
    @inbounds for j in 1:n
        @inbounds for i in 1:N-n+1
            M[i,j] = F[i+j-1]
        end
    end

    F1 = qr(M)
    Q1 = F1.Q[1:N-n,1:n]
    Q1_inv = pinv(Q1)
    Q2 = F1.Q[2:N-n+1,1:n]
    V = Q1_inv*Q2

    lambda = real(eigvals(V))
    lam_mat = zeros(N,n)
    @inbounds for i in 1:length(lambda)
        @inbounds for k in 1:N
            lam_mat[k,i] = lambda[i]^k
        end
    end
    nu = lam_mat\F
    return nu, lambda
end

#builds finite state machine from different channels
function spin_FSM(channels::Vector{<:Channel})
    #start the finite state machine with 1 as first idle
    ns = 1
    #allocate a vector of tuples for hosting transitions
    transitions = Tuple{Int,Int,String,Float64}[]
    #put identity at first and final idle 
    push!(transitions, (1,1,"I", 1.0))
    push!(transitions, (0,0,"I", 1.0))

    #loop over different coupling channels and build transitions
    for ch in channels
        ns, transitions = build_path(ns, ch, transitions)
    end
    return ns+1, transitions
end

# for finite‐range couplings
function build_path(ns::Int, coupling::FiniteRangeCoupling,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1 : ns+coupling.dx
    # idle → first state : emit op1
    push!(transitions, (path[1], 1, coupling.op1, 1.0))
    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1], "I", 1.0))
    end
    # last state → final idle : emit op2
    push!(transitions, (0, path[end], coupling.op2, coupling.weight))
    return path[end], transitions
end

# for exponential couplings
function build_path(ns::Int, coupling::ExpChannelCoupling,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns + 1
    # idle → first state : emit op1
    push!(transitions, (path, 1, coupling.op1,1.0))
    # self‑loop with identity and decay 
    push!(transitions, (path, path, "I", coupling.decay))
    # last state → final idle : emit op2
    push!(transitions, (0,    path, coupling.op2,
            coupling.amplitude * coupling.decay))
    return path, transitions
end

# for single‐site fields
function build_path(ns::Int, coupling::Field,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    # first idle → last : emit op
    push!(transitions, (0, 1, coupling.op, coupling.weight))
    return ns, transitions
end

# --- Example usage ---
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -1.0*im; 1.0*im  0.0]
sz = [1.0 0.0; 0.0  -1.0]
I2 = Matrix{Float64}(I, 2, 2)

channels = [FiniteRangeCoupling("X", "X", 2,0.5),ExpChannelCoupling("Z", "Z",0.8, 0.5),Field("Y",0.8)]
ops = Dict("X"=>sx, "Y"=>sy, "Z"=>sz,"I"=>I2)
chi, transitions = spin_FSM(channels)
println(transitions)

