using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# 1) Abstract hierarchy
# ──────────────────────────────────────────────────────────────────────────────

abstract type Channel end
abstract type TwoSite    <: Channel end
abstract type SingleSite <: Channel end

# ──────────────────────────────────────────────────────────────────────────────
# 2) Pure-spin channel structs
# ──────────────────────────────────────────────────────────────────────────────

"""
    FiniteRangeCoupling(op1, op2, dx, weight)

Finite-range coupling between spins at sites i and i+dx:

- `op1`, `op2` : names of the operators on the two spins  
- `dx`         : distance between the spins  
- `weight`     : coupling strength  
"""
struct FiniteRangeCoupling <: TwoSite
    op1::String
    op2::String
    dx::Int
    weight::Float64
end

"""
    ExpChannelCoupling(op1, op2, amplitude, decay)

Exponential two-spin coupling:

- coupling ∝ `amplitude * decay^r` at distance `r`  
- `op1`, `op2` : operator names on the two spins  
- `amplitude`  : overall prefactor  
- `decay`      : base of the exponential  
"""
struct ExpChannelCoupling <: TwoSite
    op1::String
    op2::String
    amplitude::Float64
    decay::Float64
end

"""
    Field(op, weight)

Single-site field coupling:

- `op`     : operator acting on the spin  
- `weight` : field strength  
"""
struct Field <: SingleSite
    op::String
    weight::Float64
end

# ──────────────────────────────────────────────────────────────────────────────
# 3) Pure-boson-only channel
# ──────────────────────────────────────────────────────────────────────────────

"""
    BosonOnlySB(op, weight)

Boson-only channel (no spin leg):

- `op`     : boson operator name  
- `weight` : coupling strength  
"""
struct BosonOnlySB <: SingleSite
    op::String
    weight::Float64
end

# ──────────────────────────────────────────────────────────────────────────────
# 4) Spin-boson channel structs via composition
# ──────────────────────────────────────────────────────────────────────────────

"""
    FiniteRangeCouplingSB(spin, boson_op, weight_boson)

Finite-range spin-boson coupling:

- `spin`         : a `FiniteRangeCoupling` for the spin–spin part  
- `boson_op`     : boson operator name  
- `weight_boson` : spin-boson coupling strength  
"""
struct FiniteRangeCouplingSB <: TwoSite
    spin::FiniteRangeCoupling
    boson_op::String
    weight_boson::Float64
end

"""
    ExpChannelCouplingSB(spin, boson_op, weight_boson)

Exponential spin-boson coupling:

- `spin`         : an `ExpChannelCoupling` for the spin–spin part  
- `boson_op`     : boson operator name  
- `weight_boson` : spin-boson coupling strength  
"""
struct ExpChannelCouplingSB <: TwoSite
    spin::ExpChannelCoupling
    boson_op::String
    weight_boson::Float64
end

"""
    FieldSB(field, boson_op, weight_boson)

Single-site spin-boson field coupling:

- `field`        : a `Field` for the spin part  
- `boson_op`     : boson operator name  
- `weight_boson` : spin-boson coupling strength  
"""
struct FieldSB <: SingleSite
    spin::Field
    boson_op::String
    weight_boson::Float64
end

# ──────────────────────────────────────────────────────────────────────────────
# 5) Pure-spin build path methods
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# 6) Spin-boson build path methods
# ──────────────────────────────────────────────────────────────────────────────

# for finite‐range couplings
function build_path(ns::Int,coupling::FiniteRangeCouplingSB,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1:ns+coupling.spin.dx
    # idle → first state : emit op1
    push!(transitions, (path[1],1, coupling.spin.op1, 1.0))
    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1], "I", 1.0))
    end
    # last state → final spin idle : emit op2
    push!(transitions, (path[end]+1,path[end], coupling.spin.op2,coupling.spin.weight))
    #spin idle loop
    push!(transitions,(path[end]+1,path[end]+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path[end]+1,coupling.boson_op,coupling.weight_boson))

    return path[end]+1, transitions
end

# for exponential couplings
function build_path(ns::Int,coupling::ExpChannelCouplingSB,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})

    path = ns+1
    # idle → first state : emit op1
    push!(transitions,(path,1,coupling.spin.op1,1.0))
    # self‑loop with identity and decay 
    push!(transitions, (path, path, "I",coupling.spin.decay))
    # last state → final spin idle : emit op2
    push!(transitions, (path+1,path, coupling.spin.op2,
            coupling.spin.amplitude*coupling.spin.decay))  
    #spin idle loop 
    push!(transitions,(path+1,path+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path+1,coupling.boson_op,
            coupling.weight_boson))

    return path+1, transitions
end 

# for single‐site field
function build_path(ns::Int,coupling::FieldSB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1
    #idle → first spin idle : emit op
    push!(transitions,(path,1,coupling.spin.op,coupling.spin.weight))
    #spin idle loop
    push!(transitions,(path,path,"I",1.0))
    #spin idle → end state: emit boson_op
    push!(transitions,(0,path,coupling.boson_op,coupling.weight_boson))

    return path, transitions
end 

#for boson only field
function build_path(ns::Int,coupling::BosonOnlySB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    push!(transitions,(0,1,coupling.op,coupling.weight))
    return ns,transitions
end

# ──────────────────────────────────────────────────────────────────────────────
# 6) Finite state Machine builder from paths
# ──────────────────────────────────────────────────────────────────────────────

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

function spinboson_FSM(channels::Vector{<:Channel})
    #start the finite state machine
    ns = 1
    #allocate a vector of tuples for hosting transitions
    transitions = Tuple{Int,Int,String,Float64}[]
    #put identity at initial and final node 
    push!(transitions, (1,1,"I", 1.0))

    #loop over different coupling channels
    for ch in channels
        ns, transitions = build_path(ns,ch,transitions)
    end 
    return ns+1, transitions 
end

# --- Example usage ---
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -1.0*im; 1.0*im  0.0]
sz = [1.0 0.0; 0.0  -1.0]
I2 = Matrix{Float64}(I, 2, 2)

#need to fix the definition of struct
#channels = [FiniteRangeCoupling("X", "X", 2,0.5),ExpChannelCoupling("Z", "Z",0.8, 0.5),Field("Y",0.8)]
spin1 = FiniteRangeCoupling("X","X",2,0.5)
spin2 = ExpChannelCoupling("Z","Z",0.8,0.5)
spin3 = Field("Y",0.8)
channels = [FiniteRangeCouplingSB(spin1,"Ib",1.0),ExpChannelCouplingSB(spin2,"Ib'",1.0),FieldSB(spin3,"a",2.0),BosonOnlySB("n",5.0)]

#ops = Dict("X"=>sx, "Y"=>sy, "Z"=>sz,"I"=>I2)
chi, transitions = spinboson_FSM(channels)
println(transitions)