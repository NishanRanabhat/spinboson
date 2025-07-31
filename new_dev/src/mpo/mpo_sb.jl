using LinearAlgebra

# Finite‑range coupling between spins at i and i+dx
struct FiniteRangeCouplingSB
    op1::String
    op2::String
    boson_op::String
    dx::Int
    weight_spin::Float64
    weight_boson::Float64
end

# Exponential channel: amplitude*lambda^r coupling among spins at distance r
struct ExpChannelCouplingSB
    op1::String
    op2::String
    boson_op::String
    amplitude::Float64
    decay::Float64
    weight_boson::Float64
end

struct FieldSB
    op::String
    boson_op::String
    weight_spin::Float64
    weight_boson::Float64
end

struct BosonOnlySB
    op::String
    weight::Float64
end

function spinboson_finite_state_machine(channels)
    #start the finite state machine
    ns = 1
    #allocate a vector of tuples for hosting transitions
    transitions = Tuple{Int,Int,String,Float64}[]
    #put identity at initial and final node 
    push!(transitions, (1,1,"I", 1.0))

    #loop over different coupling channels
    for ch in channels
        if ch isa FiniteRangeCouplingSB
            ns,transitions = two_site_finite_path_SB(ns,ch,transitions)
        elseif ch isa ExpChannelCouplingSB
            ns,transitions = two_site_exp_channel_path_SB(ns,ch,transitions)
        elseif ch isa FieldSB
            ns,transitions = single_site_field_path_SB(ns,ch,transitions)
        elseif ch isa BosonOnlySB
            ns,transitions = boson_only_path_SB(ns,ch,transitions)
        end 
    end 
    return ns+1, transitions 
end

function spinboson_FSM(channels)
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

#here coupling will be a struct so don't use it as a vector of tuples but as a struct, that way it is easier to call the arguments, for eg dx will be coupling.dx not coupling[3]
function two_site_finite_path_SB(ns::Int,coupling::FiniteRangeCouplingSB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1:ns+coupling.dx
    # idle → first state : emit op1
    push!(transitions, (path[1],1, coupling.op1, 1.0))
    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1], "I", 1.0))
    end
    # last state → final spin idle : emit op2
    push!(transitions, (path[end]+1,path[end], coupling.op2,coupling.weight_spin))
    #spin idle loop
    push!(transitions,(path[end]+1,path[end]+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path[end]+1,coupling.boson_op,coupling.weight_boson))

    return path[end]+1, transitions
end

# for finite‐range couplings
function build_path(ns::Int,coupling::FiniteRangeCouplingSB,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1:ns+coupling.dx
    # idle → first state : emit op1
    push!(transitions, (path[1],1, coupling.op1, 1.0))
    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1], "I", 1.0))
    end
    # last state → final spin idle : emit op2
    push!(transitions, (path[end]+1,path[end], coupling.op2,coupling.weight_spin))
    #spin idle loop
    push!(transitions,(path[end]+1,path[end]+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path[end]+1,coupling.boson_op,coupling.weight_boson))

    return path[end]+1, transitions
end

function two_site_exp_channel_path_SB(ns::Int,coupling::ExpChannelCouplingSB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})

    path = ns+1
    # idle → first state : emit op1
    push!(transitions,(path,1,coupling.op1,1.0))
    # self‑loop with identity and decay 
    push!(transitions, (path, path, "I",coupling.decay))
    # last state → final spin idle : emit op2
    push!(transitions, (path+1,path, coupling.op2,coupling.amplitude*coupling.decay))  
    #spin idle loop 
    push!(transitions,(path+1,path+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path+1,coupling.boson_op,coupling.weight_boson))

    return path+1, transitions
end 

function build_path(ns::Int,coupling::ExpChannelCouplingSB,
    transitions::Vector{Tuple{Int64, Int64, String, Float64}})

    path = ns+1
    # idle → first state : emit op1
    push!(transitions,(path,1,coupling.op1,1.0))
    # self‑loop with identity and decay 
    push!(transitions, (path, path, "I",coupling.decay))
    # last state → final spin idle : emit op2
    push!(transitions, (path+1,path, coupling.op2,
            coupling.amplitude*coupling.decay))  
    #spin idle loop 
    push!(transitions,(path+1,path+1,"I",1.0))
    #spin idle → end state: emit boson_op 
    push!(transitions,(0,path+1,coupling.boson_op,
            coupling.weight_boson))

    return path+1, transitions
end 

function single_site_field_path_SB(ns::Int,coupling::FieldSB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1
    #idle → first spin idle : emit op
    push!(transitions,(path,1,coupling.op,coupling.weight_spin))
    #spin idle loop
    push!(transitions,(path,path,"I",1.0))
    #spin idle → end state: emit boson_op
    push!(transitions,(0,path,coupling.boson_op,coupling.weight_boson))

    return path, transitions
end 

function build_path(ns::Int,coupling::FieldSB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = ns+1
    #idle → first spin idle : emit op
    push!(transitions,(path,1,coupling.op,coupling.weight_spin))
    #spin idle loop
    push!(transitions,(path,path,"I",1.0))
    #spin idle → end state: emit boson_op
    push!(transitions,(0,path,coupling.boson_op,coupling.weight_boson))

    return path, transitions
end 

function boson_only_path_SB(ns::Int,coupling::BosonOnlySB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    push!(transitions,(0,1,coupling.op,coupling.weight))
    return ns,transitions
end

function build_path(ns::Int,coupling::BosonOnlySB,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    push!(transitions,(0,1,coupling.op,coupling.weight))
    return ns,transitions
end

channels = [FiniteRangeCouplingSB("X", "X","Ib",2,0.5,1.0),ExpChannelCouplingSB("Z", "Z","Ib'",0.8, 0.5,1.0),FieldSB("Y","a",0.8,2.0),BosonOnlySB("n",5.0)]

chi, transitions = spinboson_finite_state_machine(channels)
println(transitions)
chi1,transitions1 = spinboson_FSM(channels)
println(transitions1)


