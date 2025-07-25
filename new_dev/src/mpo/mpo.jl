using LinearAlgebra

# Finite‑range coupling between i and i+dx
struct FiniteRangeCoupling
    op1::String
    op2::String
    dx::Int
    weight::Float64
end

# Exponential channel: A e^{-b r} coupling
struct ExpChannelCoupling
    op1::String
    op2::String
    amplitude::Float64
    decay::Float64
end

struct Fields
    op::String
    weight::Float64
end

function spin_finite_state_machine(channels)
    #start the finite state machine
    num_states = 1
    #allocate a vector of tuples for hosting transitions
    transitions = Tuple{Int,Int,String,Float64}[]
    #put identity at initial and final node 
    push!(transitions, (1,1,"I", 1.0))
    push!(transitions, (0,0,"I", 1.0)) # the final state is represented as 0

    #loop over different coupling channels
    for ch in channels
        if ch isa FiniteRangeCoupling
            num_states,transitions = two_site_finite_path(num_states,ch,transitions)
        elseif ch isa ExpChannelCoupling
            num_states,transitions = two_site_exp_channel_path(num_states,ch,transitions)
        elseif ch isa Fields
            transitions = single_site_field_path(ch,transitions)
        end 
    end 
    return num_states+1, transitions 
end

#here coupling will be a struct so don't use it as a vector of tuples but as a struct, that way it is easier to call the arguments, for eg dx will be coupling.dx not coupling[3]
function two_site_finite_path(num_states::Int,coupling::FiniteRangeCoupling,transitions::Vector{Tuple{Int64, Int64, String, Float64}})
    path = num_states+1:num_states+coupling.dx
    # idle → first state : emit op1
    push!(transitions, (path[1],1, coupling.op1, 1.0))

    # intermediate identity hops
    for i in 2:length(path)
        push!(transitions, (path[i], path[i-1], "I", 1.0))
    end
    # last state → final idle : emit op2
    push!(transitions, (0,path[end], coupling.op2,coupling.weight))

    return num_states+coupling.dx, transitions
end

function two_site_exp_channel_path(num_states::Int,coupling::ExpChannelCoupling,transitions::Vector{Tuple{Int64, Int64, String, Float64}})

    path = num_states+1
    # idle → first state : emit op1
    push!(transitions,(path,1,coupling.op1,1.0))

    # self‑loop with identity and decay 
    push!(transitions, (path, path, "I",coupling.decay))

    # last state → final idle : emit op2
    push!(transitions, (0,path, coupling.op2,coupling.amplitude*coupling.decay))   

    return num_states+1, transitions
end 

function single_site_field_path(coupling::Fields,transitions::Vector{Tuple{Int64, Int64, String, Float64}})

    push!(transitions(0,1,coupling.op,coupling.weight))

    return transitions
end 

# 2. Initialize empty grids
function init_grids(chi::Int, d::Int,T::Type=Float64)
    # 4D: (χ_in, χ_out, phys_in, phys_out)
    return zeros(T, chi, chi, d, d)
end

function populate_grids!(
    grids::AbstractArray{G,4},
    transitions::Vector{Tuple{Int,Int,String,W}},
    phys_ops::Dict{String,Matrix},
) where {G<:Number, W<:Number}

    chi = length(grids[:,1,1,1])
    for (row,col,opname,w) in transitions 
        op_mat = phys_ops[opname]
        grids[row==0 ? chi : row,col == 0 ? chi : col,:,:] = w*op_mat 
    end 
    return grids 
end

# --- Example usage ---
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -1.0*im; 1.0*im  0.0]
sz = [1.0 0.0; 0.0  -1.0]
I2 = Matrix{Float64}(I, 2, 2)

channels = [FiniteRangeCoupling("X", "X", 1 ,0.5),FiniteRangeCoupling("Z", "Z", 2,0.5)]
ops = Dict("X"=>sx, "Y"=>sy, "Z"=>sz,"I"=>I2)
chi, transitions = spin_finite_state_machine(channels)
println(transitions)

#d = 2
#grids = init_grids(chi,d,Float64)
#mpo = populate_grids!(grids,transitions,ops)

#channels = [
#    FiniteRangeCoupling("X", "X", 1 ,1.0),    # X_i X_{i+1}
#    FiniteRangeCoupling("Y", "Y",2,0.5),    # Y_i Y_{i+2}
    # approximate 1/r^3 as 3 exponentials (example params)
#    ExpChannelCoupling("Z", "Z",0.8, 0.5),
#    ExpChannelCoupling("Z", "Z",0.8, 0.5),
#    ExpChannelCoupling("Z", "Z",0.8, 0.5),
#]


