using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra, DataFrames
#############################################################################################################################################
################################################ Data generating ############################################################################
#############################################################################################################################################
"""
M = 250 : markets
1. X_m : Market m's characteristics M X 1 vector ~ N(3,1) 
2. Z_fm : firm characteristics ~N(0,1)
3. alpha, beta , delta = (1,1,1)
4. mu , sigma^2 = (2,1)
5. firm variables : potential firm number is 4
"""
@with_kw mutable struct parameters
    M::Int64 = 250
    α::Float64 = 1.0
    β::Float64  = 1.0
end


param = parameters()
δ = 1.0
μ = 2.0
σ = 1.0

# Generate observable market variable X ~ N(3,1) : 250 X 1 vector and fix it

X = rand(Normal(3,1), param.M)

# draw the numbers of potential entrants in each market then fix it
F = [2,3,4]
entrant = sample(MersenneTwister(342) ,F, param.M; replace = true, ordered = false)

"""
First stage
1. There are N numbers of potential entrants at each market (This potential entrant numbers take probability 1/3)
2. Fixed cost: ϕ_{fm} = Z_{fm}α + u_{fm} , u_{fm} ~ N(μ,σ^2)
3. Lowest fixed cost firms enter first (Define variable cost)
    π_{fm} = X_{m} * β - δ ln(N_{m}) - (Z_{fm}α + u_{fm}) 
Second stage 
Firms maximize their expost profit
4. Pr(N=0), Pr(N=1), Pr(N>=2)
4-1. Pr(N=0) : All the firms make less than zero profit
4-2. Pr(N=1) : One firm enters (Satisfying positive Duopoly, Monopoly profits)
"""
tru_param = [μ,σ,δ]


uf_num = rand(MersenneTwister(123), Normal(tru_param[1], tru_param[2]), sum(entrant, dims = 1)[1])
z_firm = rand(Normal(0,1), sum(entrant, dims = 1)[1])
k = 1
market_index = Vector{Int64}(undef,1)
for m in eachindex(entrant)
    temp = ones(eltype(Int64), entrant[m]) * k
    market_index = append!(market_index, temp)
    k += 1
end

market_index = market_index[2:end]

X_data = Vector{Float64}(undef,1)
entrant_data = Vector{Int64}(undef,1)
for m in eachindex(entrant)
    temp = ones(eltype(Float64), entrant[m]) .* X[m]
    X_data = append!(X_data , temp)
    temp2 = ones(eltype(Int64), entrant[m]) .* entrant[m]
    entrant_data = append!(entrant_data, temp2)
end
    
entrant_data = entrant_data[2:end]
X_data = X_data[2:end]
Z_data = copy(z_firm)
U_data = copy(uf_num)

observed = X_data - Z_data - U_data
Π = Vector{Float64}(undef,1)
k = 1
for m in eachindex(entrant)
    Π = append!(Π, sort!(observed[k : k - 1 + entrant[m]], rev = true))
    k = k + entrant[m] 
end 
Π = Π[2:end]


df = DataFrame(index = market_index, Profit = Π, fnum = entrant_data, u_firm = U_data)
eq_entered = zeros(eltype(Int64), param.M)
decision = Vector{Int64}(undef,1)
f_num = Vector{Int64}(undef,1)

for m in eachindex(entrant)
    obs = df.Profit[df.index .== m]
    temp = obs - log.(Vector(1:1:entrant[m])) - U_data[df.index .== m]
    eq_entered[m] = count(i -> (i > 0), temp)
    if eq_entered[m] == 0
        temp1 = zeros(eltype(Int64), entrant[m])
        temp2 = zeros(eltype(Int64), entrant[m]) 
    elseif eq_entered[m] > 0
        temp1 = zeros(eltype(Int64), entrant[m])
        temp2 = ones(eltype(Int64), entrant[m])
        temp2 = temp2 .* eq_entered[m]
        temp1[1:eq_entered[m]] .= 1
    end
    append!(decision, temp1)
    append!(f_num, temp2)    
end    
    
decision = decision[2:end]
f_num = f_num[2:end]


data_1 = DataFrame(index = market_index, Profit = Π, potential = entrant_data, d = decision, eq_fnum = f_num)




#############################################################################################################################################
################################################ Simulated estimator ########################################################################
#############################################################################################################################################



function simulated_mm(param1::AbstractVector, param2::parameters, data::DataFrame, entrant::AbstractVector, S::Int64)
    """
    Input:
    1. param1::AbstractVector : parameters of interest : μ, σ, δ
    2. fixed_param::parameters : other parameters [α, β, M]
    3. market::AbstractVector - marketwide observables
    4. firm_char::AbstractVector - firm observable characteristics : 'M X entrant[m]' m=1,...,M vector
    5. eq_firm::AbstractVector - equilbrium entered firm number for each market 'M' 
    6. potential::AbstractVector - potential entrants for each market
    7. S::Int64 - simulation number

    Output:
    1. Criterion function value (N* - N_simulated)' * (N* - N_simulated)
    """
    if param1[2] < 0.5
        param1[2] = 1.0
    end
    Profit = repeat(copy(data.Profit),S)
    index = repeat(copy(data.index),S)
    eq_number = repeat(data.eq_fnum, S)
    w = 1
    simu = Vector{Float64}(undef,1)
    while w < S+1
        temp = rand(MersenneTwister(w+100), Normal(param1[1], param1[2]), sum(entrant, dims = 1)[1])
        simu = append!(simu, temp)
        w += 1
    end
    simu = simu[2:end]

    dec = repeat(copy(data.d),S)
    s_index = reshape(repeat(Vector(1:1:S)',729), 729 *S)


    temp_data = DataFrame(i = index, P = Profit, si = simu, d = dec, eq= eq_number, si_index = s_index)


    eq_msm = zeros(eltype(Int64), param2.M, S)
    d_msm = Array{Int64}(undef, 1)
    f_msm = Array{Int64}(undef, 1)
    m = 1
    s = 1

    while s < S+1
        for m in eachindex(entrant)
            obs = temp_data.P[(temp_data.i .== m) .& (temp_data.si_index .== s)]
            temp = obs - log.(Vector(1:1:entrant[m])) - temp_data.si[(temp_data.i .== m) .& (temp_data.si_index .== s)]
            eq_msm[m,s] = count(i -> (i > 0), temp)
            if eq_msm[m,s] == 0
                temp1 = zeros(eltype(Int64), entrant[m])
                temp2 = zeros(eltype(Int64), entrant[m]) 
            elseif eq_msm[m,s] > 0
                temp1 = zeros(eltype(Int64), entrant[m])
                temp2 = ones(eltype(Int64), entrant[m])
                temp2 = temp2 .* eq_msm[m,s]
                temp1[1:eq_msm[m,s]] .= 1
            end
            append!(d_msm, temp1)
            append!(f_msm, temp2)    
        end  
        s += 1
    end
    

    proj = vcat(d_msm[2:end], f_msm[2:end])

    moment = (vcat(temp_data.d, temp_data.eq) - proj)
    Q = moment' * moment
end


opt_identity = Optim.optimize(vars -> simulated_mm(vars, param, data_1, entrant, 50), ones(3), Optim.Options(show_trace = true, f_tol = 1e-5 ,g_tol = 1e-5))
opt_identity.minimizer





#############################################################################################################################################
################################################ Moment inequality ##########################################################################
#############################################################################################################################################


"""
1. Simulation procedure (Following Ciliberto and Tamer, 2009)
    Step 1.  Transform the given matrix of epsilon draws into a draw with covariance matrix specified in θ. This is stored in ϵ^r
    Here I follow random number generating I used in the previous question, also I redeclare X and Z for this estimation
"""
X_meq = copy(X)
Z_meq = copy(z_firm_new)
entrant_meq = copy(entrant)

uf_meq = rand(MersenneTwister(123), Normal(tru_param[1], tru_param[2]), sum(entrant_meq, dims = 1)[1])
u_firm_meq = Vector{Float64}[]
k = 1
j = 0
    for i in 1:length(entrant_meq)
        j += entrant_meq[i]
        temp_1 = uf_meq[k:j]
        u_firm_meq = push!(u_firm_meq, temp_1)
        k = j + 1
    end

U_meq = copy(u_firm_meq)



function make_dmatrix(k::Int64)                                            
    t_num=2^k
    index=Matrix{Int64}(undef, t_num, k)                                                      
    i = 1
    for i in 1:k
        j = 1                                                                  
        z = 1
        c = Int64(t_num / (2^i))
        while j <= t_num
            if z <= c
                index[j,i] = 1
                z= z+1
                j= j+1
            else
                j= j+c
                z= 1
            end
        end
    end
    index[index .!= 1] .= 0
    return index
end
## First stage empirical frequency estimator
p_0 = count(i -> (i == 0), entered_firm) / length(entered_firm)
p_1 = count(i -> (i == 1), entered_firm) / length(entered_firm)
p_2 = count(i -> (i == 2), entered_firm) / length(entered_firm)
p_3 = count(i -> (i == 3), entered_firm) / length(entered_firm)
p_4 = count(i -> (i == 4), entered_firm) / length(entered_firm)
check = p_0 + p_1 + p_2 + p_3 + p_4
if check != 1.0 
    println("empirical probability error")
else
    println("empirical probability okay")
end
dep_var = [p_0, p_1, p_2, p_3, p_4]

S = 100
# Draw simulation 
simu_firm = repeat(entrant_meq, S)
epsi = rand(MersenneTwister(123), Normal(tru_param[1], tru_param[2]), sum(simu_firm, dims = 1)[1])

function unobs_conversion(ϵ::AbstractVector, firm::AbstractVector)
    k = 1
    u_firm = Vector{Float64}[]
    j = 0
    for i in 1:length(firm)
        j += firm[i]
        temp = ϵ[k:j]
        u_firm = push!(u_firm, temp)
        k = j + 1
    end
    return u_firm
end     

epsi_meq = unobs_conversion(epsi, simu_firm)
epsi_simu = reshape(epsi_meq, param.M, S)

## Firm 1 case at market 2 (has two potential entrant so j = 2^2)




profit = Array{Float64}(undef, 2^entrant_meq[1], entrant_meq[1], param.M)
h_1 = zeros(eltype(Float64), 2^entrant_meq[1], entrant_meq[1], param.M)
h_2 = zeros(eltype(Float64), 2^entrant_meq[1], entrant_meq[1], param.M)


epsi_simu[:,1][1]
s = 1
while s < 100
    for m in eachindex(entrant_meq)        
    dmatrix = make_dmatrix(entrant_meq[1])
    delta_meq = ones(entrant_meq[1]-1)
        for i in 1:entrant_meq[m]
            for j in 1:size(profit,1)
                if dmatrix[j,i] == 1
                    profit[j,i,m] = X_meq[m] .- Z_meq[m][i] - dmatrix[:, 1:end .!= i][j,:]' * delta_meq - epsi_simu[:,s][m][i]
                elseif dmatrix[j,i] == 0
                profit[j,i,m] = 0
                end
            end
        end
        count_temp = zeros(eltype(Int64), 2^entrant_meq[m])
        for j in eachindex(count_temp)
            count_temp[j] = count(i -> (i > 0), profit[j,:,m])
            for i in 1:entrant_meq[m]
                if profit[j,i,m] > 0 && count_temp[j] == 1
                    h_2[j,i,m] += 1.0
                end
                if profit[j,i,m] > 0
                    h_1[j,i,m] += 1.0
                end 
            end 
        end
    end
    s += 1
end


h_1 = h_1 / S
h_2 = h_2 / S