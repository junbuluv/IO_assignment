using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff
## generate data
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


# firm's observable characteristics Z ~ N(0,1) in each market : entrant_number X 1
tru_param = [μ,σ,δ]
function eq_firm(tru_param::AbstractVector, param, entrant::AbstractVector)
    # unobservable part of firm generating
    u_firm_new = Vector{Float64}[]
    for i in eachindex(entrant)
        u_firm = rand(MersenneTwister(1234),Normal(tru_param[1], tru_param[2]), entrant[i])
        u_firm_new = push!(u_firm_new, u_firm)
    end
    # observable part firm generating
    z_firm_new = Vector{Float64}[]
    for i in eachindex(entrant)
        z_firm = rand(Normal(0,1), entrant[i])
        z_firm_new = push!(z_firm_new, z_firm)
    end
    ## equilbrium firm entry : Most profitably firms enter firm
    #Profit : Π = X*β - (Z * α + u_firm)   
    n = zeros(eltype(Int64),length(entrant))
    Π = similar(u_firm_new)
    for i in eachindex(entrant)
        Π[i] = X[i] * param.β .- z_firm_new[i] * param.α - u_firm_new[i]
        sort!(Π[i], rev= true)
        for s in 1:entrant[i]
            if Π[i][s] - tru_param[3] * log(n[i]) >= 0 
                n[i] += 1
            elseif Π[i][s] <0
                n[i] = n[i]
            end
        end
    end
    return n, z_firm_new  #250 X 1 vector, firm specific cost
end


entered_firm, Z = eq_firm(tru_param, param, entrant)
entered_firm # equilibrium entered firm number (This is dependent variable)
Z # Check firm observable fixed costs


function entry_probit(int_param::AbstractVector,param, X::AbstractVector, Z::AbstractVector, entrant::AbstractVector, depvar::AbstractVector)

    Pr_0 = zeros(Float64, param.M) #Pr(N=0)
    Pr_1 = zeros(Float64, param.M) #Pr(N=1)
    Pr_2 = zeros(Float64, param.M) #Pr(N>=2)
    for m in eachindex(entrant) # Market m case
        x = X[m]
        Z_m = Z[m]
        entr_num = entrant[m]
        ## each firm's profit 
        Π_m = zeros(eltype(Float64),entr_num)
        Π_m = x * param.β .- Z_m * param.α
        # order firms by profitability
        sort!(Π_m, rev = true)
        # Pr_1 = The first profitable firm enters so the rest firms must have negative profits
        Pr_0_k = zeros(eltype(Float64), entr_num)
        dis = Normal(int_param[1],int_param[2])
        for k in 1:entr_num
            Pr_0_k[k] = cdf(dis, -Π_m[k])
        end
        Pr_0[m] = prod.(eachcol(Pr_0_k))[1]
        
        if entr_num == 2 
            # Pr_2 : More than 2 firms enter
            Pr_2[m] = cdf(dis, Π_m[1] - int_param[3]) * cdf(dis, Π_m[2] - int_param[3])
            # Pr_1 : Most profitable (Lowest fixed cost) enters first
            Pr_1[m] = 1 - Pr_0[m] - Pr_2[m]
        elseif entr_num >= 3
            Pr_1[m] = cdf(dis, Π_m[1])*cdf(dis, -Π_m[2] + int_param[3]) - (cdf(dis, -Π_m[1] + int_param[3]) - cdf(dis,-Π_m[1]))*cdf(dis,-Π_m[2]+int_param[3] - cdf(dis,-Π_m[2]))*(1- cdf(dis, (Π_m[2] - Π_m[1])/2))
            Pr_2[m] = 1 - Pr_0[m] - Pr_1[m]
        end
    end
    nofirm = depvar .== 0
    monopoly = depvar .== 1
    moretwo = depvar .>= 2
    Pr_0[Pr_0 .<= 0.0] .= 1e-10
    Pr_1[Pr_1 .<= 0.0] .= 1e-10
    Pr_2[Pr_2 .<= 0.0] .= 1e-10
        loglik = sum(sum(nofirm .* log.(Pr_0) .+ monopoly.* log.(Pr_1) .+ moretwo .* log.(Pr_2)))

    return -loglik
end

entry_probit(tru_param, param, X, Z, entrant, entered_firm)

## compute equilbrium firm numbers per market
opt = Optim.optimize(vars -> entry_probit(vars, param, X, Z,entrant, entered_firm),tru_param, Optim.Options(show_trace = true, g_tol = 1e-14))
opt.minimizer

tru_param



"""
Simulated estimator

Start from firm's fixed cost and sort them by profitability
"""



function eq_firm(tru_param::AbstractVector ,param, X::AbstractVector, entrant::AbstractVector)
    entered_firm = zeros(eltype(Int64), length(entrant))
  for m in eachindex(entrant) # Market m case
      x = X[m]
      entr_num = entrant[m] # potential entrant
      Z_m = randn(entr_num) # firm characteristics
      ## each firm's profit 
      Π_m = zeros(eltype(Float64),entr_num)
      u_m = rand(Normal(tru_param[1], tru_param[2]), entr_num)
      Π_m = x * param.β .- Z_m * param.α - u_m
      # order firms by profitability
      sort!(Π_m, rev = true)
      ## computed entered firm by profitability
            for n in 1: entr_num
                if Π_m[n] - tru_param[3] * log(n) >= 0
                   entered_firm[m] += 1
                elseif Π_m[n] - tru_param[3] * log(n) < 0
                    entered_firm[m] = n-1
                end  
            end
    end
  return entered_firm
end



#for m in eachindex(X)
        x = X[m]
        entr_num = entrant[m]
        Z_m = randn(entr_num) # firm characteristics
        ## each firm's profit 
        Π_m = zeros(eltype(Float64),entr_num)
        Π_m = x * param.β .- Z_m * param.α
        # order firms by profitability
        sort!(Π_m, rev = true)
        ### Draw simulation 
        dis = Normal(0,1)
        ϵ =  rand(MersenneTwister(1234), dis, entr_num)
        Π = zeros(eltype(Float64), entr_num)
        for k in 1: entr_num
            Π[k] = Π_m[k] - δ * log(k) + ϵ[k]
        end

        Π