using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra 
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
u_firm_new = Vector{Float64}[]
z_firm_new = Vector{Float64}[]
k = 1
j = 0
    for i in 1:length(entrant)
        j += entrant[i]
        temp_1 = uf_num[k:j]
        temp_2 = z_firm[k:j]
        u_firm_new = push!(u_firm_new, temp_1)
        z_firm_new = push!(z_firm_new, temp_2)
        k = j + 1
    end


"""
z_firm_new (Z_fm ~ N(0,1)) is fixed
"""

#############################################################################################################################################
################################################ Equilbrium firm calculation ################################################################
#############################################################################################################################################


function eq_firm_calc(tru_param::AbstractVector, other_param::parameters, market::AbstractVector, potential::AbstractVector, Z::AbstractVector, U::AbstractVector)
    """ equilbrium entered firm calculation
    Input:
    1. tru_param::AbstractVector : true parameters - [μ, σ, δ] 
    2. other_param::parameters : other true parameters [α, β, M]
    3. Market::AbstractVector - market observables
    4. potential::AbstractVector - potential entrants for each market (M vector)
    5. Z::AbstractVector - firm observable costs
    6. U::AbstractVector - firm fixed cost unobservables 
    Output: 
    1. eq_entered::AbstractVector : equilbrium entered firm number
    """
    # Initialization
    ## equilbrium firm entry : Most profitably firms enter firm
    #Profit : Π = X*β - (Z * α + u_firm)   
    eq_entered = zeros(eltype(Int64),length(potential))
    Π = similar(Z)
    for m in eachindex(potential)
        x = market[m]
        entr_num = potential[m]
        Z_m = Z[m]
        U_f = U[m]
        Π_m = zeros(eltype(Float64),entr_num)
        Π_m = x * other_param.β .- Z_m * other_param.α - U_f
        sort!(Π_m, rev= true)
        eq_entered[m] = 0
        entrant_number = Vector(1:1:potential[m])
        Profit = Π_m - tru_param[3] * log.(entrant_number)
        eq_entered[m] = count(i -> (i>=0), Profit)
        
    end
    return eq_entered  #250 X 1 vector, firm specific cost
end

entered_firm = eq_firm_calc(tru_param, param, X, entrant, z_firm_new, u_firm_new)
entered_firm # equilibrium entered firm number (This is dependent variable)

id_firm = Vector{Int64}[]    
for i in eachindex(entrant)
    temp = zeros(eltype(Int64),entrant[i])
    if entered_firm[i] == 1
        temp[1] = 1
        push!(id_firm, temp)        
    elseif entered_firm[i] == 2
        temp[1:2] .= 1
        push!(id_firm, temp)
    elseif entered_firm[i] == 3
        temp[1:3] .= 1
        push!(id_firm, temp)
    elseif entered_firm[i] == 4
        temp[1:4] .= 1
        push!(id_firm, temp)
    elseif entered_firm[i] == 0
        push!(id_firm, temp)
    end
end

id_firm_eq = copy(id_firm)
firmid_vec = zeros(eltype(Float64), sum(entrant, dims = 1)[1])
i = 1
g = 0
for m in 1: param.M
    g += length(id_firm_eq[m]) 
    firmid_vec[i:g] = id_firm_eq[m]
    i += length(id_firm_eq[m])
end
firmid_vec

Z = copy(z_firm_new) # Check firm observable fixed costs

for i in eachindex(entrant)
    sort!(Z[i], rev = true)
end




#############################################################################################################################################
################################################ Probit estimator ###########################################################################
#############################################################################################################################################

function entry_probit(param1::AbstractVector, fixed_param::parameters, market::AbstractVector, firm_char::AbstractVector, potential::AbstractVector, eq_firm::AbstractVector)
    """ loglike function
    Input:
    1. param1::AbstractVector : parameters of interest [μ, σ, δ] 
    2. fixed_param::parameters : other parameters [α, β, M]
    3. market::AbstractVector - marketwide observables : X (M vector)
    4. firm_char::AbstractVector - firm observable characteristics : 'M X entrant[m]' m=1,...,M vector
    5. potential::AbstractVector - potential entrants for each market (M vector)
    Output: 
    1. -loglik::Float64 : negative loglik value 
    """
    if param1[2] <0
        param1[2] = 1
    end

    Pr_0 = zeros(Float64, fixed_param.M) #Pr(N=0)
    Pr_1 = zeros(Float64, fixed_param.M) #Pr(N=1)
    Pr_2 = zeros(Float64, fixed_param.M) #Pr(N>=2)
    dis = Normal(param1[1],param1[2])

    for m in eachindex(potential) # Market m case
        x = market[m]
        Z_m = firm_char[m]
        entr_num = potential[m]
        ## each firm's profit 
        Π_m = zeros(eltype(Float64),entr_num)
        Π_m = x * fixed_param.β .- Z_m * fixed_param.α
        # order firms by profitability
        sort!(Π_m, rev = true)
        # Pr_1 = The first profitable firm enters so the rest firms must have negative profits
        Pr_0[m] = 1
        for i in 1: entr_num
            Pr_0[m] *= (1-cdf(dis,Π_m[i]))
        end 
        
        
        Pr_1[m] = cdf(dis, Π_m[1])*(1- cdf(dis, Π_m[2] - param1[3])) - (cdf(dis,Π_m[1]) - cdf(dis, Π_m[1] - param1[3])) * (cdf(dis,Π_m[2]) - cdf(dis,Π_m[2]-param1[3])) * (cdf(dis, Π_m[1]- Π_m[2]))
        Pr_2[m] = 1 - Pr_0[m] - Pr_1[m]
        
    end


    nofirm = eq_firm .== 0
    monopoly = eq_firm .== 1
    moretwo = eq_firm .>= 2
    Pr_0[Pr_0 .<= 0.0] .= 1e-10
    Pr_1[Pr_1 .<= 0.0] .= 1e-10
    Pr_2[Pr_2 .<= 0.0] .= 1e-10
        
    
    loglik = sum(nofirm .* log.(Pr_0) .+ monopoly.* log.(Pr_1) .+ moretwo .* log.(Pr_2))

    return -loglik
end

entry_probit(tru_param, param, X, Z, entrant, entered_firm)

## compute equilbrium firm numbers per market
opt_probit = Optim.optimize(vars -> entry_probit(vars, param, X, Z, entrant, entered_firm), ones(3), BFGS(), Optim.Options(show_trace = true, g_tol = 1e-7))
estimates_probit = opt_probit.minimizer
hessian_probit = hessian( vars -> entry_probit(vars, param, X, Z, entrant, entered_firm)  )
se_probit = diag(inv(hessian_probit(estimates_probit)))




#############################################################################################################################################
################################################ Simulated estimator ########################################################################
#############################################################################################################################################


function simulated_mm(param1::AbstractVector, param2::parameters, market::AbstractVector, firm_char::AbstractVector, eq_firm::AbstractVector, eq_firm_vec::AbstractVector, potential::AbstractVector, S::Int64, mode)
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
    if param1[2] < 0 
        param1[2] = 1.0
    end

    enter_firm = zeros(length(potential)*S)
    Z_m_temp = copy(firm_char)
    Z_m = repeat(Z_m_temp, S)
    X_m_temp = copy(market)
    X_m = repeat(X_m_temp, S)
    firm_number = repeat(potential, S)
    simu = rand(MersenneTwister(123), Normal(param1[1], param1[2]), sum(firm_number, dims = 1)[1])
    k = 1
    u_firm = Vector{Float64}[]
    j = 0
    for i in 1:length(firm_number)
        j += firm_number[i]
        temp = simu[k:j]
        u_firm = push!(u_firm, temp)
        k = j + 1
    end     
    
    if mode == "number"
        for j in 1:length(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            sort!(Pi, rev= true)
            entrant_number = Vector(1:1:firm_number[j])
            Profit = Pi - param1[3] * log.(entrant_number)
            enter_firm[j] = count(i -> (i>=0), Profit)
        end

        proj_temp = reshape(enter_firm, param2.M, S)
        proj = sum(proj_temp, dims = 2) / S
        Q = (eq_firm - proj)' * (eq_firm - proj)
        Ω = inv((eq_firm - proj)' * (eq_firm - proj))
        return Q[1]



    elseif mode == "identity"
        phat = Vector{Int64}[]
        for j in eachindex(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            n_hat = count(i -> (i >= 0), Pi)
            temp = zeros(eltype(Int64), firm_number[j])
            if n_hat == 1
                temp[1] = 1  
                push!(phat, temp)
            elseif n_hat == 2
                temp[1:2] .= 1
                push!(phat, temp)
            elseif n_hat == 3
                temp[1:3] .= 1
                push!(phat, temp)
            elseif n_hat == 4
                temp[1:4] .= 1
                push!(phat, temp)
            elseif n_hat == 0
                push!(phat, temp)
            end
        end
        phat_temp = reshape(phat, param2.M, S)
        temp_1 = Vector{Float64}[]
        for m in 1: param2.M
            push!(temp_1, sum(phat_temp[m,:]))
        end
        proj = temp_1 ./ 100

        proj_vec = zeros(eltype(Float64), sum(potential, dims = 1)[1])
        k = 1
        j = 0
        for m in 1: param2.M
            j += length(proj[m]) 
            proj_vec[k:j] = proj[m]
            k += length(proj[m])
        end

        Q = (eq_firm_vec - proj_vec)' * (eq_firm_vec - proj_vec)
        Ω = inv((eq_firm_vec - proj_vec)' * (eq_firm_vec - proj_vec))

        return Q[1]
    end

end


opt_identity = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, firmid_vec, entrant, 50, "identity"), ones(3), Optim.Options(show_trace = true, g_tol = 1e-7))
estimates_identity = opt_identity.minimizer

opt_number = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, firmid_vec, entrant, 500, "number"), ones(3), Optim.Options(show_trace = true, g_tol = 1e-7))
estimates_msm = opt_number.minimizer

simulated_mm(estimates_identity, param, X, Z, entered_firm, firmid_vec, entrant, 50, "identity")
simulated_mm(estimates_msm, param, X, Z, entered_firm, firmid_vec, entrant, 500, "number")





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
## Firm 1 case at market 2 (has two potential entrant so j = 2^2)


dmatrix = make_dmatrix(entrant_meq[1])
delta_meq = ones(entrant_meq[1]-1)
profit = zeros(eltype(Float64), 2^entrant_meq[1], entrant_meq[1])



for i in 1:entrant_meq[1]
        for j in 1:size(profit,1)
            if dmatrix[j,i] == 1
                profit[j,i] = X_meq[1] .- Z_meq[1][i] - dmatrix[:, 1:end .!= i][j,:]' * delta_meq
            elseif dmatrix[j,i] == 0
                profit[j,i] = 0
            end
        end
end

