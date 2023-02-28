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
X = rand(Normal(3,1), param.M)
F = [2,3,4]
entrant = sample(MersenneTwister(342) ,F, param.M; replace = true, ordered = false)

"""
First stage
1. There are N numbers of potential entrants at each market (This potential entrant numbers take probability 1/3)
2. Fixed cost: ϕ_{fm} = Z_{fm}α + u_{fm} , u_{fm} ~ N(μ,σ^2)
3. Lowest fixed cost firms enter first (Define variable cost)
    π_{fm} = X_{m} * β - δ ln(N_{m}) - (Z_{fm}α + u_{fm}) 
"""

tru_param = [μ,σ,δ]
uf_num = rand(MersenneTwister(123), Normal(tru_param[1], tru_param[2]), sum(entrant, dims = 1)[1])
z_firm = rand(Normal(0,1), sum(entrant, dims = 1)[1])
u_firm_new = Vector{Float64}[]
z_firm_new = Vector{Float64}[]
k = 1
j = 0

for i in eachindex(entrant)
    j += entrant[i]
    temp_1 = uf_num[k:j]
    temp_2 = z_firm[k:j]
    u_firm_new = push!(u_firm_new, temp_1)
    z_firm_new = push!(z_firm_new, temp_2)
    k = j + 1
end



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
    rank_firm = Vector{Int64}[]
    decision_firm = Vector{Int64}[]
    for m in eachindex(potential)
        x = market[m]
        entr_num = potential[m]
        Z_m = Z[m]
        U_f = U[m]
        Π_m = zeros(eltype(Float64),entr_num)
        Π_m = x * other_param.β .- Z_m * other_param.α - U_f
        Π_m_ranked =  sort(Π_m, rev= true)
        eq_entered[m] = 0
        entrant_number = Vector(1:1:potential[m])

        Profit = Π_m_ranked - tru_param[3] * log.(entrant_number)
        eq_entered[m] = count(i -> (i>=0), Profit)


        temp1 = Profit + tru_param[3] * log.(entrant_number)
        temp2 = round.(Π_m; digits = 5)
        temp3 = round.(temp1; digits = 5)
        temp4 = zeros(eltype(Int64), potential[m])
        for j in 1: potential[m]
            temp4[j] = findall(temp2 .== temp3[j])[1]
        end
        
        rank_firm = push!(rank_firm, temp4)
        temp_d = temp4 .<= eq_entered[m]
        decision_firm = push!(decision_firm, temp_d)

        
    end
    return eq_entered, decision_firm 
end



"""
Data
"""

entered_firm, decision = eq_firm_calc(tru_param, param, X, entrant, z_firm_new, u_firm_new)
entered_firm # equilibrium entered firm number (This is dependent variable)
decision # equilibrium firm entry decision
Z = copy(z_firm_new) # Check firm observable fixed costs






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
        
        
        Pr_1[m] = cdf(dis, Π_m[1])*(1- cdf(dis, Π_m[2] - param1[3])) - (cdf(dis,Π_m[1]) - cdf(dis, Π_m[1] - param1[3])) * (cdf(dis,Π_m[2]) - cdf(dis,Π_m[2]-param1[3])) * (cdf(dis, Π_m[1]- Π_m[2])/2)
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



#############################################################################################################################################
################################################ Simulated estimator ########################################################################
#############################################################################################################################################



function simulated_mm(param1::AbstractVector, param2::parameters, market::AbstractVector, firm_char::AbstractVector, eq_firm::AbstractVector, eq_firm_vec::AbstractVector, potential::AbstractVector, S::Int64, mode)
    """
    Input:
    1. param1::AbstractVector: parameters of interest : μ, σ, δ
    2. param2::parameters: other parameters [α, β, M]
    3. market::AbstractVector: marketwide observables
    4. firm_char::AbstractVector: firm observable characteristics : 'M X entrant[m]' m=1,...,M vector
    5. eq_firm::AbstractVector: equilbrium entered firm number for each market 'M' 
    6. eq_firm_vec::AbstractVector: equilbirium firm entry decisions
    6. potential::AbstractVector - potential entrants for each market
    7. S::Int64: simulation number
    8. mode: identity(identities + numbers), number, identity(reverse_order), and number(reverse_order).

    Output:
    1. Criterion function value (identity mode) Q(θ) = (Entry_decision* - decision_simulated, N* - N_simulated)' * W * (Entry_decision* - decision_simulated, N* - N_simulated)
    2. Criterion function value (identity mode) Q(θ) = (N* - N_simulated)' * W * (N* - N_simulated)
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
    simu = rand(MersenneTwister(555), Normal(param1[1], param1[2]), sum(firm_number, dims = 1)[1])
    k = 1
    u_firm = Vector{Float64}[]
    j = 0
    eq_entered = repeat(eq_firm, S)
    for i in eachindex(firm_number)
        j += firm_number[i]
        temp = simu[k:j]
        u_firm = push!(u_firm, temp)
        k = j + 1
    end     
    
    if mode == "number"
        for j in eachindex(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            sort!(Pi, rev= true)
            entrant_number = Vector(1:1:firm_number[j])
            Profit = Pi - param1[3] * log.(entrant_number)
            enter_firm[j] = count(i -> (i>=0), Profit)
        end

        proj_temp = reshape(enter_firm, param2.M, S)
        proj = sum(proj_temp, dims = 2) / S
        moment = eq_firm - proj
        Q = (moment)' * (moment)
        return Q[1] 
    elseif mode =="numberrev"

        for j in eachindex(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            sort!(Pi, rev= false)
            entrant_number = Vector(1:1:firm_number[j])
            Profit = Pi - param1[3] * log.(entrant_number)
            enter_firm[j] = count(i -> (i>=0), Profit)
        end

        proj_temp = reshape(enter_firm, param2.M, S)
        proj = sum(proj_temp, dims = 2) / S
        moment = eq_firm - proj
        Q = (moment)' * (moment)
        return Q[1] 

    elseif mode == "identity"
        decision_firm = Vector{Int64}[]
        for j in eachindex(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            Pi_ranked = sort(Pi, rev = true)
            entrant_number = Vector(1:1:firm_number[j])
            Profit = Pi_ranked - param1[3] * log.(entrant_number)
            enter_firm[j] = count(i -> (i >= 0), Profit)

            temp1 = Profit + param1[3] * log.(entrant_number)
            temp2 = round.(Pi; digits = 5)
            temp3 = round.(temp1; digits = 5)
            temp4 = zeros(eltype(Int64), firm_number[j])
            for m in 1: firm_number[j]
                temp4[m] = findall(temp2 .== temp3[m])[1]
            end
     
            
            temp_d = temp4 .<= eq_entered[j]
            decision_firm = push!(decision_firm, temp_d)
        end

        d = Vector{Int64}(undef,1)
        for m in eachindex(firm_number)
            append!(d, decision_firm[m])
        end
        d = d[2:end]
        d_temp = reshape(d, sum(potential), S)
        d_eq = Vector{Float64}(undef,1)
        for m in eachindex(potential)
            append!(d_eq, eq_firm_vec[m])
        end
        d_eq = d_eq[2:end]

        d_proj = sum(d_temp, dims = 2) ./ S

        proj_2 = reshape(enter_firm, param2.M, S)
        proj_num = sum(proj_2, dims = 2) / S
        moment = vcat((eq_firm - proj_num), (d_eq - d_proj)) 

        Q = (moment)' * (moment)

        return Q[1]

    elseif mode == "identityrev"
        decision_firm = Vector{Int64}[]
        for j in eachindex(firm_number)
            Pi = param2.β * X_m[j] .- param2.α * Z_m[j] .- u_firm[j]
            Pi_ranked = sort(Pi, rev = false)
            entrant_number = Vector(1:1:firm_number[j])
            Profit = Pi_ranked - param1[3] * log.(entrant_number)
            enter_firm[j] = count(i -> (i >= 0), Profit)

            temp1 = Profit + param1[3] * log.(entrant_number)
            temp2 = round.(Pi; digits = 6)
            temp3 = round.(temp1; digits = 6)
            temp4 = zeros(eltype(Int64), firm_number[j])
            for m in 1: firm_number[j]
                temp4[m] = findall(temp2 .== temp3[m])[1]
            end
     
            
            temp_d = temp4 .<= eq_entered[j]
            decision_firm = push!(decision_firm, temp_d)
        end

        d = Vector{Int64}(undef,1)
        for m in eachindex(firm_number)
            append!(d, decision_firm[m])
        end
        d = d[2:end]
        d_temp = reshape(d, sum(potential), S)
        d_eq = Vector{Float64}(undef,1)
        for m in eachindex(potential)
            append!(d_eq, eq_firm_vec[m])
        end
        d_eq = d_eq[2:end]

        d_proj = sum(d_temp, dims = 2) ./ S

        proj_2 = reshape(enter_firm, param2.M, S)
        proj_num = sum(proj_2, dims = 2) / S
        moment = vcat((eq_firm - proj_num), (d_eq - d_proj)) 

        Q = (moment)' * (moment)

        return Q[1]
    end

end


function msm_bootstrap(param::parameters, X::AbstractVector, Z::AbstractVector, U::AbstractVector, entrant::AbstractVector, B::Int64)
    est_id = Vector{Float64}(undef,1)
    est_num = Vector{Float64}(undef,1)
    est_id_rev = Vector{Float64}(undef,1)
    est_num_rev = Vector{Float64}(undef,1)
    b = 0
    while b < B
        Z_bt = Vector{Float64}[]
        for m in eachindex(entrant)
            temp = sample(Z[m], entrant[m]; replace = true, ordered = false)
            push!(Z_bt, temp)
        end

        entered_firm, decision = eq_firm_calc(tru_param, param, X, entrant, Z_bt, U)


        opt_identity = Optim.optimize(vars -> simulated_mm(vars, param, X, Z_bt, entered_firm, decision, entrant, 50, "identity"), ones(3), Optim.Options(show_trace = false, g_tol = 1e-5))
        estimates_identity = opt_identity.minimizer
        append!(est_id, estimates_identity)
        opt_number = Optim.optimize(vars -> simulated_mm(vars, param, X, Z_bt, entered_firm, decision, entrant, 50, "number"), ones(3), Optim.Options(show_trace = false, g_tol = 1e-5))
        estimates_msm = opt_number.minimizer
        append!(est_num, estimates_identity)
        opt_identity_rev = Optim.optimize(vars -> simulated_mm(vars, param, X, Z_bt, entered_firm, decision, entrant, 50, "identityrev"), ones(3), Optim.Options(show_trace = false, g_tol = 1e-5))
        estimates_identity_rev = opt_identity_rev.minimizer
        append!(est_id_rev, estimates_identity_rev)
        opt_number_rev = Optim.optimize(vars -> simulated_mm(vars, param, X, Z_bt, entered_firm, decision, entrant, 50, "numberrev"), ones(3), Optim.Options(show_trace = false, g_tol = 1e-5))
        estimates_number_rev = opt_number_rev.minimizer
        append!(est_num_rev, estimates_number_rev)



        b += 1
    end
    return (est_id[2:end], est_num[2:end], est_id_rev[2:end], est_num_rev[2:end])
end


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#########################################   Estimation  #####################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

"""
1. Probit estimation
"""
opt_probit = Optim.optimize(vars -> entry_probit(vars, param, X, Z, entrant, entered_firm), ones(3), BFGS(), Optim.Options(show_trace = false, g_tol = 1e-7))
estimates_probit = opt_probit.minimizer
hessian_probit = hessian( vars -> entry_probit(vars, param, X, Z, entrant, entered_firm)  )
se_probit = diag(inv(hessian_probit(estimates_probit)))

println("μ estimate: ", round(estimates_probit[1], digits =4), " σ estimate: ", round(estimates_probit[2],digits = 4), " δ estimate: ", round(estimates_probit[3], digits = 4))
println("se(̂μ): ", round(se_probit[1], digits =4) , " se(̂σ): ", round(se_probit[2], digits =4), " se(̂δ): ", round(se_probit[3],digits =4) )


"""
2. Method of Simulated Moment estimation
"""
opt_identity = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, decision, entrant, 200, "identity"), ones(3), Optim.Options(show_trace = false, f_tol = 1e-7, g_tol = 1e-5))
estimates_identity = opt_identity.minimizer

opt_number = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, decision, entrant, 200, "number"), ones(3), Optim.Options(show_trace = false, f_tol = 1e-7, g_tol = 1e-5))
estimates_msm = opt_number.minimizer

opt_identity_rev = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, decision, entrant, 200, "identityrev"), ones(3), Optim.Options(show_trace = false, f_tol = 1e-7, g_tol = 1e-5))
estimates_identity_rev = opt_identity_rev.minimizer

opt_number_rev = Optim.optimize(vars -> simulated_mm(vars, param, X, Z, entered_firm, decision, entrant, 200, "numberrev"), ones(3), Optim.Options(show_trace = false, f_tol = 1e-7, g_tol = 1e-5))
estimates_msm_rev = opt_number_rev.minimizer



"""
3. Bootstrapping for standard errors
"""
B = 20
ident , num, ident_rev, num_rev = msm_bootstrap(param, X, Z, u_firm_new, entrant, B)

bt_1 = reshape(ident, 3, B)
bt_2 = reshape(num, 3, B)
bt_3 = reshape(ident_rev, 3, B)
bt_4 = reshape(num_rev,3, B)

μ_se =  sqrt(var(bt_1[1,:]))
σ_se = sqrt(var(bt_1[2,:]))
δ_se = sqrt(var(bt_1[3,:]))

μ_se_num = sqrt(var(bt_2[1,:]))
σ_se_num = sqrt(var(bt_2[2,:]))
δ_se_num = sqrt(var(bt_2[3,:]))

μ_se_rev =  sqrt(var(bt_3[1,:]))
σ_se_rev = sqrt(var(bt_3[2,:]))
δ_se_rev = sqrt(var(bt_3[3,:]))

μ_se_num_rev = sqrt(var(bt_4[1,:]))
σ_se_num_rev = sqrt(var(bt_4[2,:]))
δ_se_num_rev = sqrt(var(bt_4[3,:]))

"""
4. Results
"""

println("Specification 1: Identity and Number: Correctly specified")
println("μ estimate: ", round(estimates_identity[1], digits =4), " σ estimate: ", round(estimates_identity[2],digits = 4), " δ estimate: ", round(estimates_identity[3], digits = 4))
println("se(̂μ): ", round(μ_se, digits =4) , " se(̂σ): ", round(σ_se, digits =4), " se(̂δ): ", round(δ_se, digits =4) )

println("Specification 2: Number: Correctly specified")
println("μ estimate: ", round(estimates_msm[1], digits =4), " σ estimate: ", round(estimates_msm[2],digits = 4), " δ estimate: ", round(estimates_msm[3], digits = 4))
println("se(̂μ): ", round(μ_se_num, digits =4) , " se(̂σ): ", round(σ_se_num, digits =4), " se(̂δ): ", round(δ_se_num, digits =4) )

println("Specification 3: Identity and Number: Inorrectly specified")
println("μ estimate: ", round(estimates_identity_rev[1], digits =4), " σ estimate: ", round(estimates_identity_rev[2],digits = 4), " δ estimate: ", round(estimates_identity_rev[3], digits = 4))
println("se(̂μ): ", round(μ_se_rev, digits =4) , " se(̂σ): ", round(σ_se_rev, digits =4), " se(̂δ): ", round(δ_se_rev, digits =4) )

println("Specification 4: Number: Inorrectly specified")
println("μ estimate: ", round(estimates_msm_rev[1], digits =4), " σ estimate: ", round(estimates_msm_rev[2],digits = 4), " δ estimate: ", round(estimates_msm_rev[3], digits = 4))
println("se(̂μ): ", round(μ_se_num_rev, digits =4) , " se(̂σ): ", round(σ_se_num_rev, digits =4), " se(̂δ): ", round(δ_se_num_rev, digits =4) )





#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
################################################ Moment inequality ##########################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


"""
1. Simulation procedure (Following Ciliberto and Tamer, 2009)
    Step 1.  Transform the given matrix of epsilon draws into a draw with covariance matrix specified in θ. This is stored in ϵ^r
    Here I follow random number generating I used in the previous question, also I redeclare X and Z for this estimation
"""
X_meq = copy(X)
Z_meq = copy(z_firm_new)
entrant_meq = copy(entrant)
decision_meq = copy(decision)
uf_meq = rand(MersenneTwister(123), Normal(tru_param[1], tru_param[2]), sum(entrant_meq, dims = 1)[1])

u_firm_meq = Vector{Float64}[]
k = 1
j = 0
for i in eachindex(entrant_meq)
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


function nonparam(potential::AbstractVector, decision_eq::AbstractVector)
    n_2 = count(i -> i == 2, potential)
    n_3 = count(i -> i == 3, potential)
    n_4 = count(i -> i == 4, potential)

    p_2 = make_dmatrix(2)
    p_3 = make_dmatrix(3)
    p_4 = make_dmatrix(4)

    pr_y2 = zeros(eltype(Int64), 2^2)
    pr_y3 = zeros(eltype(Int64), 2^3)
    pr_y4 = zeros(eltype(Int64), 2^4)


    for m in eachindex(potential)
        if length(decision_eq[m]) == 2
            for j in eachindex(pr_y2)
                if decision_eq[m] == p_2[j,:]
                    pr_y2[j] += 1
                end
            end
        elseif length(decision_eq[m]) == 3
            for j in eachindex(pr_y3)
                if decision_eq[m] == p_3[j,:]
                    pr_y3[j] += 1
                end
            end
        elseif length(decision_eq[m]) == 4
            for j in eachindex(pr_y4)
                if decision_eq[m] == p_4[j,:]
                    pr_y4[j] += 1
                end
            end
        end
    end

    Pr_2 = pr_y2 / n_2
    Pr_3 = pr_y3 / n_3
    Pr_4 = pr_y4 / n_4 

    return Pr_2, Pr_3, Pr_4
end

function unobs_conversion(ϵ::AbstractVector, firm::AbstractVector)
    k = 1
    u_firm = Vector{Float64}[]
    j = 0
    for i in eachindex(firm)
        j += firm[i]
        temp = ϵ[k:j]
        u_firm = push!(u_firm, temp)
        k = j + 1
    end
    return u_firm
end     

S = 100

function Hfunction(param1::AbstractVector, param2::parameters, entrant::AbstractVector, X::AbstractVector, Z::AbstractVector, S::Int64)
    if param1[2] < 0
        param1[2] = 1
    end
    n_2 = count(i -> i == 2, entrant)
    n_3 = count(i -> i == 3, entrant)
    n_4 = count(i -> i == 4, entrant)
    simu_firm = repeat(entrant, S)
    epsi = Vector{Float64}(undef, 1)
    for s in 1:S
        temp = rand(MersenneTwister(123+s), Normal(param1[1], param1[2]), sum(entrant, dims = 1)[1])
        epsi = append!(epsi, temp)
    end
    epsi = epsi[2:end]
    epsi_meq = unobs_conversion(epsi, simu_firm)

    X_eq = repeat(X, S)
    Z_eq = repeat(Z, S)

    h_1_2 = zeros(eltype(Float64), 2^2)
    h_2_2 = zeros(eltype(Float64), 2^2)
    h_1_3 = zeros(eltype(Float64), 2^3)
    h_2_3 = zeros(eltype(Float64), 2^3)
    h_1_4 = zeros(eltype(Float64), 2^4)
    h_2_4 = zeros(eltype(Float64), 2^4)

    d_2 = make_dmatrix(2)
    d_3 = make_dmatrix(3)
    d_4 = make_dmatrix(4)


    for m in eachindex(X_eq)
        if length(Z_eq[m]) == 2
            profit = zeros(eltype(Float64), 2^2, 2)
            delta_n2 = [param1[3], param1[4]]
            for j in eachindex(h_1_2)
                for i in eachindex(Z_eq[m])
                    if d_2[j,i] == 0
                        profit[j,i] = 0
                    elseif d_2[j,i] == 1
                        profit[j,i] = X_eq[m] .- Z_eq[m][i] .- epsi_meq[m][i] .- delta_n2[1:end .!= i]' * ones(1)
                    end
                end
                
                num = count(i -> i > 0, profit[j,:])
                if (any(i-> i>0, profit[j,:]) == true) & (num == 1)
                    h_1_2[j] += 1
                end
                if any(i-> i>0, profit[j,:]) == true 
                    h_2_2[j] += 1
                end

            end
            
        elseif length(Z_eq[m]) == 3
            profit = zeros(eltype(Float64), 2^3, 3)
            delta_n3 = [param1[5] param1[6]; param1[7] param1[8] ; param1[9] param1[10]]
            for j in eachindex(h_1_3)
                for i in eachindex(Z_eq[m])
                    if d_3[j,i] == 0
                        profit[j,i] = 0
                    elseif d_3[j,i] == 1
                        profit[j,i] = X_eq[m] .- Z_eq[m][i] .- epsi_meq[m][i] .- delta_n3[i,:]' * ones(2)
                    end
                end


                num = count(i -> i > 0, profit[j,:])
                if (any(i-> i>0, profit[j,:]) == true) & (num == 1)
                    h_1_3[j] += 1
                end
                if any(i-> i>0, profit[j,:]) == true 
                    h_2_3[j] += 1
                end
            end


        elseif length(Z_eq[m]) == 4
            profit = zeros(eltype(Float64), 2^4, 4)
            delta_n4 = [param1[11] param1[12] param1[13]; param1[14] param1[15] param1[16]; param1[17] param1[18] param1[19]; param1[20] param1[21] param1[22]]
            for j in eachindex(h_1_4)
                for i in eachindex(Z_eq[m])
                    if d_4[j,i] == 0
                        profit[j,i] = 0
                    elseif d_4[j,i] == 1
                        profit[j,i] = X_eq[m] .- Z_eq[m][i] .- epsi_meq[m][i] .- delta_n4[i,:]' * ones(3)
                    end
                end
                num = count(i -> i > 0, profit[j,:])
                if (any(i-> i>0, profit[j,:]) == true) & (num == 1)
                    h_1_4[j] += 1
                end
                if any(i-> i>0, profit[j,:]) == true 
                    h_2_4[j] += 1
                end
            end
        end
        


    end
            
    h_1_2 = h_1_2 / (n_2 * S)
    h_2_2 = h_2_2 / (n_2 * S)
    h_1_3 = h_1_3 / (n_3 * S)
    h_2_3 = h_2_3 / (n_3 * S)
    h_1_4 = h_1_4 / (n_4 * S)
    h_2_4 = h_2_4 / (n_4 * S)
    return h_1_2, h_2_2, h_1_3, h_2_3, h_1_4, h_2_4
end


function Qfunction(param1::AbstractVector, param2::parameters, decision::AbstractVector, entrant::AbstractVector, X::AbstractVector, Z::AbstractVector, S::Int64)
    h12, h22, h13, h23, h14, h24 =  Hfunction(param1, param2, entrant, X, Z, S)
    H1 = vcat(h12, h13, h14)
    H2 = vcat(h22, h23, h24)
    Pr = nonparam(entrant, decision)
    p1 = Pr[1]
    p2 = Pr[2]
    p3 = Pr[3]
    P = vcat(p1, p2, p3)

    Q = (norm(P - H1) + norm(P - H2))
    a = length(P - H1) / log(length(P - H1))

    return a*Q
end

#########################################################################################################################
###################################################Estimation############################################################
#########################################################################################################################

opt_identity = Optim.optimize(vars -> Qfunction(vars, param, decision_meq, entrant_meq, X_meq, Z_meq, 100), ones(22), Optim.Options(show_trace = false, f_tol = 1e-5, g_tol = 1e-5))
opt_identity.minimizer



"""
This part is incomplete
"""
#########################################################################################################################
#################################### Confidence Interval  ###############################################################
#########################################################################################################################
#########################################################################################################################
samplesize = 7
Z_bt = Vector{Float64}[]
for i in 1:7
    for m in eachindex(entrant)
        temp = sample(MersenneTwister(123+i),Z_meq[m], entrant[m]; replace = true, ordered = false)
        push!(Z_bt, temp)
    end
end
Z_bt