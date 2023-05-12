using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra, Plots, DataFrames, CSV


df = CSV.read("simulated_data.csv", DataFrame)



df_11 = df[ ( df.N .== 1 ) .& ( df.X .== 1),: ]
df_12 = df[ ( df.N .== 1 ) .& ( df.X .== 2),: ]
df_13 = df[ ( df.N .== 1 ) .& ( df.X .== 3),: ]
df_21 = df[ ( df.N .== 2 ) .& ( df.X .== 1),: ]
df_22 = df[ ( df.N .== 2 ) .& ( df.X .== 2),: ]
df_23 = df[ ( df.N .== 2 ) .& ( df.X .== 3),: ]
df_31 = df[ ( df.N .== 3 ) .& ( df.X .== 1),: ]
df_32 = df[ ( df.N .== 3 ) .& ( df.X .== 2),: ]
df_33 = df[ ( df.N .== 3 ) .& ( df.X .== 3),: ]
df_41 = df[ ( df.N .== 4 ) .& ( df.X .== 1),: ]
df_42 = df[ ( df.N .== 4 ) .& ( df.X .== 2),: ]
df_43 = df[ ( df.N .== 4 ) .& ( df.X .== 3),: ]
df_51 = df[ ( df.N .== 5 ) .& ( df.X .== 1),: ]
df_52 = df[ ( df.N .== 5 ) .& ( df.X .== 2),: ]
df_53 = df[ ( df.N .== 5 ) .& ( df.X .== 3),: ]


e_11 = count(i -> (i == 1), df_11.Entry)/ size(df_11,1)
e_12 = count(i -> (i == 1), df_12.Entry)/ size(df_12,1)
e_13 = count(i -> (i == 1), df_13.Entry)/ size(df_13,1)
e_21 = count(i -> (i == 1), df_21.Entry)/ size(df_21,1)
e_22 = count(i -> (i == 1), df_22.Entry)/ size(df_22,1)
e_23 = count(i -> (i == 1), df_23.Entry)/ size(df_23,1)
e_31 = count(i -> (i == 1), df_31.Entry)/ size(df_31,1)
e_32 = count(i -> (i == 1), df_32.Entry)/ size(df_32,1)
e_33 = count(i -> (i == 1), df_33.Entry)/ size(df_33,1)
e_41 = count(i -> (i == 1), df_41.Entry)/ size(df_41,1)
e_42 = count(i -> (i == 1), df_42.Entry)/ size(df_42,1)
e_43 = count(i -> (i == 1), df_43.Entry)/ size(df_43,1)
e_51 = count(i -> (i == 1), df_51.Entry)/ size(df_51,1)
e_52 = count(i -> (i == 1), df_52.Entry)/ size(df_52,1)
e_53 = count(i -> (i == 1), df_53.Entry)/ size(df_53,1)

ehat = [e_11 e_12 e_13 ; e_21 e_22 e_23 ; e_31 e_32 e_33; e_41 e_42 e_43; e_51 e_52 e_53]




d_11 = count(i -> (i == 0), df_11.Exit)/ size(df_11,1)
d_12 = count(i -> (i == 0), df_12.Exit)/ size(df_12,1) # Staying
d_13 = count(i -> (i == 0), df_13.Exit)/ size(df_13,1)


d_21 = (binomial(2,2) * (count(i -> (i == 0), df_21.Exit)/ size(df_21,1))^2  + binomial(2,1) * count(i -> (i == 1), df_21.Exit)/ size(df_21,1) * (1 - count(i -> (i == 1), df_21.Exit)/ size(df_21,1)) + binomial(2,0) *  (1 - count(i -> (i == 2), df_21.Exit)/ size(df_21,1))^2 ) / 2
d_22 = (binomial(2,2) * (count(i -> (i == 0), df_22.Exit)/ size(df_22,1))^2  + binomial(2,1) * count(i -> (i == 1), df_22.Exit)/ size(df_22,1) * (1 - count(i -> (i == 1), df_22.Exit)/ size(df_22,1)) + binomial(2,0) *  (1 - count(i -> (i == 2), df_22.Exit)/ size(df_22,1))^2  ) / 2
d_23 = (binomial(2,2) * (count(i -> (i == 0), df_23.Exit)/ size(df_23,1))^2  + binomial(2,1) * count(i -> (i == 1), df_23.Exit)/ size(df_23,1) * (1 - count(i -> (i == 1), df_23.Exit)/ size(df_23,1)) + binomial(2,0) *  (1- count(i -> (i == 2), df_23.Exit)/ size(df_23,1))^2 ) / 2

d_31 = (binomial(3,3) * (count(i -> (i == 0), df_31.Exit)/ size(df_31,1))^3  + binomial(3,1) * count(i -> (i == 1), df_31.Exit)/ size(df_31,1) * (1 - count(i -> (i == 1), df_31.Exit)/ size(df_31,1))^2 + binomial(3,2) *  (count(i -> (i == 2), df_31.Exit)/ size(df_31,1))^2  * (1- (count(i -> (i == 2), df_31.Exit)/ size(df_31,1))) + binomial(3,3) *  (1 - count(i -> (i == 3), df_31.Exit)/ size(df_31,1))^3 ) / 3
d_32 = (binomial(3,3) * (count(i -> (i == 0), df_32.Exit)/ size(df_32,1))^3  + binomial(3,1) * count(i -> (i == 1), df_32.Exit)/ size(df_32,1) * (1 - count(i -> (i == 1), df_32.Exit)/ size(df_32,1))^2 + binomial(3,2) *  (count(i -> (i == 2), df_32.Exit)/ size(df_32,1))^2  * (1- (count(i -> (i == 2), df_32.Exit)/ size(df_32,1))) + binomial(3,3) *  (1 - count(i -> (i == 3), df_32.Exit)/ size(df_32,1))^3 ) / 3
d_33 = (binomial(3,3) * (count(i -> (i == 0), df_33.Exit)/ size(df_33,1))^3  + binomial(3,1) * count(i -> (i == 1), df_33.Exit)/ size(df_33,1) * (1 - count(i -> (i == 1), df_33.Exit)/ size(df_33,1))^2 + binomial(3,2) *  (count(i -> (i == 2), df_33.Exit)/ size(df_33,1))^2  * (1- (count(i -> (i == 2), df_33.Exit)/ size(df_33,1))) + binomial(3,3) *  (1 - count(i -> (i == 3), df_32.Exit)/ size(df_32,1))^3 ) / 3


d_41 = (binomial(4,4) * (count(i -> (i == 0), df_41.Exit)/ size(df_41,1))^4  + binomial(4,1) * count(i -> (i == 1), df_41.Exit)/ size(df_41,1) * (1 - count(i -> (i == 1), df_41.Exit)/ size(df_41,1))^3 + binomial(4,2) *  (count(i -> (i == 2), df_41.Exit)/ size(df_41,1))^2  * (1- (count(i -> (i == 2), df_41.Exit)/ size(df_41,1)))^2 + binomial(4,3) *  (count(i -> (i == 3), df_41.Exit)/ size(df_41,1))^3 * (1 - count(i -> (i == 3), df_41.Exit)/ size(df_41,1)) + binomial(4,4) * (1 - count(i -> (i == 4), df_41.Exit)/ size(df_41,1))^4 ) /4
d_42 = (binomial(4,4) * (count(i -> (i == 0), df_42.Exit)/ size(df_42,1))^4  + binomial(4,1) * count(i -> (i == 1), df_42.Exit)/ size(df_42,1) * (1 - count(i -> (i == 1), df_42.Exit)/ size(df_42,1))^3 + binomial(4,2) *  (count(i -> (i == 2), df_42.Exit)/ size(df_42,1))^2  * (1- (count(i -> (i == 2), df_42.Exit)/ size(df_42,1)))^2 + binomial(4,3) *  (count(i -> (i == 3), df_42.Exit)/ size(df_42,1))^3 * (1 - count(i -> (i == 3), df_42.Exit)/ size(df_42,1)) + binomial(4,4) * (1 - count(i -> (i == 4), df_42.Exit)/ size(df_42,1))^4 ) /4
d_43 = (binomial(4,4) * (count(i -> (i == 0), df_43.Exit)/ size(df_43,1))^4  + binomial(4,1) * count(i -> (i == 1), df_43.Exit)/ size(df_43,1) * (1 - count(i -> (i == 1), df_43.Exit)/ size(df_43,1))^3 + binomial(4,2) *  (count(i -> (i == 2), df_43.Exit)/ size(df_43,1))^2  * (1- (count(i -> (i == 2), df_43.Exit)/ size(df_43,1)))^2 + binomial(4,3) *  (count(i -> (i == 3), df_43.Exit)/ size(df_43,1))^3 * (1 - count(i -> (i == 3), df_42.Exit)/ size(df_42,1)) + binomial(4,4) * (1 - count(i -> (i == 4), df_42.Exit)/ size(df_42,1))^4 ) /4

d_51 = (binomial(5,5) * (count(i -> (i == 0), df_51.Exit)/ size(df_51,1))^5  + binomial(5,1) * count(i -> (i == 1), df_51.Exit)/ size(df_51,1) * (1 - count(i -> (i == 1), df_51.Exit)/ size(df_51,1))^4 + binomial(5,2) *  (count(i -> (i == 2), df_51.Exit)/ size(df_51,1))^2  * (1- (count(i -> (i == 2), df_51.Exit)/ size(df_51,1)))^3 + binomial(5,3) *  (count(i -> (i == 3), df_51.Exit)/ size(df_51,1))^3 * (1 - count(i -> (i == 3), df_51.Exit)/ size(df_51,1))^2 + binomial(5,4) * (count(i -> (i == 4), df_51.Exit)/ size(df_51,1))^4 * (1 - count(i -> (i == 4), df_51.Exit)/ size(df_51,1)) + binomial(5,5) * (1 - count(i -> (i == 5), df_51.Exit)/ size(df_51,1))^5) /5
d_52 = (binomial(5,5) * (count(i -> (i == 0), df_52.Exit)/ size(df_52,1))^5  + binomial(5,1) * count(i -> (i == 1), df_52.Exit)/ size(df_52,1) * (1 - count(i -> (i == 1), df_52.Exit)/ size(df_52,1))^4 + binomial(5,2) *  (count(i -> (i == 2), df_52.Exit)/ size(df_52,1))^2  * (1- (count(i -> (i == 2), df_52.Exit)/ size(df_52,1)))^3 + binomial(5,3) *  (count(i -> (i == 3), df_52.Exit)/ size(df_52,1))^3 * (1 - count(i -> (i == 3), df_52.Exit)/ size(df_52,1))^2 + binomial(5,4) * (count(i -> (i == 4), df_52.Exit)/ size(df_52,1))^4 * (1 - count(i -> (i == 4), df_52.Exit)/ size(df_52,1)) + binomial(5,5) * (1 - count(i -> (i == 5), df_52.Exit)/ size(df_52,1))^5) /5
d_53 = (binomial(5,5) * (count(i -> (i == 0), df_53.Exit)/ size(df_53,1))^5  + binomial(5,1) * count(i -> (i == 1), df_53.Exit)/ size(df_53,1) * (1 - count(i -> (i == 1), df_53.Exit)/ size(df_53,1))^4 + binomial(5,2) *  (count(i -> (i == 2), df_53.Exit)/ size(df_53,1))^2  * (1- (count(i -> (i == 2), df_53.Exit)/ size(df_53,1)))^3 + binomial(5,3) *  (count(i -> (i == 3), df_53.Exit)/ size(df_53,1))^3 * (1 - count(i -> (i == 3), df_53.Exit)/ size(df_53,1))^2 + binomial(5,4) * (count(i -> (i == 4), df_53.Exit)/ size(df_53,1))^4 * (1 - count(i -> (i == 4), df_53.Exit)/ size(df_53,1)) + binomial(5,5) * (1 - count(i -> (i == 5), df_53.Exit)/ size(df_53,1))^5) /5


dhat = [d_11 d_12 d_13 ; d_21 d_22 d_23 ; d_31 d_32 d_33; d_41 d_42 d_43; d_51 d_52 d_53]

ehat
dhat


function forward_simulation_incumbent(x, firmnum, transition, entry_thres, exit_thres, parameters)

    weight_x1 = aweights(transition[1,:])
    weight_x2 = aweights(transition[2,:])
    weight_x3 = aweights(transition[3,:])
    x_path = ones(eltype(Int64),1) * x
    nlist = ones(eltype(Int64),1) * firmnum
    staying_hist = Vector{Int64}(undef,1)
    t = 0 
    time_idx = zeros(eltype(Int64),1)
    scrap = zeros(eltype(Float64),0)
    while t < 10000
        
        t += 1
        append!(time_idx, t)
        today_n = nlist[end]
        x_today = x_path[end]
       
        if today_n == 0
            today_n = 1
        end
       
        mu_draw_i = rand(Normal(parameters[1], parameters[2]), 1)
        exit_value = exit_thres[today_n, x_today]
        remain_decision_i = sum(mu_draw_i .< exit_value)
       
        if remain_decision_i == 0
            append!(staying_hist, remain_decision_i)
            append!(scrap, exit_value)
            break
        else
            if x_today == 1
                x_tmr = sample([1,2,3], weight_x1, 1)[]
            elseif x_today == 2
                x_tmr = sample([1,2,3], weight_x2, 1)[]
            elseif x_today == 3
                x_tmr = sample([1,2,3], weight_x3, 1)[]
            end
    
    

            if today_n == 1
        
                entrant_draw = rand(Normal(parameters[3], parameters[4]), 1)
                entry_value = entry_thres[today_n+1, x_tmr]
                entry_decision = sum(entrant_draw .< entry_value)
                tmr_n = today_n + entry_decision


            elseif today_n == 5

                exit_draw = rand(Normal(parameters[1], parameters[2]), today_n - 1)
                exit_value = ones(today_n - 1) .* exit_thres[today_n - 1 ,x_tmr]
                tmr_n = today_n - sum(exit_draw .> exit_value)
            else

                exit_draw = rand(Normal(parameters[1], parameters[2]), today_n - 1)
                exit_value = ones(today_n - 1) .* exit_thres[today_n - 1, x_tmr]
                entrant_draw = rand(Normal(parameters[3], parameters[4]), 1)
                entry_value = entry_thres[today_n+1, x_tmr]
                entry_decision = sum(entrant_draw .< entry_value)
                tmr_n = today_n - sum(entrant_draw .> entry_value) + entry_decision
        
            end
            exit_value = 0
            append!(staying_hist,remain_decision_i)
            append!(scrap, exit_value)
            append!(nlist, tmr_n)
            append!(x_path, x_tmr)
        end

    end
    res = 0.0
    for i in eachindex(x_path)
    if staying_hist[2:end][i] == 1
        res += 0.9^time_idx[2:end][i] * profit(x_path[i], nlist[i])
    elseif staying_hist[2:end][i] == 0
        res += 0.9^time_idx[2:end][i] * scrap[i]
    end
    end

    return res
end



function forward_simulation_entrant(x, firmnum, transition, entry_thres, exit_thres, parameters)

    weight_x1 = aweights(transition[1,:])
    weight_x2 = aweights(transition[2,:])
    weight_x3 = aweights(transition[3,:])
    x_path = ones(eltype(Int64),1) .* x
    nlist = ones(eltype(Int64),1) .* firmnum
    staying_hist = Vector{Int64}(undef,1)
    scrap = zeros(eltype(Float64),0)

    t = 0 
    time_idx = zeros(eltype(Int64),1)
    entry_value = zeros(eltype(Float64),0)
    while t < 10000
        

        t += 1
        append!(time_idx, t)
        today_n = nlist[end]
      
        if today_n == 0
            today_n = 1
        end

        x_today = x_path[end]
        mu_draw_i = rand(Normal(parameters[1], parameters[2]), 1)
        exit_value = exit_thres[today_n, x_today]
        remain_decision_i = sum(mu_draw_i .< exit_value)
        
    
        if remain_decision_i == 0
            append!(staying_hist, remain_decision_i)
            append!(scrap, exit_value)
            break
        else
    

            if today_n == 1
        
                entrant_draw = rand(Normal(parameters[3], parameters[4]), 1)
                entry_value = entry_thres[today_n+1, x_today]
                entry_decision = sum(entrant_draw .< entry_value)
                tmr_n = today_n + entry_decision
                if x_today == 1
                    x_tmr = sample([1,2,3], weight_x1, 1)[]
                elseif x_today == 2
                    x_tmr = sample([1,2,3], weight_x2, 1)[]
                elseif x_today == 3
                    x_tmr = sample([1,2,3], weight_x3, 1)[]
                end

            elseif today_n == 5

                exit_draw = rand(Normal(parameters[1], parameters[2]), today_n)
                exit_value = ones(today_n) .* exit_thres[today_n ,x_today]
                tmr_n = today_n - sum(exit_draw .> exit_value)
                if x_today == 1
                    x_tmr = sample([1,2,3], weight_x1, 1)[]
                elseif x_today == 2
                    x_tmr = sample([1,2,3], weight_x2, 1)[]
                elseif x_today == 3
                    x_tmr = sample([1,2,3], weight_x3, 1)[]
                end
            else

                exit_draw = rand(Normal(parameters[1], parameters[2]), today_n)
                exit_value = ones(today_n) .* exit_thres[today_n, x_today]
                entrant_draw = rand(Normal(parameters[3], parameters[4]), 1)
                entry_value = entry_thres[today_n+1, x_today]
                entry_decision = sum(entrant_draw .< entry_value)
                tmr_n = today_n - sum(entrant_draw .> entry_value) + entry_decision
                if x_today == 1
                    x_tmr = sample([1,2,3], weight_x1, 1)[]
                elseif x_today == 2
                    x_tmr = sample([1,2,3], weight_x2, 1)[]
                elseif x_today == 3
                    x_tmr = sample([1,2,3], weight_x3, 1)[]
                end
            end
            exit_value = 0
            append!(staying_hist,remain_decision_i)
            append!(scrap, exit_value)
            append!(nlist, tmr_n)
            append!(x_path, x_tmr)
        end

    end
    res = 0.0
    for i in eachindex(x_path)
        if staying_hist[2:end][i] == 1
            res += 0.9^time_idx[2:end][i] * profit(x_path[i], nlist[i])
        elseif staying_hist[2:end][i] == 0
            res += 0.9^time_idx[2:end][i] * scrap[i]
    end
    end

    return res
end


for j in eachindex(S)
    while iter < S[j]
        simu += forward_simulation_entrant(1, 1, transition, gamma_1, mu_1, param)
        iter += 1
    end
    pdv = simu/( S[j])
    append!(pdvlist, pdv)
end


function pdv_calc(S, x, fnum, transition, entry_thres, exit_thres, parameters)
    pdv = 0.0
    iter = 0
    simu_entr = 0.0
    simu_incumbent = 0.0
    iter
    while iter < S
        simu_entr = simu_entr + forward_simulation_entrant(x, fnum, transition, entry_thres, exit_thres, parameters)
        simu_incumbent = simu_incumbent +  forward_simulation_incumbent(x, fnum, transition, entry_thres, exit_thres, parameters)
        iter += 1
    end
    
    
    pdv = [simu_entr/ S, simu_incumbent / S]
        
    return pdv
end


function Qfunction(parameter, ehat, dhat)
    if parameter[2] < 0
        parameter[2] = 2
    elseif parameter[4] < 0
        parameter[4] = 2
    end

    entry_value = pdf.(Normal(parameter[1], parameter[2]), ehat)
    exit_value = pdf.(Normal(parameter[3], parameter[4]), dhat)
    state_list = [1,2,3]
    fnum_list = [1,2,3,4,5]
    Lambda = zeros(eltype(Float64), 5,3)
    Lambda_e = zeros(eltype(Float64), 5,3)
    for i in 1:size(fnum_list,1)
        for j in 1:size(state_list,1)
            temp = pdv_calc(10000, state_list[j], fnum_list[i], transition, entry_value, exit_value, param)
            Lambda[i,j] = temp[2]
            Lambda_e[i,j] = temp[1]
        end
    end
    res = sum(( cdf(Normal(0,1), (Lambda .- parameter[1]) / parameter[2] )   - dhat).^2) + sum(( cdf(Normal(0,1), (Lambda_e .- parameter[3]) / parameter[4] )   - ehat).^2)
    return res
end


opt = Optim.optimize(vars -> Qfunction(vars, ehat, dhat), ones(4), Optim.Options(show_trace = true, f_tol = 1e-4, g_tol = 1e-4))
estimates_identity = opt.minimizer




