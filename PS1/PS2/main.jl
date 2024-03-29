using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra, Plots, DataFrames, CSV

################################################################################################
################################################################################################
######################################   DGP  ##################################################
################################################################################################
################################################################################################

transition = [0.6 0.2 0.2 ; 0.2 0.6 0.2; 0.2 0.2 0.6]
temp = copy(transition)
β = 0.9
N_max = 5
s_max = 3


gamma_mu = 5.0
gamma_sigma = sqrt(5.0)
mu_mu = 5.0
mu_sigma = sqrt(5.0)
param = [gamma_mu, gamma_sigma, mu_mu, mu_sigma]

function profit(x, n)

    pi = ((10 .+ x )' ./ (n .+ 1)).^2 .- 5


    return pi
end


x_state = [-5,0,5]
firm_num = [1,2,3,4,5]
profit(x_state, firm_num)


### Transition matrix

function tr(init_gamma, init_mu, x_state, param)
    gamma_mean = param[1]
    gamma_sigma = param[2]
    mu_mean = param[3]
    mu_sigma = param[4] 
     # gamma(N_t = 0, x) is unnecessary since Pr(N_t+1 |N_t = 0) = 1 always
    gamma_1 = cdf(Normal(gamma_mean,gamma_sigma), init_gamma[1, x_state]) # gamma(1, x)
    gamma_2 = cdf(Normal(gamma_mean,gamma_sigma), init_gamma[2, x_state]) # gamma(2, x)
    gamma_3 = cdf(Normal(gamma_mean,gamma_sigma), init_gamma[3, x_state]) # gamma(3, x)
    gamma_4 = cdf(Normal(gamma_mean,gamma_sigma), init_gamma[4, x_state]) # gamma(4, x)
    mu_1 = cdf(Normal(mu_mean,mu_sigma), init_mu[1, x_state]) # mu(1, x)
    mu_2 = cdf(Normal(mu_mean,mu_sigma), init_mu[2, x_state]) # mu(2, x)
    mu_3 = cdf(Normal(mu_mean,mu_sigma), init_mu[3, x_state]) # mu(3, x)
    mu_4 = cdf(Normal(mu_mean,mu_sigma), init_mu[4, x_state]) # mu(4, x)
    mu_5 = cdf(Normal(mu_mean,mu_sigma), init_mu[5, x_state]) # mu(5, x)

    tr_d = [ 
    (1 - gamma_1) 
    gamma_1 
    0 
    0 
    0 ;

    (1 - gamma_2) * (1 - mu_2) 
    (gamma_2 * (1 - mu_2) + (1- gamma_2) * (mu_2)) 
    (gamma_2 * mu_2) 
    0 
    0 ;

    ((1 - gamma_3) * (1 - mu_3)^2) 
    ((gamma_3) * (1 - gamma_3)^2) + (1- gamma_3) * (binomial(2,1) * (1- mu_3) * mu_3)    
    ((1 - gamma_3) * mu_3^2 + (1- gamma_3) * binomial(2,1) * (1- mu_3) * mu_3) 
    (gamma_3) * mu_3^2   
    0 ;

    (1 - gamma_4) * (1- mu_4)^3   
    ((1 - gamma_4) * (binomial(3,2) * (1- mu_4)^2 * mu_4) + gamma_4 * (1- mu_4)^3) 
    (1- gamma_4) * binomial(3,1) * (1- mu_4) * mu_4^2 + gamma_4 * binomial(3,1) * mu_4 * (1- mu_4)^2 
    (1- gamma_4) * mu_4^3 + gamma_4 * binomial(3,1) * mu_4^2 * (1- mu_4) 
    gamma_4 * mu_4^3 ;

    (1 - mu_5)^4
    binomial(4,1) * (1-mu_5)^3 * mu_5
    binomial(4,2) * (1-mu_5)^2 * mu_5^2
    binomial(4,3) * (1-mu_5)^1 * mu_5^3
    binomial(4,4) * mu_5^4
    ]

    tr_e = [
    


    (1 - mu_1)
    mu_1 
    0
    0
    0   ;

    (1- mu_2)^2
    binomial(2,1) * mu_2 * (1- mu_2)
    mu_2^2
    0
    0   ;
    (1 - mu_3)^3
    binomial(3,1) * mu_3 * (1- mu_3)^2
    binomial(3,2) * mu_3^2 * (1- mu_3)
    mu_3^3
    0   ;

    (1- mu_4)^4
    binomial(4,1) * mu_4 * (1- mu_4)^3
    binomial(4,2) * mu_4^2 * (1- mu_4)^2
    binomial(4,3) * mu_4^3 * (1- mu_4)^1
    mu_4^4 ;

    0
    0
    0
    0
    0
    ]

    return tr_d, tr_e
end



function equilibrium(init_mu, init_gamma, init_Vbar, transition, x_state, firm_num, param)
    tol = 1.0
    iter = 1
    while tol > 1e-13
        tr_d_x1, tr_e_x1 = tr(init_gamma, init_mu, 1, param)
        tr_d_x2, tr_e_x2 = tr(init_gamma, init_mu, 2, param)
        tr_d_x3, tr_e_x3 = tr(init_gamma, init_mu, 3, param)
        
        tr_d_x1 = reshape(tr_d_x1, (5,5))'
        tr_d_x2 = reshape(tr_d_x2, (5,5))'
        tr_d_x3 = reshape(tr_d_x3, (5,5))'
        
        tr_e_x1 = reshape(tr_e_x1, (5,5))'
        tr_e_x2 = reshape(tr_e_x2, (5,5))'
        tr_e_x3 = reshape(tr_e_x3, (5,5))'

        

        Vbar_x1 = (init_Vbar[:,1]' * tr_d_x1)'
        Vbar_x2 = (init_Vbar[:,2]' * tr_d_x2)'
        Vbar_x3 = (init_Vbar[:,3]' * tr_d_x3)'

        Vbar_d_x1 = (init_Vbar[:,1]' * tr_e_x1)'
        Vbar_d_x2 = (init_Vbar[:,2]' * tr_e_x2)'
        Vbar_d_x3 = (init_Vbar[:,3]' * tr_e_x3)'


        Vbar_nprime = hcat(Vbar_x1, Vbar_x2, Vbar_x3)
        Vbar_dnprime = hcat(Vbar_d_x1, Vbar_d_x2, Vbar_d_x3)


        psi_1 = 0.9 .* (Vbar_nprime * transition)
        psi_2 = 0.9 .* (Vbar_dnprime * transition)

        

        newmu =  0.2 * profit(x_state, firm_num) .+ 0.8 * psi_1
        newgamma = psi_2
        
        
        A = ( 1 .- cdf(Normal(0,1), (newmu - init_mu)/  sqrt(5))) .* (init_mu .+  sqrt(5) .* (pdf.(Normal(0,1), (newmu- init_mu) ./  sqrt(5)) ./  ( 1 .- cdf(Normal(0,1), (newmu - init_mu)./  sqrt(5))) ))
        replace!(A, NaN => 0.00001)
        B = (cdf(Normal(0,1), (newmu - init_mu)/  sqrt(5))) .* newmu
        new_Vbar = A + B


        old = [init_mu, init_gamma, init_Vbar]
        new = [newmu, newgamma, new_Vbar]
        tol = norm(old.-new)
        iter += 1

        init_mu = newmu
        init_gamma = newgamma
        init_Vbar = new_Vbar
        if iter & 100000 == 0
            @show (iter, tol)
        end
    end 
    return init_mu, init_gamma, init_Vbar
end







init_mu_1 = profit(x_state, firm_num)
init_gamma_1 = profit(x_state, firm_num) 
init_Vbar_1 =  profit(x_state, firm_num) 


mu_1, gamma_1, Vbar_1 = equilibrium(init_mu_1, init_gamma_1, init_Vbar_1, transition, x_state, firm_num, param)
test1 = [mu_1; gamma_1; Vbar_1]


init_mu_2 = ones(5,3) * 2
init_gamma_2 = ones(5,3) *2
init_Vbar_2 = ones(5,3) * 2
mu_2, gamma_2, Vbar_2 = equilibrium(init_mu_2, init_gamma_2, init_Vbar_2, transition, x_state, firm_num, param)
test2 = [mu_2; gamma_2; Vbar_2]



init_mu_3 = ones(5,3) * 3
init_gamma_3 = ones(5,3) * 3
init_Vbar_3 = ones(5,3) * 3
mu_3, gamma_3, Vbar_3 = equilibrium(init_mu_3, init_gamma_3, init_Vbar_3, transition, x_state, firm_num, param)
test3 = [mu_3; gamma_3; Vbar_3]


init_mu_4 = ones(5,3) * 4
init_gamma_4 = ones(5,3) * 4
init_Vbar_4 = ones(5,3) * 4
mu_4, gamma_4, Vbar_4 = equilibrium(init_mu_4, init_gamma_4, init_Vbar_4, transition, x_state, firm_num, param)
test4 = [mu_4; gamma_4; Vbar_4]


init_mu_5 = ones(5,3) * 5
init_gamma_5 = ones(5,3) * 5
init_Vbar_5 = ones(5,3) * 5
mu_5, gamma_5, Vbar_5 = equilibrium(init_mu_5, init_gamma_5, init_Vbar_5, transition, x_state, firm_num, param)
test5 = [mu_5; gamma_5; Vbar_5]

#test this for different init values


testing = norm(test1 - test2) < 1e-10
testing2 = norm(test2- test3) < 1e-10
testing3 = norm(test2 - test3) < 1e-10
testing4 = norm(test3 - test4) < 1e-10
testing5 = norm(test4 - test5) < 1e-10


firm_num_1 = [1,2,3,4]


plot(firm_num, [
    mu_1[:,1],
    mu_1[:,2],
    mu_1[:,3]],  label=["X_t = -5" "X_t = 0" "X_t = 5"])
title!("Comparative Statics")
xlabel!("Number of Firms")
ylabel!("Scarp Value")
savefig("myplot.png")


plot(firm_num_1, [
    gamma_1[2:end,1],
    gamma_1[2:end,2],
    gamma_1[2:end,3]],  label=["X_t = -5" "X_t = 0" "X_t = 5"])
title!("Comparative Statics")
xlabel!("Number of Firms")
ylabel!("Entry Cost")
savefig("myplot1.png")

plot(firm_num, [
    Vbar_1[:,1],
    Vbar_1[:,2],
    Vbar_1[:,3]],  label=["X_t = -5" "X_t = 0" "X_t = 5"])
title!("Comparative Statics")
xlabel!("Number of Firms")
ylabel!("Value Function")
savefig("myplot2.png")


function firm_num_simu(today_n, state_idx, mu_eq, gamma_eq)
    if today_n > 5
        today_n = 5
    end
    
    if today_n == 0
        gamma_simu = rand(Normal(param[3], param[4]),1)
        tmr_n = today_n + 1    
        entry_decision = 1
        exit_decision = 0
    elseif today_n == 5
        mu_simu = rand(Normal(param[1], param[2]), today_n)
        exit_value = ones(today_n, 1).* (mu_eq[today_n-1, state_idx])
        tmr_n = today_n - sum(mu_simu .> exit_value)
        entry_decision = 0
        exit_decision = (mu_simu .> exit_value)
    elseif today_n == 1
        gamma_simu = rand(Normal(param[3],param[4]),1)
        mu_simu = rand(Normal(param[1],param[2]), today_n)
        entry_value = gamma_eq[2, state_idx]
        exit_value = mu_eq[1, state_idx]
        tmr_n = today_n - sum(mu_simu .> exit_value) + sum(gamma_simu .< entry_value) 
        entry_decision = sum(gamma_simu .< entry_value)
        exit_decision = (mu_simu .> exit_value)
    else
        gamma_simu = rand(Normal(param[3],param[4]),1)
        mu_simu = rand(Normal(param[1],param[2]), today_n)
        entry_value = gamma_eq[today_n, state_idx]
        exit_value = mu_eq[today_n-1, state_idx] .* ones(today_n)
        tmr_n = today_n - sum(mu_simu .> exit_value) + sum(gamma_simu .< entry_value) 
        entry_decision = sum(gamma_simu .< entry_value)
        exit_decision = (mu_simu .> exit_value)
    end

    return tmr_n, entry_decision, exit_decision
end



function firm_num_simu_withtax(today_n, state_idx, mu_eq, gamma_eq)
    if today_n > 5
        today_n = 5
    end
    
    if today_n == 0
        gamma_simu = rand(Normal(param[3], param[4]),1)
        tmr_n = today_n + 1   
        entry_decision = 1
        exit_decision = 0
    elseif today_n == 5
        mu_simu = rand(Normal(param[1], param[2]), today_n)
        exit_value = ones(today_n, 1).* (mu_eq[today_n-1, state_idx])
        tmr_n = today_n - sum(mu_simu .> exit_value)
        entry_decision = 0
        exit_decision = (mu_simu .> exit_value)
    elseif today_n == 1
        gamma_simu = rand(Normal(param[3],param[4]),1)
        mu_simu = rand(Normal(param[1],param[2]), today_n)
        entry_value = gamma_eq[2, state_idx]
        exit_value = mu_eq[1, state_idx]
        tmr_n = today_n - sum(mu_simu .> exit_value) + sum(gamma_simu .+15.0 .< entry_value ) 
        entry_decision = sum(gamma_simu .+15.0 .< entry_value )
        exit_decision = (mu_simu .> exit_value)
    else
        gamma_simu = rand(Normal(param[3],param[4]),1)
        mu_simu = rand(Normal(param[1],param[2]), today_n)
        entry_value = gamma_eq[today_n, state_idx]
        exit_value = mu_eq[today_n-1, state_idx] .* ones(today_n)
        tmr_n = today_n - sum(mu_simu .> exit_value) + sum(gamma_simu .+15.0 .< entry_value  ) 
        entry_decision = sum(gamma_simu .+ 15.0 .< entry_value )
        exit_decision = (mu_simu .> exit_value)
    end

    return tmr_n, entry_decision, exit_decision
end



function simulation(transition, simu_number)

    weight_x1 = aweights(transition[1,:])
    weight_x2 = aweights(transition[2,:])
    weight_x3 = aweights(transition[3,:])
    iter = 1

    nlist_1 = zeros(eltype(Int64),1)
    nlist_1_tax = zeros(eltype(Int64),1)
    entry_decision_1 = zeros(eltype(Int64),1)
    exit_decision_1 = zeros(eltype(Int64),1)
    x_path = zeros(eltype(Int64),1)
    xstate = 1
    nprime_x1 = 0
    nprime_x1_tax = 0
    entry_x11 = 0
    exit_x11 = 0
    while iter < simu_number
    
        push!(nlist_1, nprime_x1)
        nprime_x11, entry_x11, exit_x11 = firm_num_simu(nprime_x1, xstate, mu_1, gamma_1)
        push!(entry_decision_1, entry_x11)
        push!(exit_decision_1, sum(exit_x11))
        nprime_x1 = nprime_x11



        push!(nlist_1_tax, nprime_x1_tax)
        nprime_x11_tax, none1, none2 = firm_num_simu_withtax(nprime_x1_tax, xstate, mu_1, gamma_1)
        nprime_x1_tax = nprime_x11_tax

        push!(x_path, xstate)


        if xstate == 1
            xstate = sample([1,2,3], weight_x1, 1)[]
        elseif xstate == 2
            xstate = sample([1,2,3], weight_x2, 1)[]
        elseif xstate == 3
            xstate = sample([1,2,3], weight_x3, 1)[]
        end
        iter += 1
    end
    return nlist_1[2:end], nlist_1_tax[2:end], x_path[2:end], entry_decision_1[2:end], exit_decision_1[2:end]
end

simu_iter = 0
nt_fnum = zeros(0)
t_fnum = zeros(0)

while simu_iter < 100
    notax_fnum, tax_fnum, X, Entry, Exit = simulation(transition, 100001)
    mean_fnum = mean(notax_fnum)
    tax_mean_fnum = mean(tax_fnum)
    push!(nt_fnum, mean_fnum)
    push!(t_fnum, tax_mean_fnum)
    simu_iter += 1
end

nt_fnum
t_fnum

brange = range(2.2, 2.7, length=100)
histogram(Any[nt_fnum, t_fnum], bins =brange, label=["Without Tax" "With Tax"])
title!("Average Firm Number")
xlabel!("Firm Number")
ylabel!("Frequency")
savefig("Tax_notax.png")


firm_number, none, x_simulate_state, entry, exit = simulation(transition, 1000001)
data = DataFrame(N= firm_number, X= x_simulate_state, Entry = entry, Exit= exit)

CSV.write("simulated_data.csv", data)



################################################################################################
################################################################################################
###############################   Estimation  ##################################################
################################################################################################
################################################################################################

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


function Qfunction(parameter, ehat, dhat, simulation_num)
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
            temp = pdv_calc(simulation_num, state_list[j], fnum_list[i], transition, entry_value, exit_value, param)
            Lambda[i,j] = temp[2]
            Lambda_e[i,j] = temp[1]
        end
    end
    res = sum(( cdf(Normal(0,1), (Lambda .- parameter[1]) / parameter[2] )   - dhat).^2) + sum(( cdf(Normal(0,1), (Lambda_e .- parameter[3]) / parameter[4] )   - ehat).^2)
    return res
end

simulation_num = 10000
opt = Optim.optimize(vars -> Qfunction(vars, ehat, dhat, simulation_num ), ones(4),Optim.Options(show_trace = true, f_tol = 1e-3, g_tol = 1e-3))
estimates_identity = opt.minimizer






