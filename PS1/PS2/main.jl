using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra, Plots


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
    while tol > 1e-14
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
firm_num_1 = [0,1,2,3,4]

mu_1[:,1]
mu_1[:,2]
mu_1[:,3]

plot(firm_num, [
    mu_1[:,1],
    mu_1[:,2],
    mu_1[:,3]],  label=["X_t = -5" "X_t = 0" "X_t = 5"])
title!("Comparative Statics")
xlabel!("Number of Firms")
ylabel!("Scarp Value")
savefig("myplot.png")

plot(firm_num_1, [
    gamma_1[:,1],
    gamma_1[:,2],
    gamma_1[:,3]],  label=["X_t = -5" "X_t = 0" "X_t = 5"])
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
ylabel!("Entry Cost")
savefig("myplot2.png")


function firm_num_simu(today_n, state_idx, mu_eq, gamma_eq)
    
    if today_n == 0
        gamma_simu = rand(Normal(param[3], param[4]),1)
        entry_value = gamma_eq[1, state_idx]
        tmr_n = today_n + sum(gamma_simu .> entry_value)    
        entry_decision = sum(gamma_simu .> entry_value)
        exit_decision = 0
    elseif today_n == 5
        mu_simu = rand(Normal(param[1], param[2]), today_n)
        exit_value = ones(today_n, 1).* (mu_eq[today_n, state_idx])
        tmr_N = today_n - sum(mu_simu .> exit_value)
        entry_decision = 0
        exit_decision = (mu_simu .> exit_value)
    else
        gamma_simu = rand(Normal(param[3],param[4]),1)
        mu_simu = rand(Normal(param[1],param[2]), today_n)
        entry_value = gamma_eq[today_n, state_idx]
        exit_value = mu_eq[today_n, state_idx] .* ones(today_n)
        tmr_n = today_n - sum(mu_simu .> exit_value) + sum(gamma_simu .> entry_value) 
        entry_decision = sum(gamma_simu .> entry_value)
        exit_decision = (mu_simu .> exit_value)
    end
    
    return tmr_n, entry_decision, exit_decision
end


transition
weight_x1 = aweights(transition[1,:])
weight_x2 = aweights(transition[2,:])
weight_x3 = aweights(transition[3,:])

x1_path = sample(MersenneTwister(123),[1,2,3], weight_x1, 10000; replace = true, ordered = false)
x2_path = sample(MersenneTwister(221),[1,2,3], weight_x2, 10000; replace = true, ordered = false)
x3_path = sample(MersenneTwister(521),[1,2,3], weight_x3, 10000; replace = true, ordered = false)

nprime_x1, entry_x1, exit_x1 = firm_num_simu(0, 1, mu_1, gamma_1)
nprime_x2, entry_x2, exit_x2 = firm_num_simu(0, 2, mu_1, gamma_1)
nprime_x3, entry_x3, exit_x3 = firm_num_simu(0, 3, mu_1, gamma_1)

iter = 1


nlist = Vector{Int64}(undef,1)
entry_decision = Vector{Int64}(undef,1)
exit_decision = Vector{Any}(undef,1)

while iter < 10000

    nprime_x11, entry_x11, exit_x11 = firm_num_simu(nprime_x1, x1_path[iter], mu_1, gamma_1)
    nprime_x22, entry_x22, exit_x22 = firm_num_simu(nprime_x1, x2_path[iter], mu_1, gamma_1)
    nprime_x33, entry_x33, exit_x33 = firm_num_simu(nprime_x1, x3_path[iter], mu_1, gamma_1)

    nprime_x1 = nprime_x11
    nprime_x2 = nprime_x22
    nprime_x3 = nprime_x33



    push!(nlist, nprime_x1, nprime_x2, nprime_x3)
    push!(entry_decision, entry_x11, entry_x22, entry_x33)
    push!(exit_decision, exit_x11, exit_x22, exit_x33)

    iter += 1
    @show (iter)
end

