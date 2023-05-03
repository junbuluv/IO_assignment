using Random, Statistics, Parameters, StatsBase, Distributions, Optim, ForwardDiff, Calculus, LinearAlgebra 


transition = [0.6 0.2 0.2 ; 0.2 0.6 0.2; 0.2 0.2 0.6]
temp = copy(transition)
β = 0.9
N_max = 5
s_max = 3
@with_kw mutable struct parameters
    γ::Int64 = 5
    σ_g::Int64 = 5
    μ::Int64 = 5
    σ_m::Int64 = 5
    

end




function profit(x, n)

    pi = ((10 .+ x )' ./ (n .+ 1)).^2 .- 5


    return pi
end


x = [-5,0,5]
n = [1,2,3,4,5]
profit(x, n)


### Transition matrix


gamma = [1.0 , 1.0 , 1.0 , 1.0 , 1.0]
mu = [1.0 , 1.0 , 1.0 , 1.0 , 1.0]


function tr(init_gamma, init_mu)
    gamma_1 = cdf(Normal(0,1), init_gamma[1])
    gamma_2 = cdf(Normal(0,1), init_gamma[2])
    gamma_3 = cdf(Normal(0,1), init_gamma[3])
    gamma_4 = cdf(Normal(0,1), init_gamma[4])
    mu_1 = cdf(Normal(0,1), init_mu[1])
    mu_2 = cdf(Normal(0,1), init_mu[2])
    mu_3 = cdf(Normal(0,1), init_mu[3])
    mu_4 = cdf(Normal(0,1), init_mu[4])
    mu_5 = cdf(Normal(0,1), init_mu[5])

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
    ((gamma_3) * mu_3^2 + (1- gamma_3) * binomial(2,1) * (1- mu_3) * mu_3) 
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
    
    1
    0
    0
    0
    0   ;

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
    mu_4^4
    ]

    return tr_d, tr_e
end



gamma
mu

Vbar = ones(5,3)


tr_d, tr_e = tr(gamma, mu)
tr_d = reshape(tr_d, (5,5))'
tr_e = reshape(tr_e, (5,5))'
psi_1 = 0.9 * ((Vbar * transition)' * tr_d)'
psi_2 = 0.9 * ((Vbar * transition)' * tr_e)'

new_mu = profit(x, n) + psi_1
new_gamma = psi_2

new_Vbar = ( 1 .- cdf(Normal(0,1), (profit(x,n) .+ psi_1 .- 5 )./sqrt(5))) .* (5 .+ sqrt(5) .* (pdf(Normal(0,1), (profit(x,n) .+ psi_1 .- 5)) ./sqrt(5)) ./ (1 .- cdf(Normal(0,1), (profit(x,n) .+ psi_1 .- 5)./sqrt(5)))) .+ cdf(Normal(0,1), (profit(x,n) .+ psi_1 .- 5) ./ sqrt(5) ) .* (profit(x,5) .+ psi_1)
