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


mean(count(i -> (i == 0), df_11.Exit)/ size(df_11,1) + count(i -> (i == 1), df_11.Exit)/ size(df_11,1))

count(i -> (i == 0), df_12.Exit)/ size(df_12,1)
count(i -> (i == 1), df_12.Exit)/ size(df_12,1)

count(i -> (i == 0), df_13.Exit)/ size(df_13,1)
count(i -> (i == 1), df_13.Exit)/ size(df_13,1)

count(i -> (i == 0), df_21.Exit)/ size(df_21,1)
count(i -> (i == 1), df_21.Exit)/ size(df_21,1)
count(i -> (i == 2), df_21.Exit)/ size(df_21,1)

count(i -> (i == 0), df_22.Exit)/ size(df_22,1)
count(i -> (i == 1), df_22.Exit)/ size(df_22,1)
count(i -> (i == 2), df_22.Exit)/ size(df_22,1)


count(i -> (i == 0), df_23.Exit)/ size(df_23,1)
count(i -> (i == 1), df_23.Exit)/ size(df_23,1)
count(i -> (i == 2), df_23.Exit)/ size(df_23,1)

count(i -> (i == 0), df_31.Exit)/ size(df_31,1)
count(i -> (i == 1), df_31.Exit)/ size(df_31,1)
count(i -> (i == 2), df_31.Exit)/ size(df_31,1)
count(i -> (i == 3), df_31.Exit)/ size(df_31,1)

count(i -> (i == 0), df_32.Exit)/ size(df_32,1)
count(i -> (i == 1), df_32.Exit)/ size(df_32,1)
count(i -> (i == 2), df_32.Exit)/ size(df_32,1)
count(i -> (i == 3), df_32.Exit)/ size(df_32,1)

count(i -> (i == 0), df_33.Exit)/ size(df_33,1)
count(i -> (i == 1), df_33.Exit)/ size(df_33,1)
count(i -> (i == 2), df_33.Exit)/ size(df_33,1)
count(i -> (i == 3), df_33.Exit)/ size(df_33,1)



count(i -> (i == 0), df_41.Exit)/ size(df_41,1)
count(i -> (i == 1), df_41.Exit)/ size(df_41,1)
count(i -> (i == 2), df_41.Exit)/ size(df_41,1)
count(i -> (i == 3), df_41.Exit)/ size(df_41,1)
count(i -> (i == 4), df_41.Exit)/ size(df_41,1)


count(i -> (i == 0), df_42.Exit)/ size(df_42,1)
count(i -> (i == 1), df_42.Exit)/ size(df_42,1)
count(i -> (i == 2), df_42.Exit)/ size(df_42,1)
count(i -> (i == 3), df_42.Exit)/ size(df_42,1)
count(i -> (i == 4), df_42.Exit)/ size(df_42,1)


count(i -> (i == 0), df_43.Exit)/ size(df_43,1)
count(i -> (i == 1), df_43.Exit)/ size(df_43,1)
count(i -> (i == 2), df_43.Exit)/ size(df_43,1)
count(i -> (i == 3), df_43.Exit)/ size(df_43,1)
count(i -> (i == 4), df_43.Exit)/ size(df_43,1)



count(i -> (i == 0), df_51.Exit)/ size(df_51,1)
count(i -> (i == 1), df_51.Exit)/ size(df_51,1)
count(i -> (i == 2), df_51.Exit)/ size(df_51,1)
count(i -> (i == 3), df_51.Exit)/ size(df_51,1)
count(i -> (i == 4), df_51.Exit)/ size(df_51,1)
count(i -> (i == 5), df_51.Exit)/ size(df_51,1)

count(i -> (i == 0), df_52.Exit)/ size(df_52,1)
count(i -> (i == 1), df_52.Exit)/ size(df_52,1)
count(i -> (i == 2), df_52.Exit)/ size(df_52,1)
count(i -> (i == 3), df_52.Exit)/ size(df_52,1)
count(i -> (i == 4), df_52.Exit)/ size(df_52,1)
count(i -> (i == 5), df_52.Exit)/ size(df_52,1)


count(i -> (i == 0), df_53.Exit)/ size(df_53,1) 
count(i -> (i == 1), df_53.Exit)/ size(df_53,1)
count(i -> (i == 2), df_53.Exit)/ size(df_53,1)
count(i -> (i == 3), df_53.Exit)/ size(df_53,1)
count(i -> (i == 4), df_53.Exit)/ size(df_53,1)
count(i -> (i == 5), df_53.Exit)/ size(df_53,1)






def forward_simulate_incumbent(beta, cournot_pi_matrix, entry_TH, exit_TH, theta_distribution, s, N_init, x_idx_init):
# forward simulate incumbent's life cycle conditional on today's remaining decision
# function name: forward_simulate_incumbent
# function input: N_init, x_idx_init
# function output: PDV
# N_init: 1,2,3,4,5
# x_dix_init: 0,1,2
    gamma_loc, gamma_scale, mu_loc, mu_scale = theta_distribution
    t=0
    time_idx = [0]
    # Start with N_init
    N_history = [N_init]
    # Start with x_idx_init
    x_idx_history = [x_idx_init]
    # Remaining decision has not been made
    i_remain_history = []

    np.random.seed(s)
    # t=0 simulate
    
    #(1) x is updated exogenously 
    x_idx_pre = x_simulator(x_idx_init, s)
    
    # N_pre will be the initial value for the while loop for (t>=1)
    if N_init == 1:
        # When the incumbent is the single firm in the market
        # Potential entrant is the only factor that affects N'
        ## My decision is fixed as remain
        i_remain_decision = 1
        gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
        entry_TH_value = entry_TH[N_init][x_idx_init]
        ## Entrants decision is made
        entry_decision = int(gamma_draw < entry_TH_value)
        ## N is updated
        N_pre = N_init + entry_decision
    
    elif N_init == 5:
        # When there are five firms in the market, the potential entrant can not enter
        ## My decision is fixed as remain
        i_remain_decision = 1
        ## Other incumbents decisions
        mu_draw_array = np.random.normal(mu_loc, mu_scale, N_init-1)
        exit_TH_value_array = np.full(N_init-1, exit_TH[N_init-1][x_idx_init])
        ## N is updated
        N_pre = N_init - (mu_draw_array > exit_TH_value_array).sum()
        
    else:
        # When there are 2,3,4 firms in the market
        ## My decision is fixed as remain
        i_remain_decision = 1
        ## Other incumbents decisions
        mu_draw_array = np.random.normal(mu_loc, mu_scale, N_init-1)
        exit_TH_value_array = np.full(N_init-1, exit_TH[N_init-1][x_idx_init])
        gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
        entry_TH_value = entry_TH[N_init][x_idx_init]
        ## Entrants decision is made
        entry_decision = int(gamma_draw < entry_TH_value)
        ## N is updated
        N_pre = N_init - (mu_draw_array > exit_TH_value_array).sum() + entry_decision
        

    N_history.append(N_pre)                     # at this point len(N_history) == 2 
    x_idx_history.append(x_idx_pre)             # at this point len(x_idx_history) == 2 
    i_remain_history.append(i_remain_decision)  # at this point len(i_remain_history) == 1


    while(t < 1000):
        t += 1
        time_idx.append(t)
        N_current = N_history[-1]
        x_idx_current = x_idx_history[-1]
        
        np.random.seed(s*1000+t)
        #(1) decide i's exit decision
        i_mu_draw = np.random.normal(mu_loc, mu_scale, 1)[0]
        exit_TH_value = exit_TH[N_current-1][x_idx_current]
        i_remain_decision = int(i_mu_draw < exit_TH_value)
        
        # break condition conditional on i's exit decision
        if i_remain_decision == 0: # i exit
            i_remain_history.append(i_remain_decision)
            scrap_value = i_mu_draw
            break

        #(2) continue forward simulating
        #note that i decided to remain in the market at this point
        #(2-1) update the state variables
        x_idx_tmr = x_simulator(x_idx_current, s*1000+t)

        if N_current == 1:
            # i is the single incumbent in this market
            # so only the entrant will change the number of firms in the market
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current + entry_decision
            
        
        elif N_current == 5:
            # since there are 5 incumbents in the market, entrant can not enter the market tmr
            # hence, only the incumbents can change the number of firms in the market
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1)
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum()
            

        else:    
            mu_draw_array = np.random.normal(mu_loc, mu_scale, N_current-1)
            exit_TH_value_array = np.full(N_current-1, exit_TH[N_current-1][x_idx_current])
            gamma_draw = np.random.normal(gamma_loc, gamma_scale, 1)[0]
            entry_TH_value = entry_TH[N_current][x_idx_current]
            entry_decision = int(gamma_draw < entry_TH_value)
            N_tmr = N_current - (mu_draw_array > exit_TH_value_array).sum() + entry_decision
            
        #(2-2) record the decision and transition
        i_remain_history.append(i_remain_decision)
        N_history.append(N_tmr)
        x_idx_history.append(x_idx_tmr)

    df_incumbent_simulation = df({"t": time_idx, "N": N_history, "x_idx": x_idx_history, "remain_decision": i_remain_history})
    def incumbent_PDV(t, N, x_idx, remain_decision, scrap_value):
        if remain_decision == 0:
            pdv = (beta**t)*scrap_value
        else:
            pdv = (beta**t)*cournot_pi_matrix[N-1][x_idx]
        return pdv
    vec_incumbent_PDV = np.vectorize(incumbent_PDV)
    PDV = vec_incumbent_PDV(df_incumbent_simulation['t'].values, df_incumbent_simulation['N'].values, df_incumbent_simulation['x_idx'].values, df_incumbent_simulation['remain_decision'].values, scrap_value).sum()

    return PDV