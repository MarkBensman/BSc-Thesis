import os,sys,inspect
from brian2 import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from make_dynamic_experiments import make_dynamic_experiments
from helpers import make_spiketrain
from MI_calculation import analyze_exp
from Functions import input_simulation, adex_model, buya_spiketrain_calc, all_plots
import datetime
import seaborn as sns





#%% Input Generation

'''
Create hidden state and input with varying tau and uq.
'''
# Parameters:
qon_qoff_type = 'balanced'
baseline = 0
tau = 75    # how fast the hidden state changes
factor_ron_roff = 2
mean_firing_rate =0.0045    # uq - mean firing rate of the ANN
sampling_rate = 5
duration = 120000
I_scale = 84.68 #Scaling multiplier (current is probe2- exc)

tau_range = np.linspace(2, 82, 40, endpoint=0) 
mean_firing_rate_range = np.around(np.linspace(0.0002, 0.005, 25, endpoint=1), 4)
vary_cols = [
    'tau', 'uq','num_spikes', 'Fi', 'freq_spike_m', 'freq_spike_m_norm', 'Ef', 'Ef_norm', 'MI_i', 'MSE_i', 'MI', 'qon', 'qoff', 'MSE'
    ]
vary_lst = []
buya_res_lst = []
results_cols = [
    'hidden_state', 'input_theory', 'hidden_estimate_inp', 'hidden_estimate_out', 'state_m_v', 'state_m_w', 'spike_m_t'
    ]



sim_num = 0
for tau in tau_range:
    for mean_firing_rate in mean_firing_rate_range:
        
        
        input_theory, hidden_state, run_time = input_simulation(qon_qoff_type, baseline, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)
        state_m, spike_m = adex_model(input_theory, duration, sampling_rate, I_scale)
        freq_spike_m = spike_m.num_spikes / (duration / 1000)
        print(freq_spike_m)
        hidden_estimate_out, hidden_estimate_inp ,buya = buya_spiketrain_calc(sampling_rate, tau, factor_ron_roff, spike_m, duration, hidden_state, input_theory)
        Fi = buya.at[0,"MI"] / buya.at[0,"MI_i"] #MI_spike / MI_input
        print(Fi)
        Ef = Fi / freq_spike_m
        print(Ef)
        freq_spike_m_norm = freq_spike_m * tau
        Ef_norm = Fi / freq_spike_m_norm
        print(Ef_norm)
        print(tau, mean_firing_rate)
        
        #storing
        #all vectors to one file, buya vectors added after the loop
        vary_lst.append([
            tau, mean_firing_rate, spike_m.num_spikes, Fi, freq_spike_m, freq_spike_m_norm, Ef, Ef_norm, buya.at[0,'MI_i'], buya.at[0,'MSE_i'], buya.at[0,'MI'], buya.at[0,'qon'], buya.at[0,'qoff'], buya.at[0,'MSE']]
            )
        #seperate files for each experiment
        results_lst = [] #empty arrray
        #append results to array (columnwise)
        results_lst.append([hidden_state, input_theory, hidden_estimate_inp, hidden_estimate_out, state_m.w, state_m.v, spike_m.t])
        #save to df
        results = pd.DataFrame(results_lst, columns=results_cols)
        #export to scv - !folder naming is manual!
        pd.DataFrame(results).to_csv('Sim_Result_exc/'+'Sim'+str(sim_num)+'.'+'tau_'+str(tau)+'.uq_'+str(mean_firing_rate)+'.results.csv')  
        sim_num+=1
        
        
    #all scalers to one file    
result_sca = pd.DataFrame(vary_lst, columns=vary_cols)

pd.DataFrame(result_sca).to_csv('Sim_Result_exc/'+datetime.datetime.now().strftime('%d-%m-%Y-%H-%M')+'.result_sca.csv') #save to csv file with time stamp
