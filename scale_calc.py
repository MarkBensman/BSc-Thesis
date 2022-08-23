import os,sys,inspect
from brian2 import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from make_dynamic_experiments import make_dynamic_experiments
from helpers import make_spiketrain
from MI_calculation import analyze_exp
from Functions import input_simulation, adex_model, buya_spiketrain_calc
import datetime

'''
1. Create hidden state and input with 'probe' values(uq = 1.3, tau = 20)
2. Run adex model with varying scale to find firing rate = (2.6 and 0.27) Hz
3. Choose range start and end, choose frequency values to look for and run simulation with values in range.
4. Plot

'''
#%% 1. Create hidden state and input with 'probe' values(uq = 1.3, tau = 20)

# Parameters:
qon_qoff_type = 'balanced'
baseline = 0
tau = 50
factor_ron_roff = 2
mean_firing_rate = 0.5/1000  # uq
sampling_rate = 5
duration = 120000

input_theory, hidden_state, run_time = input_simulation(qon_qoff_type, baseline, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)

#%% 2. Run adex model with varying scale to find firing rate = (2.6 and 0.27) Hz


def find_scale(scale_rng_strt, scale_rng_end, freq_inh, freq_exc, input_theory, duration, sampling_rate):
    scale_rng_jmp = int((scale_rng_end - scale_rng_strt) *1) #jump size
    I_scale_range = np.linspace(scale_rng_strt, scale_rng_end, scale_rng_jmp, endpoint=0) #range of multipliers
    cols = ['I_scale', 'freq_spike_m']
    lst = []
    for I_scale in I_scale_range:
        state_m, spike_m = adex_model(input_theory, duration, sampling_rate, I_scale)
        freq_spike_m = spike_m.num_spikes / (duration / 1000) #freq calc. spikes / second
        lst.append([I_scale, freq_spike_m])
        print(I_scale, 'Out of ', scale_rng_end)
        print(freq_spike_m)
    df_scale_freq = pd.DataFrame(lst, columns=cols) #combine lst to df
    pd.DataFrame(df_scale_freq).to_csv('Scaling/'+'I_scaling.'+str(scale_rng_strt)+'_to_'+str(scale_rng_end)+'.duration_'
                                       +str(duration)+datetime.datetime.now().strftime('.%d-%m-%Y-%H-%M')+'.csv') #save to csv file
    
    return I_scale_range, state_m, spike_m, freq_spike_m, df_scale_freq


#%% 3. Choose range start and end, choose frequency values to look for and run simulation with values in range.
scale_rng_strt = 75
scale_rng_end = 150
freq_inh = 2.6
freq_exc = 0.27
I_scale_range, state_m, spike_m, freq_spike_m, df_scale_freq = find_scale(scale_rng_strt, scale_rng_end, freq_inh, freq_exc, input_theory, duration, sampling_rate)


scal_inh = np.interp(freq_inh, df_scale_freq['freq_spike_m'], df_scale_freq['I_scale'],  left=None, right=None, period=None)
scal_exc = np.interp(freq_exc, df_scale_freq['freq_spike_m'], df_scale_freq['I_scale'],  left=None, right=None, period=None)


#%% 4. Plot

zerovalue = [0, 0]
fig = plt.figure(figsize=(15,20))
plt.margins(x=0, y=0)
plt.title('Input Scaling\n', fontsize = 35)
plt.plot(df_scale_freq['I_scale'], df_scale_freq['freq_spike_m'], 'o', label = 'Data points')
plt.yticks(np.arange(min(zerovalue), max(df_scale_freq['freq_spike_m']), 0.2), fontsize = 30)
plt.xticks(np.arange(min(df_scale_freq['I_scale']), max(df_scale_freq['I_scale']), 10), fontsize = 30)
plt.ylabel('Firing rate (Hz)', fontsize = 30)
plt.xlabel('\nScaling multiplier (Unitless)', fontsize = 30)
plt.hlines(y = freq_inh, xmin = df_scale_freq['I_scale'].min(), xmax = scal_inh, color = 'r', linestyle = 'dashed', label = 'Inhibitory')
plt.hlines(y = freq_exc, xmin = df_scale_freq['I_scale'].min(), xmax = scal_exc, color = 'b', linestyle = 'dashed', label = 'Excitatory')
plt.vlines(x = scal_inh, ymin = df_scale_freq['freq_spike_m'].min(), ymax = freq_inh, color = 'r', linestyle = 'dashed')
plt.vlines(x = scal_exc, ymin = df_scale_freq['freq_spike_m'].min(), ymax = freq_exc, color = 'b', linestyle = 'dashed')
plt.legend(loc='upper left', fontsize = 30 )
plt.savefig('scaling_multiplier'+'.png',dpi=300, transparent=True,bbox_inches='tight')
plt.show()
    

    


