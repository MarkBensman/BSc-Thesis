import os,sys,inspect
from brian2 import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from make_dynamic_experiments import make_dynamic_experiments
from helpers import make_spiketrain
from MI_calculation import analyze_exp


'''
1. Create hidden state and ANN
2. Adaptive exponential integrate-and-fire model
3. Calculation of MI, Fi
4. Plots

'''

#%% Simulating Input

def input_simulation (qon_qoff_type, baseline, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration):
    
    [input_theory, _, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)
    dt = 1./sampling_rate
    run_time = (dt)*np.arange(len(input_theory))
    
    return input_theory, hidden_state, run_time



    
#%% Adaptive exponential integrate-and-fire model
    
def adex_model(input_theory, duration, sampling_rate, I_scale):
    start_scope()
    
    '''
      Parameters
            tau_m (Quantity): membrane time scale
            R (Quantity): membrane restistance
            v_rest (Quantity): resting potential
            v_reset (Quantity): reset potential
            v_rheobase (Quantity): rheobase threshold
            a (Quantity): Adaptation-Voltage coupling
            b (Quantity): Spike-triggered adaptation current (=increment of w after each spike)
            v_spike (Quantity): voltage threshold for the spike condition
            delta_T (Quantity): Sharpness of the exponential term
            tau_w (Quantity): Adaptation time constant
            I_stim (TimedArray): Input current
            simulation_time (Quantity): Duration for which the model is simulated
            
            Returns:
            (state_monitor, spike_monitor):
            StateMonitor for the variables 'v' and 'w' and SpikeMonitor
    '''
    
    # Scaling the input and creating a TimedArray
    dt = 1./sampling_rate * ms
    I_baseline = 0
    defaultclock.dt = dt
    Input_scaled = TimedArray(((I_scale*input_theory+I_baseline)*pamp), dt = dt)
    
    
    # Defining model parametrs
    membrane_time_scale_tau_m = 5*ms
    membrane_resistance_R = 500 * Mohm
    V_rest = -70.0 * mV
    V_reset = -51.0 * mV
    rheobase_threshold_v_rh = -50.0 * mV
    sharpness_delta_T = 2.0 * mV
    adaptation_voltage_coupling_a = 0.5 * nS
    adaptation_time_constant_tau_w = 100.0 * ms
    spike_triggered_adaptation_increment_b = 7.0 * pA
    
    #when to reset vm to v_reset
    firing_threshold_v_spike = -30. * mV
    
    
    def model_simulation(
            stupd_var = 2, #for some reason TimedArray does not work without a random first variable
            tau_m = membrane_time_scale_tau_m,
            R=membrane_resistance_R,
            v_rest=V_rest,
            v_reset=V_reset,
            v_rheobase=rheobase_threshold_v_rh,
            a=adaptation_voltage_coupling_a,
            b=spike_triggered_adaptation_increment_b,
            v_spike=firing_threshold_v_spike,
            delta_T=sharpness_delta_T,
            tau_w=adaptation_time_constant_tau_w,
            I_stim = Input_scaled,
            simulation_time=duration * ms):
    
        v_spike_str = 'v>{:f}*mvolt'.format(v_spike / mvolt)
    
        eqs = """
            dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t) - R * w)/(tau_m) : volt
            dw/dt=(a*(v-v_rest)-w)/tau_w : amp
            """
        G = NeuronGroup(1, eqs, threshold=v_spike_str, reset= 'v=v_reset;w+=b', method='euler')
    
        # initial values of v and w:
        G.v = v_rest
        G.w = 0.0 * pA
        
        # Monitoring membrane voltage (v) and w
        state_monitor = StateMonitor(G, ['v', 'w'], record=0)
        spike_monitor = SpikeMonitor(G)
        
    
        # running simulation
        run(simulation_time)
        return state_monitor, spike_monitor
    
    
    [state_monitor, spike_monitor] = model_simulation(I_stim = Input_scaled)
    state_m = state_monitor
    spike_m = spike_monitor
    print('Simulation complete')
    
    return state_m, spike_m

#%% (Unused) Methods Comparison for Spiketrain
    
    # def spiketrain_method1 (spike_m, duration, dt):
    #         spiketrain = make_spiketrain(spike_m, duration, dt)
    #         return spiketrain
    
    
    
    # def spiketrain_method2 (hidden_state, spike_m, dt):
        
    #     spiketrain = zeros (len(hidden_state))
    #     spikenumber = spike_m.num_spikes
    #     spiketimes = spike_m.spike_trains()
    #     spiketimes = spiketimes[0]
    #     spike_indices = spiketimes/dt
    #     spike_indices = spike_indices.astype(int)
    #     for n in range(0, spikenumber):
    #         spiketrain[spike_indices[n]] = 1
    #     spiketrain = np.array([spiketrain])
        
    #     return spiketrain
    
    # print(spiketrain)
    # spiketrain testin
    # np.any(spiketrain[:,:] == 1)
    # np.shape(spiketrain)
    # print('^^^^^^^^^^^^^^^^^^^^^')
#%% Calculation of MI, Fi
  
def buya_spiketrain_calc(sampling_rate, tau, factor_ron_roff, spike_m, duration, hidden_state, input_theory):
    dt = 1./sampling_rate
    ron = 1./(tau*(1+factor_ron_roff))
    roff = factor_ron_roff*ron
    
    spiketrain = make_spiketrain(spike_m, duration, dt)
    buya = analyze_exp(ron, roff, hidden_state, input_theory, dt, 0, spiketrain)
  
    
    '''
    OUTPUT (from analyze_exp)
        Output-dictionary with keys:
        MI_i        : mutual information between hidden state and input current
        xhat_i      : array with hidden state estimate based on input current 
        MSE_i       : mean-squared error between hidden state and hidden state estimate based on input current 
        MI          : mutual information between hidden state and spike train
        qon, qoff   : spike frequency during on, off state in spike train 
        xhatspikes  : array with hidden state estimate based on spike train
        MSE          : mean-squared error between hidden state and hidden state estimate based on spike train
    '''
    

    # spike_prediction
    [hidden_estimate_out] = buya.xhatspikes.to_numpy()
    [hidden_estimate_inp] = buya.xhat_i.to_numpy()
    buya = buya.drop(["xhat_i", "xhatspikes"], axis = 1)
    return hidden_estimate_out, hidden_estimate_inp ,buya
    




#%% Plots  

def all_plots(run_time, hidden_state,input_theory, state_m, spike_m, buya, hidden_estimate_out, hidden_estimate_inp, plot_maybe = False):
    if plot_maybe:  
        #%%% plot hidden state and input theory
            
        fig, ax1 = plt.subplots(figsize = (25,5))
        color1 = 'black'
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Hidden', color=color1)
        ax1.plot(run_time, hidden_state, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx() 
        
        color2 = 'tab:blue'
        ax2.set_ylabel('Theoretical current input', color=color2)
        ax2.plot(run_time, input_theory, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        
        fig.tight_layout()
        plt.show()
        
        #%%% Plot input_theory and output spike train
        
        fig, ax1 = plt.subplots(figsize = (25,5))
        color1 = 'tab:blue'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Input', color=color1)
        ax1.plot(run_time, input_theory, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color2 = 'tab:red'
        ax2.set_ylabel('Output', color=color2)  # we already handled the x-label with ax1
        ax2.plot(state_m.t/ms,  state_m.v[0], color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        #%%% Plot Hidden state and output 
        
            
        fig, ax1 = plt.subplots(figsize = (25,5))
        color1 = 'black'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Input', color=color1)
        ax1.plot(run_time, hidden_state, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color2 = 'tab:red'
        ax2.set_ylabel('Output', color=color2)  # we already handled the x-label with ax1
        ax2.plot(state_m.t/ms,  state_m.v[0], color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        #%%% Plot spike prediction on hidden state
        
        fig, ax1 = plt.subplots(figsize = (25,5))
        color1 = 'tab:purple'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Hidden State', color=color1)
        ax1.plot(run_time, hidden_state, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx() 
        
        color2 = 'black'
        ax2.set_ylabel('Hidden State reconstruction', color=color2)
        ax2.plot(run_time,  hidden_estimate_out, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        
        fig.tight_layout()
        plt.show()
        
        
        
        #%%%
        
        
        
        fig, ax1 = plt.subplots(figsize = (25,5))
        color1 = 'tab:purple'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Input Theory', color=color1)
        ax1.plot(run_time, hidden_state, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx() 
        
        color2 = 'black'
        ax2.set_ylabel('hidden_estimate_inp', color=color2)
        ax2.plot(run_time, hidden_estimate_inp, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        
        fig.tight_layout()
        plt.show()
        
        # return
        
# all_plots(run_time, hidden_state,input_theory, state_m, spike_m, buya, hidden_estimate_out, plot_maybe = 1)
