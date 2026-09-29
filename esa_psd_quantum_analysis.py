#!/usr/bin/env python3
"""
Plot all ESA (Electrical Spectrum Analyzer) PSD trace files in a folder.

Expected file format (comma-separated, '#'-prefixed header/comment lines):

    # f [Hz], PSD [dBVrms/sqrt(Hz)]
    #
    0.000000,-106.971200
    250.626566,-112.991800
    ...

Run directly from your IDE (edit the CONFIG block below, then hit Run).
"""

import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import regex as re 
from scipy import stats, signal
from scipy.fft import rfft, fft, fftfreq  # scipy.fft is preferred over np.fft for speed

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# ============================== CONFIG ==============================
FOLDER = r"../data/calibration data/ESA_8-21/LIGHT" # OTHER | BASELINE | LIGHT     # folder containing the ESA .txt trace files
PATTERN = "*.txt"             # glob pattern for input files
LOGX = False                   # log-scale frequency axis?
OUT = None                    # e.g. "overlay.png" to save; None to just display
# ======================================================================


def load_esa_file(filepath):
    """Load a two-column (freq [Hz], PSD [dBVrms/sqrt(Hz)]) file, skipping '#' comments."""
    data = np.genfromtxt(filepath, delimiter=",", comments="#")
    if data.size == 0:
        return np.array([]), np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    freq, psd = data[:, 0], data[:, 1]
    valid = ~np.isnan(freq) & ~np.isnan(psd)
    return freq[valid], psd[valid]

def load_exp_file(filepath):
    """Load a two-column (time [s], voltage [V]) file, skipping '#' comments."""
    data = np.genfromtxt(filepath, delimiter=",", skip_header=4)
    if data.size == 0:
        return np.array([]), np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t, A = data[:, 0], data[:, 1]
    valid = ~np.isnan(t) & ~np.isnan(A)
    return t[valid], A[valid]

def unpack_trace(fp):
    label = os.path.splitext(os.path.basename(fp))[0]
    freq, psd = load_esa_file(fp)

    # convert to linear PSD
    psd = np.power(10,np.divide(psd,10)) # V^2 / Hz linear basis

    return (label,freq,psd)

# models
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def noise_floor_model_log(f,a,b,c):
    # fit in log space
    return a*1/f+b*f**2+c





def main():
    # filepaths = sorted(glob.glob(os.path.join(FOLDER, PATTERN)), key=lambda f: float(m.group(1)) if (m := re.search(r'([\d.]+)\s*nw', f, re.I)) else float('inf'))

    rootp = r"/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/calibration data/ESA_8-28/"



    # named traces
    dark_trace = rootp+r"BASELINE/PSD_SRS760_TIA_PD_RB_0nw_0.txt"

    # define traces
    # dark_trace = rootp+r"PSD_SRS760_s1_dark_20avg_0.txt"
    dark_trace = rootp+r"PSD_SRS760_s1_dark_2000avg_0.txt"
    s1_noise_only_trace = rootp+r"PSD_SRS760_s1_noise_only_ph54nw_20avg_0.txt"
    s1_sig_only_trace = rootp+r"PSD_SRS760_s1_sig_only_ph54nw_20avg_0.txt"
    s1_both_trace = rootp+r"PSD_SRS760_s1_both_ph54nw_20avg_0.txt"





    traces = [s1_noise_only_trace,s1_sig_only_trace,s1_both_trace]


    # define color map(s)
    data_colors = plt.cm.viridis(np.linspace(0, 1, len(traces)+1))
    magenta_cmap = mcolors.LinearSegmentedColormap.from_list("magenta_gradient", ["#300030", "#800080", "#e0115f", "#ff00ff", "#ff66ff"])
    PSD_colors = magenta_cmap(np.linspace(0, 1, 3))

    # create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_prop_cycle(color=data_colors)

    # unpack traces + do math in V^2/Hz linear units, plot in dB

    # dark trace
    label,freq,psd_dark = unpack_trace(dark_trace)
    label = "PSD SR760, Dark Trace"
    ax.plot(freq, np.multiply(10,np.log10(psd_dark)), label=label, linewidth=1.2)

    # noise only
    label,freq,psd_noise_only = unpack_trace(s1_noise_only_trace)
    ax.plot(freq, np.multiply(10,np.log10(psd_noise_only)), label=label, linewidth=1.2)

    # signal only
    label,freq,psd_sig_only = unpack_trace(s1_sig_only_trace)
    ax.plot(freq, np.multiply(10,np.log10(psd_sig_only)), label=label, linewidth=1.2)

    # signal + noise
    label,freq,psd_both = unpack_trace(s1_both_trace)
    ax.plot(freq, np.multiply(10,np.log10(psd_both)), label=label, linewidth=1.2)

    # extract noise power @ homodyne
    Ph = int(re.search(r"ph(.*?)nw", label).group(1)) # nW

    # define matched filter
    # TODO




    # # Estimate SNR hierarchy

    # TODO add real, measured numbers
    e = 1.602176634e-19
    c = 3e8
    h=6.626e-34
    lambda_pc = 1530e-9
    S=1.23 # ideal responsivity of PD 
    Z_TIA=5e7 # TIA gain (V/A)
    Ps=100e-12 # Watts (measured @ SPDC src crystal)
    Pb=535e-6 # watts, pre-PCR
    hnu = 1.25e-19
    W=1.88e12 # optical BW (Hz)
    Rb=16e3 # Bit rate (Hz)
    NS=Ps/(hnu*W) # TODO derive
    NB=Pb/(hnu*W)
    kappa=2e-2
    T = 1/Rb # bit window duration
    M=W*T
    SNR_C = 2*kappa*NS*M/NB
    SNR_Q = 2*SNR_C

    # SNR: ceiling based on ideal pulse shape, ideal filtering, real noise
    # NOTE: pulse amplitude based on REAL photon-to-voltage conversion
    # generate ideal pulse spectrum
    Fs = 250e3 # sample rate (SPS)
    duration = 1/Rb # seconds
    photon_flux = M*np.sqrt(2*kappa*NS) # photon / sec
    optical_power = photon_flux * h*c/(lambda_pc) # watts
    s0 = optical_power*S*Z_TIA*Rb # Watts * (amp / Watt) * (V / A) = volts

    # s0_alt = (e*Z_TIA/T) * M*np.sqrt(2*kappa*NS)   # classical case
    # print(s0_alt)
    # return
    N_pulse = int(Fs * duration)  # Total number of samples 
    N_pad = 200*N_pulse

    s_t_padded = np.zeros(N_pad)
    s_t_padded[:N_pulse] = s0        # the pulse itself, unchanged -- zeros elsewhere are just resolution padding
    freq_dense = np.fft.rfftfreq(N_pad, d=1/Fs)

    # ideal pulse: frequency domain
    f_max = 50e3 
    s_f = (2/Fs)*rfft(s_t_padded) # FFT of ideal pulse shape, one-sided convention (factor of 2)
    mask = freq_dense <= f_max

    # limit to ESA max freq
    freq_dense = freq_dense[mask]
    s_f = s_f[mask]


    # interpolate sampled up to ideal
    f_cubic = interp1d(freq, psd_noise_only, kind='cubic')
    psd_noise_only_interp = f_cubic(freq_dense)


    # TODO debug
    # SNR_ceil = (2/Rb)*np.trapezoid(np.divide(np.power(np.abs(s_f),2),psd_noise_only_interp),freq_dense) # \int (S_both(f) - S_noise(f)) / S_noise(f) df (V^2)

    # calculate SNR with real pulse shape
    SNR_ceil_pulse = (2/Rb)*np.trapezoid(np.divide(psd_both-psd_noise_only,psd_noise_only),freq) # \int (S_both(f) - S_noise(f)) / S_noise(f) df (V^2)

    # calculate Deltas
    # Delta_A
    # TODO

    # SNR report
    print(f"SNRT_T,C: {SNR_C:.4f}")
    # print(f"SNRT_ceil: {SNR_ceil:.4f}") # TODO debug
    print(f"SNRT_ceil,pulse: {SNR_ceil_pulse:.4f}")
    print(f"SNRT_T,Q: {SNR_Q:.4f}")


    # print(f"NS: {NS:.4f}")
    # print(f"NB: {NB:.4f}")
    # print(f"freq min: {np.min(freq):.4f}")
    # print(f"freq max: {np.max(freq):.4f}")
    # print(f"df:{(freq[1]-freq[0]):.4f}")


    ## NOISE FLOOR FIT
    # TODO need to takr averaged + interpolated dark trace
    guess = [1,1,1]
    popt, pcov = curve_fit(noise_floor_model_log,freq,10*np.log10(psd_dark),p0=guess)

    optimized_a, optimized_b, optimized_c = popt
    print(f"Optimized Parameters:\na = {optimized_a:.4f}\nb = {optimized_b:.4f}\nc = {optimized_c:.4f}")
    perr = np.sqrt(np.diag(pcov))
    print(f"Parameter Errors: {perr}")

    guess_a = -0.075
    guess_b = 0.0000001
    guess_c = -86.5

    guess_fit = guess_a*freq+guess_c+guess_b*freq**2



    # Plot
    # create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_prop_cycle(color=data_colors)
    ax.plot(freq,10*np.log10(psd_dark),label="SR760 Dark Trace")
    # ax.plot(freq,guess_fit,label="fit")
    ax.set_ylim([-106,-99])
    ax.set_xlim([0,50e3])
    plt.show()
    print(frw)
    print(10*np.log10(psd_dark))

    return

    # mod frequency call out + downsample
    fmod = 8e3
    ax.axvline(fmod,linestyle='--',color='g',label='f_mod')
    mod_index = np.abs(freq-fmod).argmin()

    # # # THEORETICAL PSD # # # 
    # extract shot noise term from Ph
    Ph=Ph*1e-9 # convert to W basis
    Z_TIA=5e7 # TIA gain (V/A)
    e=1.6e-19 # J
    eta_plus = 0.934 # collection eta PD_+
    eta_minus = 0.96 # collection eta PD_-
    S=0.99/(eta_plus*eta_minus) # responsivity = quantum efficiency discounted by total collection efficiency
    S_shot = 2*e*S*Ph*(Z_TIA**2)# V^2/Hz
    S_total = S_shot # total PSD V^2/Hz
    S_shot_dbvrms = 10*np.log10(S_shot)# dBVrms/\sqrt(Hz)

    # # include effect of TIA input-referred current noise
    i_n=30e-15 # TIA current noise ASD [V/sqrt(Hz)] @ f_eval=10kHz, close enough to our f_mod
    S_i_TIA = (i_n*Z_TIA)**2 # V^2/Hz; current noise contribution to PSD by TIA 
    S_total += S_i_TIA
    S_total_dbvrms = 10*np.log10(S_total) # total PSD dbVrms/\sqrt(Hz)

    # plot theoretical PSD terms (matched units)
    c_idx = 0    
    ax.axhline(S_shot_dbvrms,linestyle='--',color=PSD_colors[c_idx],label=f"S_shot(f,Ph={Ph*1e9:.0f} nW)")
    c_idx+=1

    ax.axhline(S_total_dbvrms,linestyle='--',color=PSD_colors[c_idx],label=f"S_total((f,Ph={Ph*1e9:.0f} nW))")
    c_idx+=1



    # styling for power traces
    TITLE = "ESA Noise Floor; Experimental"
    ax.set_xlim([500,50e3]) # cut off flicker noise
    ax.set_ylim([-120,-85])

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [dBVrms/$\sqrt{Hz}$]")
    ax.set_title(TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    plt.show()
    

    #      ##      ##      ##      ##      ##      ##      ##      #
    #      ##      ##      ##      ##      ##      ##      ##      #
    #      ##      ##      ##      ##      ##      ##      ##      #
    #      ##      ##      ##      ##      ##      ##      ##      #





if __name__ == "__main__":
    main()


    