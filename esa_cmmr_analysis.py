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
import regex as re 
from scipy import stats, signal
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# from scipy.integrate import integrate


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

def unpack_trace(fp,convert=True):
    label = os.path.splitext(os.path.basename(fp))[0]
    freq, psd = load_esa_file(fp)

    # convert to linear PSD
    if(convert):
        psd = np.power(10,np.divide(psd,10)) # V^2 / Hz linear basis

    return (label,freq,psd)

# models
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c



def main():
    # filepaths = sorted(glob.glob(os.path.join(FOLDER, PATTERN)), key=lambda f: float(m.group(1)) if (m := re.search(r'([\d.]+)\s*nw', f, re.I)) else float('inf'))

    rootp = r"/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/calibration data/ESA_8-27/"



    # named traces
    unbalanced_pd1 = rootp+r"PSD_SRS760_CMMR_mod_unbal_vrms_pd1_0.txt"
    unbalanced_pd2 = rootp+r"PSD_SRS760_CMMR_mod_unbal_vrms_pd2_0.txt"
    balanced = rootp+r"PSD_SRS760_CMMR_mod_bal_vrms_0.txt"
    
    traces = [unbalanced_pd1,unbalanced_pd2,balanced]


    f_mod=50e3 # for intensity mod CMMR characterization

    colors = plt.cm.viridis(np.linspace(0, 1, len(traces)))

    # create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    # ax.set_prop_cycle(color=colors)

    # unpack traces + do math in V^2/Hz linear units, plot in dB

    sampled = {}
    for trace in traces:
        label,freq,spectrum = unpack_trace(trace,convert=False)
        spectrum_dbv = 20*np.log10(np.sqrt(2)*spectrum)
        ax.plot(freq, spectrum_dbv, label=label, linewidth=1.2)

        pk_idx = np.argmin(abs(freq-f_mod))+1

        # sample
        sampled[label] = spectrum[pk_idx]

    ax.axvline(freq[pk_idx],label="Modulation Frequency",linestyle="--",color="r")


    avg_unbal = 1/2*(sampled["PSD_SRS760_CMMR_mod_unbal_vrms_pd1_0"] + sampled["PSD_SRS760_CMMR_mod_unbal_vrms_pd2_0"])
    bal = sampled["PSD_SRS760_CMMR_mod_bal_vrms_0"]

    CMMR = 20*np.log10(avg_unbal/(2*bal))

    x_text = 52e3
    y_text = -40
    ax.annotate(
        text=f"CMMR = {CMMR:.2f} dB", 
        xy=(x_text, y_text),          
        xytext=(x_text - 1, y_text + 2),      
    )
    ax.set_xlim([f_mod-15e3,f_mod+15e3]) # cutoff DC
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("dBV")
    ax.set_title("Intensity Modulation CMMR Characterization")
    ax.legend()
    plt.show()
    return

    # # ESA noise floor
    # label,freq,psd = unpack_trace(esa_term_trace)
    # # ax.plot(freq, np.multiply(10,np.log10(psd)), label=label, linewidth=1.2)

    # # dark trace
    label,freq,psd_dark = unpack_trace(dark_trace)
    # ax.plot(freq, np.multiply(10,np.log10(psd_dark)), label=label, linewidth=1.2)

    # # optical power sweep trace(s)
    # label,freq,psd_42 = unpack_trace(p42_trace)
    # ax.plot(freq, np.multiply(10,np.log10(psd_42)), label=label, linewidth=1.2)

    # label,freq,psd_85 = unpack_trace(p85_trace)
    # ax.plot(freq, np.multiply(10,np.log10(psd_85)), label=label, linewidth=1.2)

    # label,freq,psd_190 = unpack_trace(p190_trace)
    # ax.plot(freq, np.multiply(10,np.log10(psd_190)), label=label, linewidth=1.2)

    # label,freq,psd_280 = unpack_trace(p280_trace)
    # ax.plot(freq, np.multiply(10,np.log10(psd_280)), label=label, linewidth=1.2)

    # experimental data, convert to PSD units FFT
    t,A = load_exp_file(p_exp_trace)
    dt = np.median(np.diff(t))
    fs=1.0/dt 
    # scaling='density' -> V^2/Hz ; return_onesided=True -> one-sided convention (0 to fs/2)
    N_bins = 400 # num bins on ESA
    span_hz = 50e3
    nperseg = round(fs * N_bins / span_hz)   # span_hz = 100e3 for full SR760 span
    f_welch, Pxx_welch = signal.welch(
        A, fs=fs,
        window='hann',
        nperseg=nperseg,        # tune: freq resolution = fs/nperseg
        noverlap=np.floor(nperseg/2),        # 50% overlap is standard
        detrend='constant',   #  'linear' if you suspect drift, 'constant' otherwise'
        scaling='density',
        return_onesided=True
    )

    # exp params
    pb=4407
    G_PCR=1+50.03e-5
    kappa_PCR=0.195
    PH_exp = pb*1e-6*(G_PCR-1)*kappa_PCR*1e9# power at homodyne BS, experimental dataset
    powers.append(PH_exp)   
    label_welch = "Experimental Data ~"+str(round(PH_exp))+" nw"

    # adjust for LNA voltage gain
    G_LNA = 5e2 # voltage gain (V/V); from 4-30 batch_5. (see 5-1-26 report)
    Pxx_welch = np.divide(Pxx_welch, np.power(G_LNA,2))

    # adjust for BPF in LNA
    # build out LNA BPF transfer function; single pole HP + single pole LP (6 dB / dec -> N=1, simple RC)
    # checked frequencies in exp. photos from 4-30-2026
    f_hp = 3e3 # Hz
    f_lp = 10e3 # Hz
    # general form: (f_welch/f_0)^2/(1+(f_welch/f_0)^2)
    H_HP = np.divide(np.power(np.divide(f_welch,f_hp),2),(1+np.power(np.divide(f_welch,f_hp),2)))
    H_LP = np.divide(1,(1+np.power(np.divide(f_welch,f_lp),2)))
    # general form: (f_welch/f_lp)^2/(1+(f_welch/f_lp)^2)
    H_LNA_BPF = np.multiply(H_HP,H_LP) # total transfer function

    # undo filter (LNA)
    Pxx_welch = np.divide(Pxx_welch,H_LNA_BPF)

    # simulate moving avg
    M=100 # num avgs on ESA trace
    Pxx_smooth = np.convolve(Pxx_welch, np.ones(M)/M, mode='same')
    valid_Pxx_smooth = np.isfinite(Pxx_smooth)

    ax.plot(f_welch, np.multiply(10,np.log10(Pxx_welch)), label=label_welch, linewidth=1.2)
    ax.plot(f_welch[valid_Pxx_smooth], np.multiply(10,np.log10(Pxx_smooth))[valid_Pxx_smooth], label=label_welch+str("_smooth"), linewidth=1.2,color='r')

    # styling for power traces
    # ax.set_ylim([-92, -120])
    # ax.invert_yaxis()
    # TITLE = "ESA PSD Overlay"     # plot title
    TITLE = "ESA Noise Floor; Experimental Overlay; w/ Filter Detrend"
    ax.set_xlim([500,50e3]) # cut off flicker noise
    # ax.set_ylim([-110,-80]) # no filter detrend
    ax.set_ylim([-97,-85]) # w/ filter detrend

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [dBVrms/$\sqrt{Hz}$]")
    ax.set_title(TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    plt.colorbar()

    plt.show()
    return

    # validation that Pxx normalization is correct
    # --- 3. Manual single-segment periodogram, for cross-validation of normalization ---
    window = signal.windows.hann(len(A))
    U = np.sum(window**2)          # window power, needed for correct normalization
    A_detrended = A - np.mean(A)
    Aw = A_detrended * window

    X = np.fft.rfft(Aw)
    freqs = np.fft.rfftfreq(len(A), d=dt)

    S_two = (np.abs(X)**2) / (fs * U)     # two-sided PSD, V^2/Hz
    S_one = S_two.copy()
    S_one[1:-1] *= 2                       # fold negative freqs onto positive
    if len(A) % 2 == 0:
        S_one[-1] /= 1  # Nyquist bin not doubled for even N (already unique); adjust if needed

    # --- 4. Parseval check: does the PSD integrate back to the time-domain variance? ---
    var_time = np.var(A_detrended)
    var_freq = np.trapezoid(S_one, freqs)
    print(f"Time-domain variance: {var_time:.6e} V^2")
    print(f"Integrated one-sided PSD: {var_freq:.6e} V^2")
    print(f"Ratio (should be ~1.00): {var_freq/var_time:.4f}")




    # plot 


    # separate plot for delta
    mod=8e3
    detune=10e3
    mod_target=mod+detune # for sampling later

    fig, ax = plt.subplots(figsize=(9, 6))  
    ax.set_prop_cycle(color=colors)

    # delta_42 = psd_42 - psd_dark
    # delta_85 = psd_85 - psd_dark
    # delta_190 = psd_190 - psd_dark
    # delta_280 = psd_280 - psd_dark

    deltas = []
    for trace in traces:
        label,freq,psd = unpack_trace(trace)
        label+=" Delta"
        delta = psd-psd_dark
        deltas.append(delta)
        ax.plot(freq, delta, label=label, linewidth=1.2)

    # interp to welch freq arr density
    cs = CubicSpline(freq, psd_dark)
    psd_dark_interp = cs(f_welch)

    # store experimental delta
    delta_exp = Pxx_welch - psd_dark_interp # smooth for ds, single-shot for plot  
    # delta_exp = Pxx_smooth - psd_dark_interp # smooth for ds, single-shot for plot
    deltas.append(delta_exp)
    ax.plot(f_welch, delta_exp, label="Experimental (Delta)", linewidth=1.2)




    # # can't plot, 6 OOM higher than fit... why?
    # delta_1000_exp = Pxx_welch - psd_dark_interp

    # ax.plot([],[]) # for cmap consistency
    # ax.plot(freq, delta_42, label="delta_42", linewidth=1.2)
    # ax.plot(freq, delta_85, label="delta_85", linewidth=1.2)
    # ax.plot(freq, delta_190, label="delta_190", linewidth=1.2)
    # ax.plot(freq, delta_280, label="delta_280", linewidth=1.2)
    ax.axvline(mod_target,linestyle='--',label="Sampling Frequency",color='r')
    # ax.plot(f_welch,delta_1000_exp,label="delta_1000_exp",linewidth=1.2)


    # styling for delta
    TITLE = "PSD Relative to Dark Trace"
    ax.set_ylim([0 , 3.5e-9])
    ax.set_xlim([500,50e3]) # cut off flicker noise
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"\Delta PSD S\(f,I\)-S\(f,\) (V^2/Hz)")
    ax.set_title(TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    # plt.show()
    # return


    # sample deltas at mod frequency
    mod_index = np.abs(freq-mod_target).argmin()
    # powers = [42,85,190,280,1000]
    # sampled_PSD = [delta_42[mod_index], delta_85[mod_index],delta_190[mod_index],delta_280[mod_index],delta_1000_exp[mod_index]] # linear
    sampled_PSD = [x[mod_index] for x in deltas]

    # linear fit
    # only use calibration powers
    reg_calib = stats.linregress(powers, sampled_PSD)

    # quadratic fit
    # use calibration AND experimental power
    popt, pcov = curve_fit(quadratic_model, powers ,sampled_PSD)
    p_smooth = np.linspace(0,1e3,100)





    # model derivation
    # TODO

    # separate plot for delta scaling
    fig, ax = plt.subplots(figsize=(9, 6))  
    ax.scatter(powers[0:-1],sampled_PSD[0:-1],label="Calibration")
    ax.scatter(powers[-1],sampled_PSD[-1],label="Experimental")
    ax.plot(powers, np.multiply(reg_calib.slope,powers) + reg_calib.intercept,linestyle='--',color='r',label='Linear Fit (Calibration only); R^2='+str(round(reg_calib.rvalue**2,3)))

    # plot quad fit
    # ax.plot(p_smooth, quadratic_model(p_smooth, *popt), color='b', label='Quadratic Fit')

    # styling for delta sampled
    red_blue_label = "red" if detune < 0 else "blue"
    TITLE = "Sampled Delta PSD at f_mod ("+str(round(detune/1e3))+"kHz "+red_blue_label+" detune)"
    # ax.set_xlim([500,50e3]) # cut off flicker noise
    # ax.set_ylim([0,1e-8])
    ax.set_xlabel("Optical Power [nW]")
    ax.set_ylabel(r"\Delta PSD S(f,I)-S(f,0) (V^2/Hz)")
    ax.set_title(TITLE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()






    if OUT:
        fig.savefig(OUT, dpi=200)
        print(f"Saved figure ({n_plotted} traces) to {OUT}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


    # old naive plot
    # for fp in filepaths:
    #     label = os.path.splitext(os.path.basename(fp))[0]
    #     try:
    #         freq, psd = load_esa_file(fp)
    #     except Exception as exc:
    #         print(f"Warning: could not read {fp}: {exc}", file=sys.stderr)
    #         continue
    #     if freq.size == 0:
    #         print(f"Warning: no valid data rows in {fp}", file=sys.stderr)
    #         continue
    #     ax.plot(freq, psd, label=label, linewidth=1.2)
    #     n_plotted += 1

    # if n_plotted == 0:
    #     sys.exit("No valid data found in any matched file; nothing to plot.")

    # if LOGX:
    #     ax.set_xscale("log")
