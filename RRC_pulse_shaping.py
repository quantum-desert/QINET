import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fft import rfft, fft, rfftfreq, fftfreq  # scipy.fft is preferred over np.fft for speed

# functions
def comm_sinc(x):
	return np.sinc(np.pi*x)/(np.pi*x)

# define the shape of the raised cosine pulse
def rc_pulse(t, T, beta):
    p = np.zeros_like(t)
    denom = 1 - (2*beta*t/T)**2
    singular = np.isclose(denom, 0, atol=1e-9)
    safe = ~singular
    p[safe] = 1/T*np.sinc(t[safe]/T) * np.cos(np.pi*beta*t[safe]/T) / denom[safe]
    p[singular] = (np.pi/4)*np.sinc(1/(2*beta))
    return p

# define constants
Fs=250e3
Rb = 15.625e3 # bit rate
T = 1/Rb # time for one bit

f_cutoff = 10e3 # Hz, desired cutoff to choose beta
# N_bits = 5
# duration = T*N_bits

# # define functions
# beta=0.5
# t = np.linspace(0,duration,int(Fs*duration))
# RC_pulse = RC_impulse(t,T,beta)
# RC = RC_t(t,T,beta)

beta = 2/Rb*f_cutoff-1
print(f"Selected beta: {beta:.2f}")
# beta = 0.5 # roll off factor (0 = rectangular filter)
# theoretical BW
BW_RC = Rb/2*(1+beta) # converges to Rb in beta=1 (square wave) limit

span_symbols = 2   # truncate the pulse's tails at +/- this many symbol periods (see caveat below)

t_pulse = np.arange(-span_symbols*T, span_symbols*T, 1/Fs)
p_t = rc_pulse(t_pulse, T, beta)

# random symbol sequence
Nsym = 100
b = np.random.choice([-1,1], size=Nsym)

# build the waveform by superposition (this is the part that replaces simple concatenation)
samples_per_sym = int(Fs*T)
x = np.zeros(Nsym*samples_per_sym)
for k, bk in enumerate(b):
	start = k*samples_per_sym
	advance_len = min(len(p_t),len(x)-start)
	x[start:start+advance_len] += bk*p_t[0:advance_len]   # note: += , not =, since pulses overlap

# build time array
t = np.linspace(0,Nsym*samples_per_sym*Fs,Nsym*samples_per_sym)

# build ground truth
gt = max(abs(x))*np.repeat(b,samples_per_sym)

# concolve with matched filter

# compute fft
freq = rfftfreq(np.size(x),d=1./Fs)
ft = np.abs(rfft(x))


# build plot
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,6))
ax1.plot(t,x) # real signal
ax1.plot(t,gt)#  ground truth

ax2.plot(freq,ft)
ax2.axvline(BW_RC,label="RC Bandwidth")
plt.show()



