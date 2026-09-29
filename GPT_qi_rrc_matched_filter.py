"""
qi_rrc_matched_filter.py

Purpose
-------
A compact simulation of the classical DSP layer of a quantum-illumination-type
experiment in which:

    1. A known random binary (+1/-1) modulation sequence is generated.
    2. The symbols are pulse-shaped with a root-raised-cosine (RRC) filter.
    3. The weak returned waveform is embedded in colored additive noise.
    4. The receiver uses the time-reversed RRC as the matched filter.
    5. The result is sampled once per symbol.
    6. Output SNR is evaluated.
    7. beta (the RRC roll-off) can be swept to find the best SNR for a
       bandwidth-limited / colored-noise experiment.

This is NOT a quantum-state simulation.  It models the modulation and
demodulation layer that can sit around a QI correlation measurement.

Important assumptions
---------------------
* Binary symbols are independent and equiprobable: a_k in {-1, +1}.
* The RRC transmitter and RRC matched filter form an overall raised-cosine
  response.
* The nominal symbol rate is Rs = 1/T.
* For an ideal RRC, the positive-frequency cutoff is

      B = (1 + beta) Rs / 2.

* The ordinary matched filter is strictly SNR-optimal for white noise.
  Here we intentionally use bandwidth selection to reject a colored noise
  floor that rises outside the useful signal band.
* If the noise is strongly colored *inside* the RRC passband, the next step
  should be a whitened matched filter proportional to P*(f)/S_n(f).

Dependencies
------------
numpy
scipy
matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch


# ============================================================================
# 1. USER PARAMETERS
# ============================================================================

# Sampling rate [samples/s].
#
# Choose Fs/Rs to be an integer for a simple symbol-synchronous simulation.
# Here 256 kSa/s / 16 ksym/s = 16 samples/symbol exactly.
#
# If your hardware is fixed at 250 kSa/s and Rs = 16 ksym/s, then
# Fs/Rs = 15.625.  That is physically fine, but a real implementation should
# use fractional-delay timing recovery / rational resampling rather than
# rounding to 15 or 16 samples/symbol.
Fs = 256e3

# Symbol rate [symbols/s].  For binary modulation, this is also the bit rate.
Rs = 16e3
T = 1.0 / Rs

# RRC roll-off used for the detailed example.
beta = 0.25

# Length of the truncated RRC pulse, in symbols.
# An ideal RRC extends infinitely in time; 10-16 symbols is a common
# simulation range.  Larger span -> smaller truncation-induced ISI.
span_symbols = 12

# Number of random symbols.
n_symbols = 20_000

# Reproducible random sequence.
rng = np.random.default_rng(7)

# Weak return amplitude.
#
# This is a dimensionless simulation scale factor.  In an experiment you
# would replace it with the gain relating your known modulation waveform to
# the measured correlation / photocurrent observable.
signal_gain = 0.10

# RMS amplitude of the underlying WHITE detector-noise process before the
# high-frequency excess is applied.  This fixes the low-frequency noise floor.
# The total colored-noise RMS will therefore be larger than this value.
base_white_noise_rms = 1.0

# Synthetic colored-noise model:
# the PSD is approximately flat below noise_corner and rises above it.
#
# For beta = 0.25 and Rs = 16 ksym/s, the ideal RRC cutoff is:
#     (1 + 0.25)*16 kHz/2 = 10 kHz.
#
# Thus noise_corner = 10 kHz illustrates the situation where the detector
# becomes noisy just beyond the desired signal band.
noise_corner = 10e3

# PSD multiplier well above the corner.
# 100 means 20 dB higher noise PSD at high frequency.
high_frequency_noise_factor = 100.0

# Smoothness of the transition from low to high noise [Hz].
noise_transition = 1.0e3


# ============================================================================
# 2. ROOT-RAISED-COSINE FILTER
# ============================================================================

def rrc_taps(beta, span, sps):
    """
    Generate a symmetric, unit-energy root-raised-cosine FIR filter.

    Parameters
    ----------
    beta : float
        RRC roll-off factor, 0 <= beta <= 1.
    span : int
        Total pulse length in symbols.
    sps : int
        Samples per symbol.

    Returns
    -------
    h : ndarray
        RRC FIR taps.
    t_over_T : ndarray
        Time axis normalized by symbol period T.

    Mathematical expression
    -----------------------
    Let x = t/T.

    For x != 0 and x != +/-1/(4 beta),

        h(x) =
          [ sin(pi(1-beta)x)
            + 4 beta x cos(pi(1+beta)x) ]
          ------------------------------------------------
          [ pi x (1 - (4 beta x)^2) ].

    The removable singularities are filled using their analytic limits.

    The final discrete taps are normalized so that

        sum_n |h[n]|^2 = 1.

    That normalization is convenient for DSP:
    an isolated +/-1 symbol has unit matched-filter peak because

        h_MF[n] = h*[N-1-n]

    and, for this real symmetric pulse,

        sum_n h[n] h_MF[N-1-n] = sum_n h[n]^2 = 1.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must satisfy 0 <= beta <= 1.")
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive.")

    # Use an odd number of taps so the impulse response has an exact center.
    # span*sps should therefore be even; with sps=16 this is automatic.
    N = span * sps
    if N % 2 != 0:
        raise ValueError(
            "span*sps must be even so the symmetric FIR has an exact center tap."
        )
    half = N // 2
    n = np.arange(-half, half + 1)
    x = n / sps                      # x = t/T

    h = np.zeros_like(x, dtype=float)

    if beta == 0.0:
        # beta -> 0 gives the Nyquist sinc pulse.
        h = np.sinc(x)
    else:
        at_zero = np.isclose(x, 0.0, atol=1e-14)
        at_singular = np.isclose(
            np.abs(x), 1.0 / (4.0 * beta), atol=1e-12
        )
        regular = ~(at_zero | at_singular)

        xr = x[regular]

        h[regular] = (
            np.sin(np.pi * (1.0 - beta) * xr)
            + 4.0 * beta * xr
            * np.cos(np.pi * (1.0 + beta) * xr)
        ) / (
            np.pi * xr
            * (1.0 - (4.0 * beta * xr) ** 2)
        )

        # Analytic limit at t = 0.
        h[at_zero] = 1.0 - beta + 4.0 * beta / np.pi

        # Analytic limits at t = +/- T/(4 beta).
        h[at_singular] = (
            beta / np.sqrt(2.0)
            * (
                (1.0 + 2.0 / np.pi)
                * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi)
                * np.cos(np.pi / (4.0 * beta))
            )
        )

    # Unit-energy normalization for a discrete-time matched-filter simulation.
    h /= np.sqrt(np.sum(h**2))

    return h, x


# ============================================================================
# 3. RANDOM MODULATION AND TRANSMIT PULSE SHAPING
# ============================================================================

def make_random_symbols(N, rng):
    """
    Generate i.i.d. equiprobable binary modulation symbols.

        P(a_k = +1) = 1/2
        P(a_k = -1) = 1/2

    A +/-1 sequence is preferable to 0/1 here because it is approximately
    zero mean, so it does not intentionally create a large DC component.
    """
    return rng.choice(np.array([-1.0, +1.0]), size=N)


def pulse_shape(symbols, h, sps):
    """
    Upsample the symbols and convolve them with the transmit RRC.

    The discrete impulse train is

        x[n] = sum_k a_k delta[n - k*sps].

    The transmitted waveform is

        s[n] = x[n] * h[n].
    """
    upsampled = np.zeros(len(symbols) * sps)
    upsampled[::sps] = symbols

    # Full convolution is used so filter delay is explicit and easy to track.
    tx = fftconvolve(upsampled, h, mode="full")
    return tx


# ============================================================================
# 4. COLORED DETECTOR NOISE
# ============================================================================

def detector_noise_psd_shape(
    f,
    corner=noise_corner,
    high_factor=high_frequency_noise_factor,
    transition=noise_transition,
):
    """
    Dimensionless model for the *power* spectral density S_n(f).

    Below 'corner', S_n ~ 1.
    Above 'corner', S_n ~ high_factor.

    A tanh transition avoids an unrealistically sharp discontinuity.

    Replace THIS function with an interpolation of your measured detector
    noise PSD when you are ready to use experimental data.
    """
    step = 0.5 * (1.0 + np.tanh((f - corner) / transition))
    return 1.0 + (high_factor - 1.0) * step


def make_colored_noise(N, Fs, base_white_rms, rng):
    """
    Generate a real-valued Gaussian noise record with the desired PSD shape.

    Method:
      1. Start with white Gaussian noise w[n] with RMS = base_white_rms.
      2. FFT it.
      3. Multiply each spectral amplitude by sqrt(S_n(f)).
      4. Transform back to time.

    We deliberately DO NOT renormalize the final colored-noise record.
    Therefore the low-frequency floor stays fixed while the high-frequency
    noise is genuinely added on top of it.

    sqrt(S_n) is used because S_n is a POWER spectral density.
    """
    white = rng.normal(scale=base_white_rms, size=N)

    f = np.fft.rfftfreq(N, d=1.0 / Fs)
    W = np.fft.rfft(white)

    Sn_shape = detector_noise_psd_shape(f)
    Nf = W * np.sqrt(Sn_shape)

    noise = np.fft.irfft(Nf, n=N)

    # Remove accidental DC offset.
    noise -= np.mean(noise)

    return noise


# ============================================================================
# 5. RRC MATCHED FILTER AND SYMBOL SAMPLING
# ============================================================================

def matched_filter(h):
    """
    Construct the discrete-time matched filter

        h_MF[n] = h*[N - 1 - n].

    For a real symmetric RRC pulse, this is numerically the same shape as h,
    but writing it this way makes the matched-filter operation explicit.
    """
    return np.conj(h[::-1])


def sample_matched_filter_output(y_mf, n_symbols, sps, h):
    """
    Sample at the ideal symbol decision instants.

    Each RRC FIR has group delay

        D = (len(h)-1)/2.

    We pass through BOTH the transmit RRC and receive RRC, so the total delay is

        2D = len(h)-1.

    Therefore symbol k is sampled at

        n_k = (len(h)-1) + k*sps.
    """
    total_delay = len(h) - 1
    sample_indices = total_delay + np.arange(n_symbols) * sps
    return y_mf[sample_indices], sample_indices


# ============================================================================
# 6. SNR CALCULATION
# ============================================================================

def evaluate_link(symbols, h, sps, colored_noise, signal_gain):
    """
    Run one transmit/receive realization and calculate output SNR.

    We propagate the signal and noise separately through the matched filter.
    That lets us calculate a clean SNR without having to estimate signal and
    noise from the same finite data record.

    Two SNRs are reported:

    snr_noise_only:
        desired symbol power / matched-filtered detector-noise power.

    snr_effective:
        desired symbol power /
        (detector-noise power + finite-filter ISI power).

    The second is useful when beta is very small and the finite RRC span
    truncates the long pulse tails.
    """
    tx = pulse_shape(symbols, h, sps)

    if len(colored_noise) != len(tx):
        raise ValueError("Noise record and transmitted waveform lengths differ.")

    # Weak returned signal plus detector/background noise.
    returned_signal = signal_gain * tx
    received = returned_signal + colored_noise

    h_mf = matched_filter(h)

    # Filter total received data.
    y = fftconvolve(received, h_mf, mode="full")

    # Also filter signal and noise separately for diagnostics.
    y_sig = fftconvolve(returned_signal, h_mf, mode="full")
    y_noise = fftconvolve(colored_noise, h_mf, mode="full")

    z, idx = sample_matched_filter_output(
        y, len(symbols), sps, h
    )
    z_sig, _ = sample_matched_filter_output(
        y_sig, len(symbols), sps, h
    )
    z_noise, _ = sample_matched_filter_output(
        y_noise, len(symbols), sps, h
    )

    # With unit-energy RRC taps, the ideal desired matched-filter peak is
    # signal_gain * a_k.
    desired = signal_gain * symbols

    # Residual deterministic error from the finite RRC truncation.
    # In an infinitely long ideal RRC+RRC cascade this goes to zero at the
    # other symbol sampling instants.
    isi = z_sig - desired

    desired_power = np.mean(desired**2)
    noise_power = np.mean(z_noise**2)
    isi_power = np.mean(isi**2)

    snr_noise_only = desired_power / noise_power
    snr_effective = desired_power / (noise_power + isi_power)

    # Coherent recovery: multiply each measured sample by the known random
    # modulation sign.  The desired signal then adds with the same sign.
    #
    # This is the discrete analogue of correlation with the known modulation.
    correlated_samples = z * symbols
    coherent_mean = np.mean(correlated_samples)

    # Simple bit/sign decision, mainly as a sanity check.
    decisions = np.sign(z)
    decisions[decisions == 0] = 1.0
    symbol_error_rate = np.mean(decisions != symbols)

    return {
        "tx": tx,
        "received": received,
        "matched_output": y,
        "samples": z,
        "sample_indices": idx,
        "desired_power": desired_power,
        "noise_power": noise_power,
        "isi_power": isi_power,
        "snr_noise_only": snr_noise_only,
        "snr_effective": snr_effective,
        "coherent_mean": coherent_mean,
        "symbol_error_rate": symbol_error_rate,
    }


# ============================================================================
# 7. OPTIONAL SWEEP OF RRC ROLL-OFF beta
# ============================================================================

def sweep_beta(symbols, sps, span, Fs, signal_gain, noise):
    """
    Sweep beta while keeping:
      * the same random symbols,
      * the same detector-noise realization,
      * the same transmitted energy per symbol.

    This isolates the bandwidth/ISI tradeoff.

    Smaller beta:
        narrower frequency support, but longer time-domain tails and therefore
        more sensitivity to finite-span truncation and timing error.

    Larger beta:
        wider frequency support, but shorter / better-behaved time-domain pulse.
    """
    beta_values = np.linspace(0.05, 1.0, 40)

    snr_db = np.zeros_like(beta_values)
    isi_db = np.zeros_like(beta_values)
    bandwidths = np.zeros_like(beta_values)

    for i, b in enumerate(beta_values):
        h_b, _ = rrc_taps(b, span, sps)
        result = evaluate_link(
            symbols=symbols,
            h=h_b,
            sps=sps,
            colored_noise=noise,
            signal_gain=signal_gain,
        )

        snr_db[i] = 10.0 * np.log10(result["snr_effective"])

        # ISI relative to desired symbol power.
        if result["isi_power"] > 0:
            isi_db[i] = 10.0 * np.log10(
                result["isi_power"] / result["desired_power"]
            )
        else:
            isi_db[i] = -np.inf

        # Ideal positive-frequency RRC cutoff.
        bandwidths[i] = (1.0 + b) * Rs / 2.0

    return beta_values, bandwidths, snr_db, isi_db


# ============================================================================
# 8. MAIN PROGRAM
# ============================================================================

def main():
    # ------------------------------------------------------------------------
    # Basic rate checks
    # ------------------------------------------------------------------------
    sps_float = Fs / Rs
    sps = int(round(sps_float))

    if not np.isclose(sps_float, sps):
        raise ValueError(
            f"Fs/Rs = {sps_float:.6f}, not an integer. "
            "For this simple demonstration choose an integer samples/symbol. "
            "For your 250 kSa/s hardware at 16 ksym/s, use fractional timing "
            "or rational resampling instead of rounding."
        )

    print("============================================================")
    print("QI random-modulation / RRC matched-filter simulation")
    print("============================================================")
    print(f"Fs                   = {Fs/1e3:.3f} kSa/s")
    print(f"Rs                   = {Rs/1e3:.3f} ksym/s")
    print(f"T                    = {T*1e6:.3f} us")
    print(f"samples/symbol       = {sps}")
    print(f"RRC beta             = {beta:.3f}")
    print(
        "ideal RRC cutoff     = "
        f"{(1.0 + beta)*Rs/2.0/1e3:.3f} kHz"
    )
    print(f"colored-noise corner = {noise_corner/1e3:.3f} kHz")
    print()

    # ------------------------------------------------------------------------
    # Known random modulation sequence
    # ------------------------------------------------------------------------
    symbols = make_random_symbols(n_symbols, rng)

    print(f"mean(symbols)         = {np.mean(symbols):+.5f}")
    print(
        "lag-1 correlation    = "
        f"{np.mean(symbols[:-1]*symbols[1:]):+.5f}"
    )
    print()

    # ------------------------------------------------------------------------
    # RRC transmitter
    # ------------------------------------------------------------------------
    h, t_over_T = rrc_taps(beta, span_symbols, sps)

    # Unit energy check.
    print(f"sum(h^2)              = {np.sum(h**2):.12f}")

    # The RRC followed by its matched filter should approximate a raised cosine.
    rc_discrete = fftconvolve(h, matched_filter(h), mode="full")
    rc_center = len(rc_discrete) // 2

    # Sample the cascade at +/- integer symbol intervals.
    test_k = np.arange(-5, 6)
    rc_symbol_samples = rc_discrete[rc_center + test_k * sps]

    print("RC cascade samples at kT:")
    for k, value in zip(test_k, rc_symbol_samples):
        print(f"  k={k:+2d}: {value:+.6e}")
    print()

    # ------------------------------------------------------------------------
    # Make one transmitted waveform so we know the required noise-record length
    # ------------------------------------------------------------------------
    tx = pulse_shape(symbols, h, sps)

    # Generate a SINGLE detector-noise realization.  We will reuse it in the
    # beta sweep so the beta comparison is fair.
    noise = make_colored_noise(
        N=len(tx),
        Fs=Fs,
        base_white_rms=base_white_noise_rms,
        rng=rng,
    )

    # ------------------------------------------------------------------------
    # Evaluate the selected beta
    # ------------------------------------------------------------------------
    result = evaluate_link(
        symbols=symbols,
        h=h,
        sps=sps,
        colored_noise=noise,
        signal_gain=signal_gain,
    )

    snr_noise_db = 10.0 * np.log10(result["snr_noise_only"])
    snr_eff_db = 10.0 * np.log10(result["snr_effective"])

    print(f"matched-filter SNR        = {snr_noise_db:+.3f} dB")
    print(f"effective SNR incl. ISI   = {snr_eff_db:+.3f} dB")
    print(
        "ISI / desired power       = "
        f"{10*np.log10(result['isi_power']/result['desired_power']):+.3f} dB"
    )
    print(f"symbol error rate          = {result['symbol_error_rate']:.5f}")
    print(f"mean after code correlation= {result['coherent_mean']:+.6e}")
    print()

    # ------------------------------------------------------------------------
    # Sweep beta to find the best effective SNR for this noise PSD
    # ------------------------------------------------------------------------
    betas, bandwidths, snr_sweep_db, isi_sweep_db = sweep_beta(
        symbols=symbols,
        sps=sps,
        span=span_symbols,
        Fs=Fs,
        signal_gain=signal_gain,
        noise=noise,
    )

    i_best = np.argmax(snr_sweep_db)

    print("Best beta in this simulation:")
    print(f"  beta              = {betas[i_best]:.3f}")
    print(f"  cutoff bandwidth  = {bandwidths[i_best]/1e3:.3f} kHz")
    print(f"  effective SNR     = {snr_sweep_db[i_best]:+.3f} dB")
    print()

    # ========================================================================
    # 9. PLOTS
    # ========================================================================

    # ------------------------------------------------------------------------
    # Plot 1: RRC pulse in time
    # ------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_over_T, h)
    plt.xlabel("Time / symbol period  t/T")
    plt.ylabel("RRC tap amplitude")
    plt.title(f"Root-raised-cosine pulse, beta={beta}")
    plt.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------------------
    # Plot 2: RRC and RRC*RRC frequency responses
    # ------------------------------------------------------------------------
    Nfft = 65536
    f = np.fft.rfftfreq(Nfft, d=1.0/Fs)

    H = np.fft.rfft(h, n=Nfft)
    H_rc = np.fft.rfft(rc_discrete, n=Nfft)

    # Normalize for shape comparison only.
    H_mag = np.abs(H) / np.max(np.abs(H))
    Hrc_mag = np.abs(H_rc) / np.max(np.abs(H_rc))

    plt.figure()
    plt.plot(f/1e3, 20*np.log10(np.maximum(H_mag, 1e-8)),
             label="RRC")
    plt.plot(f/1e3, 20*np.log10(np.maximum(Hrc_mag, 1e-8)),
             label="RRC x RRC = RC")
    plt.axvline(
        (1.0 + beta)*Rs/2.0/1e3,
        linestyle="--",
        label="Ideal cutoff",
    )
    plt.xlim(0, 30)
    plt.ylim(-100, 5)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Normalized magnitude [dB]")
    plt.title("Transmit RRC and overall raised-cosine response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------------
    # Plot 3: PSD of transmitted random modulation and detector noise
    # ------------------------------------------------------------------------
    f_tx, Ptx = welch(
        result["tx"],
        fs=Fs,
        nperseg=min(8192, len(result["tx"])),
        scaling="density",
    )
    f_n, Pn = welch(
        noise,
        fs=Fs,
        nperseg=min(8192, len(noise)),
        scaling="density",
    )

    # Normalize each curve to its own maximum; the point here is spectral shape.
    Ptx_db = 10*np.log10(np.maximum(Ptx/np.max(Ptx), 1e-12))
    Pn_db = 10*np.log10(np.maximum(Pn/np.min(Pn[Pn > 0]), 1e-12))

    plt.figure()
    plt.plot(f_tx/1e3, Ptx_db, label="RRC-shaped random modulation")
    plt.plot(f_n/1e3, Pn_db, label="Detector noise PSD (relative to floor)")
    plt.axvline(
        (1.0 + beta)*Rs/2.0/1e3,
        linestyle="--",
        label="RRC cutoff",
    )
    plt.xlim(0, 40)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Relative PSD [dB]")
    plt.title("Signal bandwidth versus colored detector noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------------
    # Plot 4: Effective matched-filter SNR versus beta
    # ------------------------------------------------------------------------
    plt.figure()
    plt.plot(betas, snr_sweep_db, marker="o")
    plt.axvline(betas[i_best], linestyle="--")
    plt.xlabel("RRC roll-off beta")
    plt.ylabel("Effective output SNR [dB]")
    plt.title("Choose beta to maximize recovered SNR")
    plt.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------------------
    # Plot 5: Same SNR sweep, but versus actual RRC cutoff bandwidth
    # ------------------------------------------------------------------------
    plt.figure()
    plt.plot(bandwidths/1e3, snr_sweep_db, marker="o")
    plt.axvline(noise_corner/1e3, linestyle="--",
                label="Noise-floor corner")
    plt.xlabel("Ideal positive-frequency RRC cutoff [kHz]")
    plt.ylabel("Effective output SNR [dB]")
    plt.title("Recovered SNR versus occupied detection bandwidth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
