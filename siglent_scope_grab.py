#!/usr/bin/env python3
"""
siglent_scope.py -- LAN download of channels from a Siglent SDS scope to CSV,
plus phase statistics for atan2(Y, X) of LIA X/Y outputs.

HOW TO RUN FROM A TEXT EDITOR / IDE
-----------------------------------
1. Edit the USER SETTINGS block just below.
2. Set ACTION to one of: "probe", "dump", "capture", "analyze".
3. Press your editor's Run button (VS Code, PyCharm, Spyder, IDLE F5, ...).
   No command-line arguments are used.
Requires only Python 3 + numpy (matplotlib optional if PLOT = True).

ACTIONS
-------
  "probe"    Identify the scope and dump the waveform-header fields this
             script relies on.  RUN THIS FIRST.
  "dump"     Download CHANNELS. By default writes ONE CSV PER CHANNEL, named like
             the scope's own export (SDS5104X_HD_CSV_DC<V_DC>V_C<n>_<idx>.csv),
             each with 4 header lines then "time, volts" rows.
  "capture"  Download X_CH / Y_CH, decimate, compute gamma_hat = atan2(Y, X),
             save raw (.npz), decimated (.csv) and statistics (.json).
  "analyze"  Re-run decimation/statistics offline on a saved .npz (no scope).

STATUS / WHAT IS AND ISN'T VERIFIED
-----------------------------------
The waveform-download sequence (:WAVEFORM:SOURCE / START / POINT /
PREAMBLE? / DATA?), the 346-byte header layout, and the code->volts scaling
follow a third-party script written against the SDS2000X Plus (FW >= 1.3.5R3).
They are NOT confirmed for your model.  Each download prints mean/min/max in
volts per channel: compare these with the scope's on-screen readout before
trusting any number.  Nothing here arms or triggers the scope: press
SINGLE/STOP on the front panel so the acquisition you want is on screen.
"""

# =============================================================================
#                              USER SETTINGS
# =============================================================================
ACTION = "dump"            # "probe" | "dump" | "capture" | "analyze"

SCOPE_IP = "10.11.13.220"
SCOPE_PORT = 5025           # Siglent raw-socket SCPI port
TIMEOUT_S = 30.0            # network timeout while downloading

# Output location.  None = folder containing this script (falls back to the
# current working directory if the editor doesn't define __file__).
OUT_DIR = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/9-29-Phase/phiLO=+33/"
OUT_BASENAME = "run01"      # files: run01.csv, run01.npz, ...

# --- probe / dump ------------------------------------------------------------
CHANNELS = ["C2","C3", "C4"]     # e.g. ["C1", "C3"];  None = all channels reported ON
STRIDE = 1                  # dump: keep every Nth sample (shrinks huge records)

# --- dump: one file per channel, named like the scope's own CSV export --------
PER_CHANNEL_FILES = True    # False = one combined CSV named OUT_BASENAME.csv
V_DC = 4                  # DC voltage (V) applied at the PM for this dump; goes in
                            # every file name (same for all channels).  A number is
                            # written like 1.5 -> "DC1.5V"; a string is used verbatim.
# {vdc} = V_DC text, {ch} = channel number, {idx} = repeat number at this V_DC
# ("auto" below = 1 + highest index already in OUT_DIR for the same V_DC).
FILENAME_TEMPLATE = "SDS5104X_HD_CSV_DC{vdc}V_C{ch}_{idx}.csv"
FILENAME_GLOB = "SDS5104X_HD_CSV_DC*V_C*_*.csv"   # every name is checked against this
FILE_INDEX = "auto"         # "auto", or an integer to force the index
HEADER_LINES = 4            # lines before the data (your MATLAB uses NumHeaderLines=4)

# --- capture / analyze (phase estimation from LIA X, Y) -----------------------
LOCK = "C2"
X_CH = "C4"                 # scope channel wired to LIA X output
Y_CH = "C3"                 # scope channel wired to LIA Y output
SPACING_S = 0.078           # sample spacing (s); >= LIA filter wait time
DECIMATE_MODE = "point"     # "point" = sample every SPACING_S; "boxcar" = block mean
SKIP_S = 1.0                # discard first SKIP_S seconds (filter settling)
PLOT = False                # show matplotlib plots after capture
NPZ_FILE = "run01.npz"      # analyze: file to re-process (relative to OUT_DIR)

# Advanced: override ADC codes-per-division used for volt scaling (None = auto).
CODE_PER_DIV = None
# =============================================================================

import fnmatch
import glob
import json
import os
import re
import socket
import struct
import time

import numpy as np


# ----------------------------------------------------------------------------
# Raw-socket SCPI client
# ----------------------------------------------------------------------------
class Scope:
    def __init__(self, ip, port=5025, timeout=10.0):
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.timeout = timeout

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def write(self, cmd):
        self.sock.sendall((cmd.strip() + "\n").encode("ascii"))

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise ConnectionError("socket closed while reading")
            buf += chunk
        return bytes(buf)

    def _drain(self, wait=0.05):
        """Swallow trailing terminator bytes so they don't poison the next reply."""
        self.sock.settimeout(wait)
        try:
            while self.sock.recv(4096):
                pass
        except (socket.timeout, BlockingIOError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def query(self, cmd, timeout=None):
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            self.write(cmd)
            line = bytearray()
            while True:
                b = self.sock.recv(1)
                if not b:
                    raise ConnectionError("socket closed")
                if b == b"\n":
                    if line:            # ignore stray leading newlines
                        break
                    continue
                line += b
            return line.decode("latin_1").strip()
        finally:
            self.sock.settimeout(self.timeout)

    def query_block(self, cmd):
        """Send query, return payload of an IEEE-488.2 definite-length block."""
        self.write(cmd)
        b = self.sock.recv(1)
        while b != b"#":                # skip any prefix bytes
            if not b:
                raise ConnectionError("socket closed before block header")
            b = self.sock.recv(1)
        ndig = int(self._recv_exact(1))
        if not 1 <= ndig <= 9:
            raise ValueError(f"unexpected block header digit count {ndig}")
        length = int(self._recv_exact(ndig))
        payload = self._recv_exact(length)
        self._drain()
        return payload


# ----------------------------------------------------------------------------
# Waveform header parsing and code -> volts scaling
# ----------------------------------------------------------------------------
def parse_wavedesc(d):
    """Field offsets follow the SDS2000X Plus layout (346-byte header)."""
    if d[0:8] != b"WAVEDESC":
        raise ValueError("preamble does not start with 'WAVEDESC' -- this model "
                         "may use a different dialect; run ACTION='probe' and inspect")
    w = {}
    w["comm_type"], w["comm_order"], w["desc_len"] = struct.unpack("<hhl", d[32:40])
    w["unit_frame_points"] = struct.unpack("<l", d[116:120])[0]
    w["v_gain"], w["v_offset"], w["code_per_div"] = struct.unpack("<fff", d[156:168])
    w["adc_bit"], w["frame_index"], w["h_interval"], w["h_offset"] = struct.unpack(
        "<hhfd", d[172:188])
    _, _, w["probe_factor"], _, _ = struct.unpack("<hhfhh", d[324:336])
    if w["comm_order"] != 0:
        raise ValueError("MSB-first data not handled")
    return w


def codes_to_volts(codes, w, code_per_div=None):
    """volts = code * gain*probe/code_per_div - offset*probe (per reference script)."""
    if code_per_div is None:
        code_per_div = 30.0 * (256.0 if w["comm_type"] == 1 else 1.0)
    return (codes.astype(np.float64) * (w["v_gain"] * w["probe_factor"] / code_per_div)
            - w["v_offset"] * w["probe_factor"])


def fetch_channel(sc, ch, code_per_div=None, verbose=True):
    sc.write(f":WAVEFORM:SOURCE {ch}")
    sc.write(":WAVEFORM:START 0")
    sc.write(":WAVEFORM:POINT 0")       # 0 = as many points per read as allowed
    w = parse_wavedesc(sc.query_block(":WAVEFORM:PREAMBLE?"))
    bps = w["comm_type"] + 1
    n_total = w["unit_frame_points"]
    if verbose:
        print(f"  {ch}: {n_total} pts, {8 * bps}-bit words, dt={w['h_interval']:.6g} s "
              f"(fs={1 / w['h_interval']:.6g} Hz), ADC={w['adc_bit']} bit, "
              f"probe={w['probe_factor']:g}")
    buf, got = bytearray(), 0
    while got < n_total:
        sc.write(f":WAVEFORM:START {got}")
        chunk = sc.query_block(":WAVEFORM:DATA?")
        if not chunk:
            raise RuntimeError("empty data block; is the scope stopped with data?")
        buf += chunk
        got += len(chunk) // bps
    dtype = "<i2" if bps == 2 else "<i1"
    codes = np.frombuffer(bytes(buf), dtype=dtype)[:n_total]
    return codes_to_volts(codes, w, code_per_div), w


def enabled_channels(sc, n_max=4):
    """Analog channels whose display switch is ON (uses :CHANnel<n>:SWITch?)."""
    out = []
    for i in range(1, n_max + 1):
        try:
            if sc.query(f":CHANnel{i}:SWITch?", timeout=2.0).upper().startswith("ON"):
                out.append(f"C{i}")
        except (socket.timeout, OSError):
            sc._drain(0.3)              # channel doesn't exist on this model
    return out


# ----------------------------------------------------------------------------
# Decimation + circular phase statistics
# ----------------------------------------------------------------------------
def decimate(x, y, dt, spacing, mode="point", skip=0.0):
    """Return X, Y at `spacing` seconds after discarding `skip` seconds.
    mode='point': sample every spacing.  mode='boxcar': mean over each block."""
    i0 = int(round(skip / dt))
    step = max(1, int(round(spacing / dt)))
    x, y = x[i0:], y[i0:]
    if mode == "point":
        return x[::step], y[::step], step * dt
    n = (len(x) // step) * step
    return (x[:n].reshape(-1, step).mean(1), y[:n].reshape(-1, step).mean(1), step * dt)


def phase_stats(gamma):
    """Circular mean, sample std about it, lag-1 autocorr, effective N and SEM."""
    n = len(gamma)
    mean = np.angle(np.mean(np.exp(1j * gamma)))
    dev = np.angle(np.exp(1j * (gamma - mean)))          # wrapped to (-pi, pi]
    std = np.sqrt(np.sum(dev ** 2) / (n - 1))
    r1 = float(np.sum(dev[1:] * dev[:-1]) / np.sum(dev ** 2)) if n > 2 else float("nan")
    neff = n * (1 - r1) / (1 + r1) if r1 > 0 else float(n)
    return dict(n=n, mean=float(mean), std=float(std), lag1=r1,
                n_eff=float(min(neff, n)), sem=float(std / np.sqrt(min(neff, n))))


def report(x, y, dt, spacing, mode="point", skip=0.0):
    xs, ys, step_t = decimate(x, y, dt, spacing, mode, skip)
    g = np.arctan2(ys, xs)
    s = phase_stats(g)
    R = float(np.mean(np.hypot(xs, ys)))
    print(f"\nDecimation: {mode}, actual spacing {step_t * 1e3:.3f} ms "
          f"(requested {spacing * 1e3:.3f} ms), skipped first {skip:g} s")
    print(f"N samples            : {s['n']}")
    print(f"mean R = |X+iY|      : {R:.6g} V")
    print(f"gamma_hat (circ mean): {s['mean']:.6f} rad")
    print(f"std (single-sample)  : {s['std']:.6f} rad")
    print(f"lag-1 autocorr       : {s['lag1']:.3f}   (should be ~0 if samples independent)")
    print(f"N_eff                : {s['n_eff']:.1f}")
    print(f"SEM = std/sqrt(N_eff): {s['sem']:.6f} rad")
    if s["lag1"] > 0.1:
        hint = ("increase SPACING_S" if mode == "boxcar"
                else "increase SPACING_S (or set DECIMATE_MODE = 'boxcar')")
        print(f"WARNING: samples look correlated; {hint}.")
    return xs, ys, g, s


# ----------------------------------------------------------------------------
# Actions (each takes explicit arguments, so they can also be called from a
# REPL / notebook:  dump(ip="10.11.13.220", channels=["C1"], out_path="x") )
# ----------------------------------------------------------------------------
def probe(ip, port, channels):
    sc = Scope(ip, port)
    print("IDN:", sc.query("*IDN?"))
    for q in [":CHANnel1:SWITch?", ":CHANnel2:SWITch?", ":CHANnel3:SWITch?",
              ":CHANnel4:SWITch?", ":ACQuire:MODE?", ":ACQuire:SRATe?",
              ":ACQuire:MDEPth?", ":ACQuire:POINts?", ":ACQuire:RESolution?",
              ":TIMebase:SCALe?", ":WAVEFORM:MAXPOINT?"]:
        try:
            print(f"{q:24s} -> {sc.query(q, timeout=2.0)}")
        except (socket.timeout, OSError):
            sc._drain(0.3)
            print(f"{q:24s} -> (no reply; unsupported on this model/firmware?)")
    for ch in (channels or ["C1", "C2"]):
        print(f"\nPreamble for {ch}:")
        try:
            sc.write(f":WAVEFORM:SOURCE {ch}")
            raw = sc.query_block(":WAVEFORM:PREAMBLE?")
            print(f"  block payload length: {len(raw)} bytes (expected 346 on SDS2000X Plus)")
            print(f"  first 16 bytes: {raw[:16]!r}")
            print("  parsed:", json.dumps(parse_wavedesc(raw), indent=2))
        except Exception as e:      # noqa: BLE001 -- diagnostic tool
            print("  FAILED:", repr(e))
    sc.close()


# --- scope-style file naming (used by dump) ---------------------------------
def fmt_vdc(v):
    """V_DC text for the file name: 1.5 -> '1.5', -2 -> '-2', 0 -> '0'; str kept as is."""
    if isinstance(v, str):
        return v.strip()
    return f"{float(v) + 0.0:g}"          # "+ 0.0" turns -0.0 into 0.0


def next_file_index(out_dir, template, vdc_text):
    """1 + the largest trailing _<n>.csv index among files with the same V_DC."""
    pat = template.format(vdc=vdc_text, ch="*", idx="*")
    best = 0
    for f in glob.glob(os.path.join(out_dir, pat)):
        m = re.search(r"_(\d+)\.csv$", f)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def write_scope_csv(path, t, v, ch, vdc_text, header_lines):
    n = len(v)
    header = [f"Record Length,{n},Points",
              f"Sample Interval,{t[1] - t[0]:.7g},s" if n > 1 else "Sample Interval,,s",
              f"Channel,{ch},V_DC,{vdc_text}V",
              "Second,Volt"]
    header = header[-header_lines:] if header_lines > 0 else []
    with open(path, "w", newline="") as f:
        for line in header:
            f.write(line + "\n")
        np.savetxt(f, np.column_stack([t, v]), delimiter=",", fmt="%.9g")


def dump(ip, port, channels, out_dir, out_base, per_channel=True, v_dc=0.0,
         template=None, glob_pat=None, file_index="auto", header_lines=4, stride=1,
         timeout=30.0, code_per_div=None):
    """Download channels to CSV (one file per channel, scope-style names)."""
    sc = Scope(ip, port, timeout=timeout)
    print("Connected:", sc.query("*IDN?"))
    chans = [c.upper() for c in channels] if channels else enabled_channels(sc)
    if not chans:
        raise RuntimeError("No channels selected and none reported ON; set CHANNELS")
    print("Channels:", ", ".join(chans), "(scope must be STOPPED with data on screen)")
    data, dts = {}, {}
    for ch in chans:
        print(f"Fetching channel: {ch}")
        v, w = fetch_channel(sc, ch, code_per_div)
        data[ch], dts[ch] = v[::stride], w["h_interval"] * stride
        print(f"     mean {v.mean():+.5f} V, min {v.min():+.5f}, max {v.max():+.5f}")
    sc.close()

    if per_channel:
        vdc = fmt_vdc(v_dc)
        idx = next_file_index(out_dir, template, vdc) if str(file_index).lower() == "auto" \
            else int(file_index)
        for ch in chans:                      # validate ALL names before writing any
            name = template.format(vdc=vdc, ch=ch[1:], idx=idx)
            if glob_pat and not fnmatch.fnmatchcase(name, glob_pat):
                raise ValueError(f"File name {name!r} does not match your pattern "
                                 f"{glob_pat!r}. Fix FILENAME_TEMPLATE or V_DC.")
        for ch in chans:
            name = template.format(vdc=vdc, ch=ch[1:], idx=idx)
            t = np.arange(len(data[ch])) * dts[ch]
            path = os.path.join(out_dir, name)
            write_scope_csv(path, t, data[ch], ch, vdc, header_lines)
            print(f"Wrote {path}: {len(t)} rows")
        print("(time 0 = first sample, not the trigger)")
        return

    same = (len({len(v) for v in data.values()}) == 1
            and max(dts.values()) - min(dts.values()) < 1e-12)
    out_path = os.path.join(out_dir, out_base)
    if same:
        n, dt = len(data[chans[0]]), dts[chans[0]]
        cols = [np.arange(n) * dt] + [data[c] for c in chans]
        np.savetxt(out_path + ".csv", np.column_stack(cols), delimiter=",", fmt="%.9g",
                   header="time_s," + ",".join(f"{c}_V" for c in chans), comments="")
        print(f"Wrote {out_path}.csv: {n} rows x {len(chans) + 1} cols "
              f"(dt = {dt:.6g} s; time 0 = first sample, not the trigger)")
    else:
        print("Channels differ in length/sample interval -> one file per channel.")
        for c in chans:
            t = np.arange(len(data[c])) * dts[c]
            np.savetxt(f"{out_path}_{c}.csv", np.column_stack([t, data[c]]), delimiter=",",
                       fmt="%.9g", header=f"time_s,{c}_V", comments="")
            print(f"Wrote {out_path}_{c}.csv: {len(data[c])} rows")


def capture(ip, port, x_ch, y_ch, out_path, spacing, mode="point", skip=0.0,
            timeout=30.0, code_per_div=None, plot=False):
    sc = Scope(ip, port, timeout=timeout)
    print("Connected:", sc.query("*IDN?"))
    print("Downloading (scope must be STOPPED with the acquisition you want) ...")
    t0 = time.time()
    x, wx = fetch_channel(sc, x_ch, code_per_div)
    y, wy = fetch_channel(sc, y_ch, code_per_div)
    sc.close()
    print(f"Download took {time.time() - t0:.1f} s")
    if len(x) != len(y) or abs(wx["h_interval"] - wy["h_interval"]) > 1e-12:
        raise RuntimeError("X and Y records differ in length/sample interval")
    dt = wx["h_interval"]
    print(f"Record length: {len(x) * dt:.4f} s")
    for name, v in (("X", x), ("Y", y)):
        print(f"  {name}: mean {v.mean():+.5f} V, min {v.min():+.5f}, max {v.max():+.5f}"
              "   <-- compare with scope on-screen readout")
    np.savez(out_path + ".npz", x=x.astype(np.float32), y=y.astype(np.float32), dt=dt,
             x_ch=x_ch, y_ch=y_ch)
    xs, ys, g, s = report(x, y, dt, spacing, mode, skip)
    np.savetxt(out_path + "_decimated.csv", np.c_[xs, ys, g], delimiter=",",
               header="X_V,Y_V,gamma_rad", comments="")
    with open(out_path + "_stats.json", "w") as f:
        json.dump(s, f, indent=2)
    print(f"\nSaved {out_path}.npz, {out_path}_decimated.csv, {out_path}_stats.json")
    if plot:
        show_plots(x, y, dt, g)


def analyze(npz_path, spacing, mode="point", skip=0.0):
    d = np.load(npz_path)
    report(d["x"].astype(np.float64), d["y"].astype(np.float64), float(d["dt"]),
           spacing, mode, skip)


def show_plots(x, y, dt, g):
    import matplotlib.pyplot as plt
    t = np.arange(len(x)) * dt
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].plot(t, x, lw=0.5, label="X"); ax[0].plot(t, y, lw=0.5, label="Y")
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("V"); ax[0].legend()
    ax[1].plot(g, ".", ms=3); ax[1].set_xlabel("sample #"); ax[1].set_ylabel("gamma_hat (rad)")
    ax[2].hist(g, bins=40); ax[2].set_xlabel("gamma_hat (rad)")
    fig.tight_layout(); plt.show()


# ----------------------------------------------------------------------------
# Entry point: uses the USER SETTINGS at the top of the file
# ----------------------------------------------------------------------------
def _out_dir():
    if OUT_DIR:
        d = OUT_DIR
    else:
        try:
            d = os.path.dirname(os.path.abspath(__file__))
        except NameError:               # some interactive consoles
            d = os.getcwd()
    os.makedirs(d, exist_ok=True)
    return d


def main():
    out_path = os.path.join(_out_dir(), OUT_BASENAME)
    print(f"ACTION = {ACTION!r}   output folder: {os.path.dirname(out_path)}")
    if ACTION == "probe":
        probe(SCOPE_IP, SCOPE_PORT, CHANNELS)
    elif ACTION == "dump":
        dump(SCOPE_IP, SCOPE_PORT, CHANNELS, _out_dir(), OUT_BASENAME, PER_CHANNEL_FILES,
             V_DC, FILENAME_TEMPLATE, FILENAME_GLOB, FILE_INDEX, HEADER_LINES, STRIDE,
             TIMEOUT_S, CODE_PER_DIV)
    elif ACTION == "capture":
        capture(SCOPE_IP, SCOPE_PORT, X_CH, Y_CH, out_path, SPACING_S, DECIMATE_MODE,
                SKIP_S, TIMEOUT_S, CODE_PER_DIV, PLOT)
    elif ACTION == "analyze":
        npz = NPZ_FILE if os.path.isabs(NPZ_FILE) else os.path.join(_out_dir(), NPZ_FILE)
        analyze(npz, SPACING_S, DECIMATE_MODE, SKIP_S)
    else:
        raise ValueError(f"Unknown ACTION {ACTION!r}; use probe/dump/capture/analyze")


if __name__ == "__main__":
    main()