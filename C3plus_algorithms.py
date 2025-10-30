
# C3plus_algorithms.py
# ------------------------------------------------------------
# Modèle C³⁺ (Version 2.0) — Algorithmes de référence
# Auteur : Franck Charpentier (© 2025) — Tous droits réservés (usage non commercial)
# Dépendances : numpy, scipy, (optionnel) mne, (fallback) pyedflib
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import scipy.signal as sps
import scipy.linalg as spla

try:
    import mne
except Exception:
    mne = None
try:
    import pyedflib
except Exception:
    pyedflib = None

def bandpass(x, fs, lo, hi, order=4):
    b, a = sps.butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return sps.filtfilt(b, a, x)

def spectral_flatness(psd):
    psd = np.clip(np.asarray(psd, dtype=float), 1e-12, None)
    geo = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(geo/arith)

def kuramoto_order(phases):
    z = np.exp(1j*phases)
    R_t = np.abs(np.mean(z, axis=0))
    return float(R_t.mean()), float(R_t.std())

def load_edf(path, prefer_channels=('Fpz-Cz','Pz-Oz','EEG Fpz-Cz','EEG Pz-Oz')):
    if mne is not None:
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq'])
        ch_names = raw.ch_names
        picks = []
        for cand in prefer_channels:
            if cand in ch_names:
                picks.append(ch_names.index(cand))
        if len(picks) < 2:
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            picks = eeg_picks[:2].tolist()
        data = raw.get_data(picks=picks)
        sel_names = [ch_names[i] for i in picks]
        return data, fs, sel_names
    if pyedflib is None:
        raise RuntimeError("Ni MNE ni pyedflib disponibles pour lire l'EDF.")
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    fs = float(f.getSampleFrequency(0))
    labels = [f.getLabel(i) for i in range(n)]
    picks = []
    for cand in prefer_channels:
        if cand in labels:
            picks.append(labels.index(cand))
    if len(picks) < 2:
        picks = [0, 1] if n >= 2 else [0]
    data = np.vstack([f.readSignal(i) for i in picks])
    f.close()
    sel_names = [labels[i] for i in picks]
    return data, fs, sel_names

def window_iter(X, fs, win_s=60):
    step = int(win_s * fs)
    n = X.shape[1]
    for a in range(0, n - step, step):
        yield X[:, a:a+step]

def energy_norm(X):
    en = (X**2).mean(axis=1)
    en = en / (np.median(en) + 1e-12)
    return float(np.mean(en))

def I_phi(X):
    Xz = X - X.mean(axis=1, keepdims=True)
    Sigma = np.cov(Xz)
    Sigma += 1e-8 * np.eye(Sigma.shape[0])
    logdet_whole = np.log(np.linalg.det(Sigma) + 1e-24)
    logdet_diag = np.sum(np.log(np.diag(Sigma) + 1e-24))
    return float(max(0.0, logdet_diag - logdet_whole))

def Co_adapt_lite(X, fs):
    Xc = X - X.mean(axis=1, keepdims=True)
    U, S, Vt = spla.svd(Xc, full_matrices=False)
    k = min(5, X.shape[0])
    Y = (U[:, :k].T @ Xc)
    Yf = np.vstack([bandpass(y, fs, 8, 30) for y in Y])
    Yh = sps.hilbert(Yf)
    phases = np.angle(Yh)
    R_bar, M = kuramoto_order(phases)
    sf_vals = []
    for y in Y:
        f, Pxx = sps.welch(y, fs=fs, nperseg=min(2048, len(y)))
        sf_vals.append(spectral_flatness(Pxx))
    SF = float(np.mean(sf_vals))
    return SF, R_bar, M

def robust_iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))

def auto_calibrate(E_hist, R_hist, M_hist):
    med = np.median
    Eopt = float(med(E_hist))
    sigma = float(robust_iqr(E_hist)/1.349 + 1e-12)
    Ropt = float(med(R_hist))
    Mopt = float(med(M_hist))
    tau  = float(robust_iqr(R_hist)/1.349 + 1e-12)
    nu   = float(robust_iqr(M_hist)/1.349 + 1e-12)
    return Eopt, sigma, Ropt, Mopt, tau, nu

def gauss_gate(x, x0, s):
    s = max(s, 1e-6)
    return float(np.exp(-((x - x0)**2) / (2*(s**2))))

def R_energy(E, Eopt, sigma):
    return float(np.exp(-((E - Eopt)**2)/(2*(sigma**2) + 1e-24)))

def C3_plus_score(Iphi, SF, R_bar, M, E, params, alpha=1.0, beta=1.15):
    Eopt, sigma, Ropt, Mopt, tau, nu = params
    GR = gauss_gate(R_bar, Ropt, tau)
    GM = gauss_gate(M, Mopt, nu)
    Co = SF * GR * GM
    return float((Iphi**alpha) * (Co**beta) * R_energy(E, Eopt, sigma))

def compute_c3plus_on_edf(path, win_s=60):
    X, fs, chs = load_edf(path)
    E_hist, R_hist, M_hist, blocks = [], [], [], []
    for seg in window_iter(X, fs, win_s=win_s):
        E = energy_norm(seg)
        SF, R_bar, M = Co_adapt_lite(seg, fs)
        Iph = I_phi(seg)
        E_hist.append(E); R_hist.append(R_bar); M_hist.append(M)
        blocks.append((Iph, SF, R_bar, M, E))
    import numpy as np
    E_hist = np.array(E_hist); R_hist = np.array(R_hist); M_hist = np.array(M_hist)
    params = auto_calibrate(E_hist, R_hist, M_hist)
    scores = [C3_plus_score(Iph, SF, Rb, Mm, Ee, params) for (Iph,SF,Rb,Mm,Ee) in blocks]
    return {'channels': chs, 'fs': fs, 'params': {'Eopt': params[0], 'sigma': params[1], 'Ropt': params[2], 'Mopt': params[3], 'tau': params[4], 'nu': params[5]},
            'n_windows': len(scores), 'mean_C': float(np.mean(scores)), 'median_C': float(np.median(scores))}
