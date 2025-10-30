# C3plus_lite.py
# ------------------------------------------------------------
# Modèle C³⁺ (Version 2.0) — Implémentation légère (temps réel / embarqué)
# Auteur : Franck Charpentier (© 2025) — Tous droits réservés (usage non commercial)
# Dépendances : numpy, scipy  (pas de MNE requis)
# ------------------------------------------------------------

import numpy as np
import scipy.signal as sps
import scipy.linalg as spla

def spectral_flatness(psd):
    psd = np.clip(np.asarray(psd, dtype=float), 1e-12, None)
    geo = np.exp(np.mean(np.log(psd)))
    arith = np.mean(psd)
    return float(geo/arith)

def bandpass(x, fs, lo, hi, order=4):
    b, a = sps.butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return sps.filtfilt(b, a, x)

def kuramoto_order(phases):
    z = np.exp(1j*phases)
    R_t = np.abs(np.mean(z, axis=0))
    return float(R_t.mean()), float(R_t.std())

def energy_norm(X):
    en = (X**2).mean(axis=1)
    en = en / (np.median(en) + 1e-12)
    return float(np.mean(en))

def I_phi_lite(X):
    X = X - X.mean(axis=1, keepdims=True)
    Sigma = np.cov(X)
    Sigma += 1e-8*np.eye(Sigma.shape[0])
    logdet_whole = np.log(np.linalg.det(Sigma) + 1e-24)
    logvar_sum   = np.sum(np.log(np.var(X, axis=1) + 1e-24))
    return float(max(0.0, 1.0 - (logvar_sum / (logdet_whole if logdet_whole!=0 else 1.0))))

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

def C3plus_lite(Iphi, SF, R_bar, M, E, params, alpha=1.0, beta=1.15):
    Eopt, sigma, Ropt, Mopt, tau, nu = params
    GR = gauss_gate(R_bar, Ropt, tau)
    GM = gauss_gate(M, Mopt, nu)
    Co = SF * GR * GM
    return float((Iphi**alpha) * (Co**beta) * R_energy(E, Eopt, sigma))

if __name__ == '__main__':
    fs = 100.0
    t = np.arange(0, 60.0, 1/fs)
    x1 = np.sin(2*np.pi*10*t) + 0.3*np.sin(2*np.pi*20*t)
    x2 = 0.9*np.sin(2*np.pi*10*t + 0.5) + 0.2*np.sin(2*np.pi*20*t + 1.1)
    X = np.vstack([x1, x2])
    E_hist = np.array([energy_norm(X)]*30)
    SF, R_bar, M = Co_adapt_lite(X, fs)
    R_hist = np.array([R_bar]*30)
    M_hist = np.array([M]*30)
    params = auto_calibrate(E_hist, R_hist, M_hist)
    Iph = I_phi_lite(X)
    E = energy_norm(X)
    C = C3plus_lite(Iph, SF, R_bar, M, E, params)
    print({'Iphi': Iph, 'SF': SF, 'R': R_bar, 'M': M, 'E': E, 'params': params, 'C': C})
