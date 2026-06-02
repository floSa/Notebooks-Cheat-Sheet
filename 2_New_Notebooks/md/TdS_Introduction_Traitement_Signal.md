---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 🌊 Traitement du Signal — Introduction
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur le **traitement du signal** : la boîte à outils mathématique pour analyser, filtrer, décomposer des signaux 1D (audio, capteurs, séries temporelles physiques).

Couvre :

1. **Signal continu vs discret** — échantillonnage, Nyquist.
2. **Transformée de Fourier** (FFT) — spectre fréquentiel.
3. **Fenêtrage** et STFT (Short-Time Fourier Transform) — spectrogrammes.
4. **Ondelettes** (Wavelets) — analyse temps-fréquence multi-échelle.
5. **Transformée de Hilbert** — enveloppe et phase instantanée.
6. **Filtrage** — passe-bas, passe-haut, passe-bande (Butterworth, FIR/IIR).
7. **Débruitage** — moyenne mobile, savgol, médian, wavelet denoising.
8. **Décomposition** — EMD (Empirical Mode Decomposition).
9. **Applications 2026** : audio (librosa), bioacoustique, IoT.

> Pour les **séries temporelles statistiques/ML** (forecasting, anomalies), voir `TS_Time_Series_Overview`.
<!-- #endregion -->

<!-- #region -->
## 1. Signal discret et échantillonnage
<!-- #endregion -->

<!-- #region -->
**Théorème de Shannon-Nyquist** : pour reconstruire un signal continu de fréquence max `f_max`, il faut l'échantillonner à `f_s ≥ 2·f_max` (fréquence de Nyquist).

Sinon → **aliasing** : les hautes fréquences sont reconstruites comme des basses fréquences fantômes.

En pratique : pour de l'audio CD, `f_s = 44.1 kHz` (Nyquist = 22.05 kHz, juste au-dessus de l'audible humain).
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

fs = 1000   # Hz
T = 1.0     # 1 seconde
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Signal : 50 Hz + 120 Hz + bruit
sig = (np.sin(2 * np.pi * 50 * t)
       + 0.5 * np.sin(2 * np.pi * 120 * t)
       + 0.2 * np.random.randn(len(t)))

plt.figure(figsize=(12, 3))
plt.plot(t[:300], sig[:300])
plt.xlabel("Temps (s)"); plt.ylabel("Amplitude")
plt.title("Signal échantillonné (300 premières mesures)")
plt.tight_layout()
```

<!-- #region -->
## 2. Transformée de Fourier (FFT)
<!-- #endregion -->

<!-- #region -->
La **FFT** décompose un signal en somme de sinusoïdes. Elle révèle les **fréquences** présentes.

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi j k n / N}
$$

**Lecture du spectre** :

- `|X[k]|` : amplitude de la fréquence `k * fs / N`.
- `arg(X[k])` : phase.

Pour un signal réel, on n'utilise que la moitié du spectre (l'autre est conjuguée).
<!-- #endregion -->

```python
from scipy.fft import rfft, rfftfreq

X = rfft(sig)
freqs = rfftfreq(len(sig), 1 / fs)

plt.figure(figsize=(10, 3))
plt.plot(freqs, np.abs(X))
plt.xlabel("Fréquence (Hz)"); plt.ylabel("|X(f)|")
plt.title("Spectre — 2 pics nets attendus à 50 Hz et 120 Hz")
plt.xlim(0, 250)
plt.tight_layout()
```

<!-- #region -->
## 3. STFT — spectrogramme (temps-fréquence)
<!-- #endregion -->

<!-- #region -->
La FFT donne le **contenu fréquentiel global**. Si le signal **change dans le temps** (parole, musique), on veut savoir *quelle fréquence à quel moment*.

**STFT** = FFT sur des **fenêtres glissantes**. Le résultat est un **spectrogramme** : `(time, frequency, amplitude)`.

**Trade-off Heisenberg** : on ne peut pas avoir simultanément bonne résolution temporelle ET fréquentielle. Fenêtre courte = bonne résolution temps, mauvaise fréquence. Inverse aussi.
<!-- #endregion -->

```python
from scipy.signal import spectrogram

# Signal modulé : 50 Hz puis 120 Hz à mi-parcours
sig_mod = np.concatenate([
    np.sin(2*np.pi*50*t[:fs//2]),
    np.sin(2*np.pi*120*t[fs//2:]),
])

f, time_bins, Sxx = spectrogram(sig_mod, fs=fs, nperseg=128, noverlap=64)

plt.figure(figsize=(10, 4))
plt.pcolormesh(time_bins, f, 10 * np.log10(Sxx + 1e-10), shading="auto", cmap="viridis")
plt.ylim(0, 200)
plt.ylabel("Fréquence (Hz)"); plt.xlabel("Temps (s)")
plt.title("Spectrogramme STFT — transition 50→120 Hz")
plt.colorbar(label="dB")
plt.tight_layout()
```

<!-- #region -->
## 4. Ondelettes (Wavelets)
<!-- #endregion -->

<!-- #region -->
Les **ondelettes** analysent un signal à **plusieurs échelles** simultanément, contrairement à la STFT qui fixe une fenêtre.

- **Onde-mère** (Morlet, Daubechies, Haar, Mexican Hat) : prototype localisé en temps et fréquence.
- **Transformée continue (CWT)** : pour analyse temps-échelle visuelle (scalogramme).
- **Transformée discrète (DWT)** : pour compression et débruitage (pyramide d'approximations + détails).

Application phare : **JPEG 2000** (image), **débruitage** (seuillage des coefficients de détail).

```python
"""
import pywt

# Continuous Wavelet Transform
coefs, freqs = pywt.cwt(sig, scales=np.arange(1, 64), wavelet="morl", sampling_period=1/fs)
plt.pcolormesh(t, freqs, np.abs(coefs))

# Discrete Wavelet Transform — décomposition multi-niveau
coeffs = pywt.wavedec(sig, "db4", level=4)
# coeffs[0] = approximation (basse fréquence), coeffs[1:] = détails (fréquences croissantes)
"""
```
<!-- #endregion -->

<!-- #region -->
## 5. Transformée de Hilbert — enveloppe et phase
<!-- #endregion -->

<!-- #region -->
La **transformée de Hilbert** construit le **signal analytique** `s_a(t) = s(t) + j·H(s)(t)` dont :

- **Module** `|s_a(t)|` = **enveloppe** du signal (amplitude instantanée).
- **Phase** `arg(s_a(t))` = phase instantanée.
- Dérivée de la phase = **fréquence instantanée**.

Très utile pour analyser des signaux modulés (AM/FM), démoduler, extraire l'enveloppe.
<!-- #endregion -->

```python
from scipy.signal import hilbert

# Signal AM : porteuse 100 Hz modulée par 5 Hz
carrier = np.cos(2 * np.pi * 100 * t)
modulation = 1 + 0.5 * np.cos(2 * np.pi * 5 * t)
am = carrier * modulation

analytic = hilbert(am)
envelope = np.abs(analytic)

plt.figure(figsize=(12, 4))
plt.plot(t, am, alpha=0.5, label="Signal AM")
plt.plot(t, envelope, color="red", label="Enveloppe (Hilbert)")
plt.legend(); plt.title("Extraction d'enveloppe par Hilbert")
plt.tight_layout()
```

<!-- #region -->
## 6. Filtrage — Butterworth passe-bas
<!-- #endregion -->

<!-- #region -->
**Filtres FIR / IIR** : élimine certaines fréquences (passe-bas, passe-haut, passe-bande, coupe-bande).

**Butterworth** : amplitude maximalement plate dans la bande passante.
**Chebyshev** : transition plus rapide mais oscillations dans la passante.
**Elliptique** : transition la plus rapide mais oscillations partout.
**FIR (windowed)** : linéaire en phase (utile pour audio).

Toujours appliquer avec **`filtfilt`** (filter forward+backward) pour éviter le **déphasage**.
<!-- #endregion -->

```python
from scipy.signal import butter, filtfilt

# Passe-bas Butterworth d'ordre 4, coupure à 80 Hz
nyq = fs / 2
b, a = butter(N=4, Wn=80 / nyq, btype="lowpass")
filtered = filtfilt(b, a, sig)

# FFT pour comparer
X_orig = np.abs(rfft(sig))
X_filt = np.abs(rfft(filtered))
freqs = rfftfreq(len(sig), 1 / fs)

plt.figure(figsize=(10, 3))
plt.plot(freqs, X_orig, alpha=0.5, label="Original")
plt.plot(freqs, X_filt, label="Après LP 80 Hz")
plt.axvline(80, color="red", linestyle="--", alpha=0.5)
plt.xlim(0, 200); plt.xlabel("Hz"); plt.ylabel("|X(f)|"); plt.legend()
plt.title("Filtre Butterworth passe-bas (80 Hz) — élimine le pic 120 Hz")
plt.tight_layout()
```

<!-- #region -->
## 7. Débruitage
<!-- #endregion -->

```python
from scipy.signal import medfilt, savgol_filter

# Bruit gaussien fort
sig_noisy = sig + 0.5 * np.random.randn(len(sig))

# Moyenne mobile (filtre rectangulaire) — flou
ma = np.convolve(sig_noisy, np.ones(15) / 15, mode="same")

# Filtre médian — robuste aux outliers (peaks)
med = medfilt(sig_noisy, kernel_size=15)

# Savitzky-Golay — préserve mieux les variations rapides (polynôme local)
sg = savgol_filter(sig_noisy, window_length=15, polyorder=3)

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for ax, (label, s) in zip(axes, [
    ("Noisy", sig_noisy), ("Moving Avg", ma), ("Median", med), ("Savitzky-Golay", sg)
]):
    ax.plot(t[:300], s[:300])
    ax.set_ylabel(label); ax.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
## 8. EMD — Empirical Mode Decomposition
<!-- #endregion -->

<!-- #region -->
**EMD** (Huang 1998) décompose un signal en **IMFs** (Intrinsic Mode Functions) — des composantes oscillantes locales. Particulièrement utile pour signaux **non-stationnaires** et **non-linéaires** où FFT/ondelettes peinent.

Dispo via `PyEMD` (`pip install EMD-signal`).

```python
"""
from PyEMD import EMD
emd = EMD()
IMFs = emd.emd(sig)
# IMFs.shape = (N_IMFs, len(sig)) — décomposition complète
"""
```

Application typique : analyse de signaux **bioacoustiques**, **séismologie**, **finance**.
<!-- #endregion -->

<!-- #region -->
## 9. Applications 2026
<!-- #endregion -->

<!-- #region -->
### 9.1 Audio — `librosa`
<!-- #endregion -->

<!-- #region -->
**librosa** est la référence en Python pour le traitement audio :

```python
"""
import librosa
y, sr = librosa.load(librosa.example("trumpet"))

# Spectrogramme Mel (échelle perceptuelle humaine)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S), sr=sr, x_axis="time", y_axis="mel")

# MFCC — features audio classiques pour speech / music ML
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Pitch / tempo / onset detection
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
"""
```
<!-- #endregion -->

<!-- #region -->
### 9.2 Autres librairies
<!-- #endregion -->

<!-- #region -->
- **`scipy.signal`** — base solide (filtres, fenêtres, transformées).
- **`PyWavelets`** — ondelettes.
- **`PyEMD`** — Empirical Mode Decomposition.
- **`obspy`** — sismologie / signaux géophysiques.
- **`mne-python`** — neurosciences (EEG/MEG).
- **`pyAudioAnalysis`** — features + classification audio bas niveau.

### 9.3 ML sur signaux

- **Speech to text** : Whisper (HF), Conformer.
- **Audio classification** : YAMNet, AST (Audio Spectrogram Transformer).
- **Music generation** : MusicGen, AudioGen.
- **Anomalie sur signal capteur** : STUMPY (matrix profile) — voir `Detection_Outliers`.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [scipy.signal docs](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [librosa docs](https://librosa.org/doc/main/index.html)
- [PyWavelets docs](https://pywavelets.readthedocs.io/)
- [Oppenheim — Discrete-Time Signal Processing (livre de référence)](https://www.pearson.com/)
- Notebooks liés : `TS_Time_Series_Overview`, `Detection_Outliers`.
<!-- #endregion -->
