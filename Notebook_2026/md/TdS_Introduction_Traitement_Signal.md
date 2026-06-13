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
# 📡 Introduction au traitement du signal
<!-- #endregion -->

<!-- #region -->
Notebook **tutoriel + wiki + cheat-sheet**. On part de la théorie (Fourier, ondelettes, Hilbert)
pour aller vers la pratique : analyse spectrale, filtrage, analyse temps-fréquence, audio et image.

Un **signal** est une grandeur qui évolue le long d'un axe (le temps, l'espace…). L'idée centrale
du domaine : **changer de point de vue**. Un signal vu dans le **domaine temporel** (amplitude vs
temps) peut être ré-exprimé dans le **domaine fréquentiel** (puissance vs fréquence) — et beaucoup
de problèmes difficiles dans l'un deviennent simples dans l'autre.

**Plan :**
1. Setup et charte graphique.
2. Contexte théorique : Fourier, ondelettes, Hilbert.
3. Échantillonnage : Nyquist-Shannon et aliasing.
4. DFT / FFT avec NumPy (partie réelle/imaginaire, `fftfreq`, `fftshift`).
5. DFT naïve $O(N^2)$ vs FFT $O(N\log N)$ (Cooley-Tukey).
6. Approximation de la transformée de Fourier continue.
7. Analyse spectrale d'un signal (détrend, recherche de périodicités).
8. Fuites spectrales, fenêtrage, densité spectrale (Welch).
9. STFT / spectrogramme.
10. Filtrage : seuillage FFT et filtres numériques (Butterworth).
11. Transformée de Hilbert : enveloppe d'amplitude.
12. Ondelettes : débruitage (DWT) et analyse temps-échelle (CWT).
13. Applications audio (librosa) : spectrogramme, mel, MFCC, chroma.
14. Application image : FFT 2D.
15. GAN pour la génération de signaux (PyTorch).
16. Récapitulatif et sources.

> Notebooks frères : `TS_Time_Series_Intro` (séries temporelles), `TS_ARIMA`, `DL_PyTorch`.
<!-- #endregion -->

<!-- #region -->
## 1. Setup
<!-- #endregion -->

<!-- #region -->
On importe la pile scientifique du traitement du signal : **NumPy** (FFT), **`scipy.signal`**
(filtres, fenêtres, Hilbert, STFT), **`scipy.fft`** (transformées), **PyWavelets** (`pywt`,
ondelettes) et **librosa** (audio). On fixe les graines pour la reproductibilité et on définit une
petite charte de couleurs (un signal est une série continue neutre → couleur primaire unique).
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.fft import fft, ifft, rfft, rfftfreq, fftfreq, fftshift, ifftshift
import pywt
import random


def set_seed(seed: int = 42) -> None:
    """Fixe les graines (numpy, random) pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

# Palette CHART — un signal = série continue neutre → couleur primaire unique.
PRIMARY = "#00798c"
MAUVAIS = "#d1495b"
MOYEN = "#edae49"
ACCENT = "#66a182"
ACCENT_DARK = "#2e4057"
LAVENDER = "#9d83b8"
DUSTY = "#b8848e"
BEIGE = "#c9b78b"

plt.rcParams.update({"figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3})
print("Setup OK — numpy", np.__version__, "| pywt", pywt.__version__)
```

<!-- #region -->
## 2. Contexte théorique
<!-- #endregion -->

<!-- #region -->
### 2.1 Transformée de Fourier
<!-- #endregion -->

<!-- #region -->
Dans notre esprit, un signal 1D est une **série temporelle** : un axe $x$ = le temps, un axe $y$ =
la quantité mesurée (tension, température, pression acoustique…). Faire une **transformée de
Fourier** d'un signal revient à le **voir dans un autre domaine** : on passe du domaine temporel au
domaine fréquentiel.

![Du domaine temporel au domaine fréquentiel : un sinus pur devient un pic unique](images/tds_fourier_domains.png)

Sur la partie gauche, un seul sinus de fréquence 1. À droite, on ne voit **un pic que là où la
fréquence vaut 1** et zéro partout ailleurs. Si plusieurs sinus/cosinus interagissent, on verra
plusieurs pics : un par mode présent dans le signal.

**Formellement.** La transformée de Fourier d'un signal continu $x(t)$ est :

$$\hat{x}(f) = \int_{-\infty}^{+\infty} x(t)\, e^{-2\pi i f t}\, dt$$

Pour un signal **échantillonné** $x[n]$ ($n = 0,\dots,N-1$), on utilise la **transformée de Fourier
discrète (DFT)** :

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-2\pi i \frac{kn}{N}}, \qquad k = 0,\dots,N-1$$

À quoi ça sert ?
- **Filtrer** : un signal a en général une bande de fréquence limitée ; ce qui en sort est souvent
  du bruit (par ex. du bruit blanc, qui « vit » à toutes les fréquences) → on le supprime.
- **Analyser la périodicité** : les pics du spectre révèlent les composantes saisonnières/cycliques.
<!-- #endregion -->

<!-- #region -->
### 2.2 Transformée en ondelettes
<!-- #endregion -->

<!-- #region -->
La transformée de Fourier projette le signal sur des **sinus/cosinus** — des fonctions qui
s'étendent sur **tout** l'axe du temps. Conséquence : elle dit *quelles* fréquences sont présentes,
mais **pas quand**. Pour un signal dont le contenu fréquentiel change au cours du temps (parole,
musique, signaux transitoires), c'est une limite.

Les **ondelettes** projettent au contraire le signal sur une fonction localisée — l'**ondelette
mère** $\psi$ — que l'on **contracte/dilate** (échelle $a$) et **translate** (position $b$) :

$$W(a,b) = \frac{1}{\sqrt{a}} \int x(t)\, \psi^{*}\!\left(\frac{t-b}{a}\right) dt$$

Une petite échelle capte les **hautes fréquences** (détails brefs), une grande échelle les **basses
fréquences** (tendance). Ci-dessous l'ondelette **Daubechies db2** :

![Forme de l'ondelette Daubechies db2](images/tds_wavelet_db2.png)

En jouant avec cette « petite fonction » — en la contractant et l'étendant — on analyse les
projections sur l'ensemble du signal. La projection donne un signal « filtré » et le résidu est la
différence avec l'original : c'est une façon **plus locale et plus générale** de filtrer le bruit
que Fourier, comme on le verra en pratique.
<!-- #endregion -->

<!-- #region -->
### 2.3 Transformée de Hilbert
<!-- #endregion -->

<!-- #region -->
Parfois on a un signal plein de hauts et de bas mais on ne veut que son **enveloppe** (son
amplitude instantanée). Mathématiquement, la transformée de Hilbert $\mathcal{H}$ est une
**convolution** par le noyau $\frac{1}{\pi t}$. À partir d'elle on construit le **signal
analytique** $x_a(t) = x(t) + i\,\mathcal{H}\{x\}(t)$, dont :

- le **module** $|x_a(t)|$ donne l'**enveloppe d'amplitude** ;
- la **dérivée de la phase** $\frac{1}{2\pi}\frac{d}{dt}\arg x_a(t)$ donne la **fréquence
  instantanée**.

On en verra la puissance dans la partie pratique (§11).
<!-- #endregion -->

<!-- #region -->
## 3. Échantillonnage : Nyquist-Shannon et aliasing
<!-- #endregion -->

<!-- #region -->
Avant toute FFT, une question fondatrice : à quelle fréquence faut-il échantillonner un signal
continu pour ne rien perdre ? Le **théorème de Nyquist-Shannon** répond :

$$f_s > 2\, f_{\max}$$

La fréquence d'échantillonnage $f_s$ doit dépasser **deux fois** la plus haute fréquence $f_{\max}$
du signal. En dessous, des fréquences hautes se **replient** (aliasing) et se font passer pour des
fréquences basses — impossible de les distinguer après coup. Démonstration : un sinus à 9 Hz
échantillonné à seulement 10 Hz apparaît comme un sinus à 1 Hz.
<!-- #endregion -->

```python
def demo_aliasing() -> plt.Figure:
    """Illustre l'aliasing : une sinusoïde de 9 Hz échantillonnée à 10 Hz se replie en 1 Hz."""
    f_signal = 9.0          # fréquence vraie (Hz)
    fs_low = 10.0           # fréquence d'échantillonnage trop basse (< 2*f_signal)
    t_cont = np.linspace(0, 1, 2000)               # quasi-continu
    t_samp = np.arange(0, 1, 1 / fs_low)           # échantillons

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_cont, np.sin(2 * np.pi * f_signal * t_cont),
            color=PRIMARY, lw=1, label="signal 9 Hz (vrai)")
    ax.plot(t_cont, np.sin(2 * np.pi * 1.0 * t_cont),
            color=MOYEN, ls="--", lw=2, label="alias 1 Hz perçu")
    ax.plot(t_samp, np.sin(2 * np.pi * f_signal * t_samp),
            "o", color=MAUVAIS, ms=8, label=f"échantillons à fs={fs_low:.0f} Hz")
    ax.set(xlabel="temps (s)", ylabel="amplitude",
           title="Aliasing : sous-échantillonnage (fs=10 Hz < 2·9 Hz)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


demo_aliasing();
```

<!-- #region -->
Les échantillons (points rouges) tombent exactement sur la sinusoïde lente : à partir d'eux,
impossible de deviner qu'il s'agissait d'un 9 Hz. D'où l'usage systématique d'un **filtre
anti-repliement** (passe-bas) avant numérisation.
<!-- #endregion -->

<!-- #region -->
## 4. DFT et FFT avec NumPy
<!-- #endregion -->

<!-- #region -->
NumPy implémente la DFT selon la convention :

![Formule de la DFT directe](images/tds_dft_formula.png)

$$A_k = \sum_{m=0}^{n-1} a_m\, e^{-2\pi i \frac{mk}{n}}, \qquad k = 0,\dots,n-1$$

et la **DFT inverse** :

![Formule de la DFT inverse](images/tds_idft_formula.png)

$$a_m = \frac{1}{n}\sum_{k=0}^{n-1} A_k\, e^{+2\pi i \frac{mk}{n}}, \qquad m = 0,\dots,n-1$$

Elle diffère de la transformée directe par le **signe** de l'exponentielle et la **normalisation**
$1/n$. Commençons par visualiser la partie réelle et imaginaire de la DFT d'une **impulsion**
$a[1]=1$.
<!-- #endregion -->

```python
def fft_real_imag(a: np.ndarray, title: str) -> plt.Figure:
    """Affiche un signal et les parties réelle/imaginaire de sa DFT (périodisées)."""
    A = np.fft.fft(a)
    B = np.append(A, A[0])  # on referme pour la périodicité
    fig, axes = plt.subplots(3, 1, figsize=(9, 6))
    axes[0].plot(np.append(a, a[0]), color=PRIMARY)
    axes[0].set_ylabel("signal $a$")
    axes[1].plot(np.real(B), color=ACCENT)
    axes[1].set_ylabel("partie réelle")
    axes[2].plot(np.imag(B), color=MAUVAIS)
    axes[2].set_ylabel("partie imaginaire")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


n = 20
delta = np.zeros(n)
delta[1] = 1.0
fft_real_imag(delta, "DFT d'une impulsion $a[1]=1$");
```

<!-- #region -->
La DFT d'une impulsion décalée est une **exponentielle complexe** : parties réelle (cosinus) et
imaginaire (sinus) oscillent. On peut aussi encoder le **module** et la **phase** sur un seul
graphe, en colorant la courbe d'amplitude par la phase (roue HSV).
<!-- #endregion -->

```python
def fft_color(a: np.ndarray) -> plt.Figure:
    """Trace |DFT| avec la phase encodée en couleur (roue HSV)."""
    A = np.fft.fft(a)
    k = np.arange(len(a))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    axes[0].plot(np.append(a, a[0]), color=PRIMARY)
    axes[0].set_title("signal")

    x = np.append(k, k[-1] + k[1] - k[0])
    z = np.append(A, A[0])
    X = np.array([x, x])
    y = np.abs(z)
    Y = np.array([np.zeros(len(x)), y])
    C = np.angle(np.array([z, z]))
    axes[1].plot(x, y, "k", lw=0.8)
    pcm = axes[1].pcolormesh(X, Y, C, shading="gouraud", cmap="hsv",
                             vmin=-np.pi, vmax=np.pi)
    fig.colorbar(pcm, ax=axes[1], label="phase (rad)")
    axes[1].set_title("|DFT| colorée par la phase")
    fig.tight_layout()
    return fig


fft_color(delta);
```

<!-- #region -->
Comparons maintenant un **cosinus** et un **sinus** purs de même fréquence. Un cosinus est une
fonction **paire** → sa DFT est **réelle** ; un sinus est **impair** → sa DFT est **imaginaire pure**.
<!-- #endregion -->

```python
m = np.arange(n)
cos_sig = np.cos(m * 2 * np.pi / n)
sin_sig = np.sin(m * 2 * np.pi / n)
fft_real_imag(cos_sig, "DFT d'un cosinus pur (réel → pics réels)");
```

```python
fft_real_imag(sin_sig, "DFT d'un sinus pur (imaginaire pur)");
```

<!-- #region -->
Pour relier les **indices** de la DFT à de vraies **fréquences**, NumPy fournit `np.fft.fftfreq`,
qui renvoie les fréquences (en cycles par pas de temps) associées à chaque bin :

- $\text{freq} = [0, 1, \dots, n/2-1, -n/2, \dots, -1] / (d\cdot n)$ si $n$ est pair,
- $\text{freq} = [0, 1, \dots, (n-1)/2, -(n-1)/2, \dots, -1] / (d\cdot n)$ si $n$ est impair,

où $d$ est le pas d'échantillonnage. Exemple sur un signal à deux composantes.
<!-- #endregion -->

```python
def demo_fftfreq() -> plt.Figure:
    """Signal 2 cosinus + sinus ; affiche le signal et sa DFT (réel/imag) vs fréquence."""
    dt, T1, T2 = 0.1, 2.0, 5.0
    t = np.arange(0, T1 * T2, dt)
    s = 2 * np.cos(2 * np.pi / T1 * t) + np.sin(2 * np.pi / T2 * t)
    fourier = np.fft.fft(s)
    freq = np.fft.fftfreq(s.size, d=dt)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(t, s, color=PRIMARY)
    axes[0].set(xlabel="temps (s)", title="signal $2\\cos(2\\pi t/2)+\\sin(2\\pi t/5)$")
    axes[1].plot(freq, fourier.real, color=ACCENT, label="réel")
    axes[1].plot(freq, fourier.imag, color=MAUVAIS, label="imaginaire")
    axes[1].set(xlabel="fréquence (Hz)", title="DFT via fftfreq")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


demo_fftfreq();
```

<!-- #region -->
Enfin, `fftshift` réordonne le spectre pour centrer la fréquence nulle (pratique pour l'affichage),
et `ifftshift` annule l'opération.
<!-- #endregion -->

```python
n8 = 8
freq8 = np.fft.fftfreq(n8, d=0.1)
print("freq    :", np.round(freq8, 3))
print("fftshift:", np.round(np.fft.fftshift(freq8), 3))
print("ifftshift:", np.round(np.fft.ifftshift(np.fft.fftshift(freq8)), 3))
```

<!-- #region -->
## 5. DFT naïve $O(N^2)$ vs FFT $O(N\log N)$
<!-- #endregion -->

<!-- #region -->
La DFT s'écrit comme un **produit matriciel** $X = M x$ avec $M_{kn} = e^{-2\pi i kn/N}$ : coût
$O(N^2)$. La **FFT** (Cooley-Tukey) exploite la symétrie en séparant indices pairs/impairs :

$$X_k = E_k + e^{-2\pi i k/N} O_k$$

où $E$ et $O$ sont les DFT des sous-séquences paire et impaire. La récurrence
$T(N) = 2\,T(N/2) + O(N)$ donne un coût $O(N\log N)$ — la transformation qui a rendu le traitement
du signal numérique praticable. Vérifions d'abord que les deux implémentations coïncident avec
`np.fft.fft`.
<!-- #endregion -->

```python
def dft_naive(x: np.ndarray) -> np.ndarray:
    """DFT naïve par produit matriciel (complexité O(N^2))."""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    nn = np.arange(N)
    k = nn.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * nn / N)
    return M @ x


def fft_recursive(x: np.ndarray) -> np.ndarray:
    """FFT récursive de Cooley-Tukey (l'entrée doit avoir une longueur puissance de 2)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 1:
        return x
    X_even = fft_recursive(x[::2])
    X_odd = fft_recursive(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([X_even + factor[: N // 2] * X_odd,
                           X_even + factor[N // 2:] * X_odd])


x_test = np.random.random(1024)
err_naive = np.abs(dft_naive(x_test) - np.fft.fft(x_test)).max()
err_rec = np.abs(fft_recursive(x_test) - np.fft.fft(x_test)).max()
print(f"erreur DFT naïve vs np.fft : {err_naive:.2e}")
print(f"erreur FFT récursive vs np.fft : {err_rec:.2e}")
```

<!-- #region -->
Les deux reproduisent `np.fft.fft` à la précision machine près. Mesurons maintenant le **temps de
calcul** selon la taille $N$ : la DFT naïve explose en $O(N^2)$, la FFT reste quasi-linéaire.
<!-- #endregion -->

```python
def bench_dft() -> plt.Figure:
    """Chronomètre DFT naïve / FFT récursive / np.fft.fft selon la taille."""
    import timeit
    sizes = [2 ** k for k in range(4, 10)]
    t_naive, t_rec, t_np = [], [], []
    for N in sizes:
        xs = np.random.random(N)
        t_naive.append(timeit.timeit(lambda: dft_naive(xs), number=3) / 3)
        t_rec.append(timeit.timeit(lambda: fft_recursive(xs), number=3) / 3)
        t_np.append(timeit.timeit(lambda: np.fft.fft(xs), number=50) / 50)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog(sizes, t_naive, "o-", color=MAUVAIS, label="DFT naïve O(N²)")
    ax.loglog(sizes, t_rec, "s-", color=MOYEN, label="FFT récursive O(N log N)")
    ax.loglog(sizes, t_np, "^-", color=PRIMARY, label="np.fft.fft (optimisée)")
    ax.set(xlabel="taille N", ylabel="temps (s)", title="DFT naïve vs FFT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


bench_dft();
```

<!-- #region -->
## 6. Approximation de la transformée de Fourier continue
<!-- #endregion -->

<!-- #region -->
Pour calculer la TF **continue** d'une fonction sur ordinateur, on doit la **discrétiser** et la
**tronquer** (durée finie). En approchant l'intégrale par une somme de rectangles de largeur
$\Delta t$ sur $n$ points :

![Approximation par discrétisation](images/tds_fft_approx_1.png)

![Somme de rectangles sur n points](images/tds_fft_approx_2.png)

![Choix des fréquences discrètes](images/tds_fft_approx_3.png)

![Lien final avec la FFT](images/tds_fft_approx_4.png)

$$X(f_k) \approx \Delta t \sum_{m=0}^{n-1} x(t_m)\, e^{-2\pi i f_k t_m}
        = \Delta t \sum_{m=0}^{n-1} x(t_m)\, e^{-2\pi i \frac{mk}{n}}
        \approx \Delta t \cdot \mathrm{fft}(x)$$

La TF continue est donc approchée par $\Delta t \cdot \mathrm{fft}(x)$ (avec les bons `fftshift`).
Vérifions sur une **gaussienne**, dont la TF est connue analytiquement :
$\hat{x}(f) = \sqrt{\pi/\alpha}\, e^{-(\pi f)^2/\alpha}$.
<!-- #endregion -->

```python
def fft_gaussian_approx(shift: float = 0.0, colored: bool = False) -> plt.Figure:
    """Compare la FFT d'une gaussienne à sa TF analytique exacte.

    Args:
        shift: décalage temporel du centre de la gaussienne.
        colored: si True, affiche la phase en couleur au lieu des sous-graphes réel/imag.
    """
    alpha, nc, dt = 10.0, 40, 0.1
    tmax, tmin = (nc - 1) * dt, -nc * dt
    t = np.linspace(tmin, tmax, 2 * nc)
    x = np.exp(-alpha * (t - shift) ** 2)

    a = np.fft.ifftshift(x)
    A = np.fft.fft(a)
    X = dt * np.fft.fftshift(A)
    freq = np.fft.fftshift(np.fft.fftfreq(t.size, d=dt))

    if not colored:
        fig, axes = plt.subplots(4, 1, figsize=(9, 7))
        axes[0].plot(t, x, color=PRIMARY); axes[0].set_title("gaussienne $e^{-\\alpha t^2}$")
        axes[1].plot(a, color=BEIGE); axes[1].set_title("après ifftshift")
        axes[2].plot(freq, np.real(X), color=ACCENT, label="fft")
        axes[2].plot(freq, np.sqrt(np.pi / alpha) * np.exp(-(np.pi * freq) ** 2 / alpha),
                     "--", color=MAUVAIS, label="exact")
        axes[2].legend(fontsize=8); axes[2].set_title("partie réelle : fft vs exact")
        axes[3].plot(freq, np.imag(X), color=MOYEN); axes[3].set_title("partie imaginaire")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    axes[0].plot(t, x, color=PRIMARY); axes[0].set_title("gaussienne décalée")
    xf = np.append(freq, freq[0]); z = np.append(X, X[0])
    Xm = np.array([xf, xf]); ym = np.abs(z)
    Ym = np.array([np.zeros(len(xf)), ym]); C = np.angle(np.array([z, z]))
    axes[1].plot(xf, ym, "k", lw=0.8)
    pcm = axes[1].pcolormesh(Xm, Ym, C, shading="gouraud", cmap="hsv", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(pcm, ax=axes[1], label="phase")
    axes[1].set_title("|TF| colorée par la phase")
    fig.tight_layout()
    return fig


fft_gaussian_approx();
```

<!-- #region -->
La partie réelle de la FFT (× $\Delta t$) **se superpose** à la solution exacte, et la partie
imaginaire est nulle à la précision machine (gaussienne centrée → fonction paire). En la
**décalant** dans le temps, le module reste inchangé mais la **phase** tourne — d'où l'intérêt de la
visualisation colorée.

![Visualisation en couleur de la transformée de Fourier](images/tds_fft_color_viz.png)
<!-- #endregion -->

```python
fft_gaussian_approx(colored=True);
```

```python
fft_gaussian_approx(shift=1.0, colored=True);
```

<!-- #region -->
## 7. Analyse spectrale d'un signal
<!-- #endregion -->

<!-- #region -->
Passons à un cas concret. On construit un **signal horaire synthétique** de type « consommation
d'énergie » : une légère tendance, un **cycle journalier** (période 24 h), son **harmonique** (12 h),
un **cycle hebdomadaire** (168 h), et du bruit gaussien. Comme les fréquences sont **connues**, on
pourra vérifier que la FFT les retrouve.

> L'original utilisait un fichier Google-Drive (`AEP_hourly.csv`) aujourd'hui perdu ; on le remplace
> par un signal synthétique déterministe (seed = 42), reproductible et pédagogiquement transparent.
<!-- #endregion -->

```python
def make_energy_signal(n_days: int = 84, seed: int = 42):
    """Signal horaire type 'consommation' : tendance + cycles jour/12h/semaine + bruit.

    Returns:
        (idx, clean, noisy, fs_hours) — index horaire, signal propre, bruité, fs=1/h.
    """
    rng = np.random.default_rng(seed)
    n = 24 * n_days
    t = np.arange(n, dtype=float)
    trend = 0.002 * t                                  # légère dérive
    daily = 3.0 * np.sin(2 * np.pi * t / 24)           # cycle journalier (période 24 h)
    halfday = 1.2 * np.sin(2 * np.pi * t / 12 + 0.5)   # harmonique 12 h
    weekly = 2.0 * np.sin(2 * np.pi * t / 168)         # cycle hebdo (période 168 h)
    clean = 10.0 + trend + daily + halfday + weekly
    noisy = clean + rng.normal(0, 1.0, n)
    return np.arange(n), clean, noisy, 1.0


idx, clean_sig, energy, fs_h = make_energy_signal()
date_array = pd.date_range("2024-01-01", periods=len(idx), freq="h")


def plot_energy() -> plt.Figure:
    """Vue d'ensemble du signal synthétique de consommation."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_array, energy, color=PRIMARY, lw=0.7)
    ax.set(xlabel="date", ylabel="consommation (u.a.)",
           title="Signal horaire synthétique (cycles 24 h / 12 h / 168 h + bruit)")
    fig.tight_layout()
    return fig


plot_energy();
```

<!-- #region -->
### 7.1 Détrend
<!-- #endregion -->

<!-- #region -->
Le signal a une **tendance** et une **moyenne non nulle**, qui perturbent l'analyse fréquentielle
(elles créent une grosse composante à fréquence ~0 qui écrase le reste). On les retire avec
`scipy.signal.detrend` (régression linéaire soustraite).
<!-- #endregion -->

```python
y_detrend = sig.detrend(energy)


def plot_detrend() -> plt.Figure:
    """Compare le signal brut et le signal détendancé (detrend linéaire)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_array, energy, color=ACCENT_DARK, alpha=0.6, label="brut")
    ax.plot(date_array, y_detrend, color=MAUVAIS, label="détendancé")
    ax.set(xlabel="date", ylabel="amplitude", title="signal.detrend")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


plot_detrend();
```

<!-- #region -->
### 7.2 Analyse de fréquence
<!-- #endregion -->

<!-- #region -->
On calcule la FFT du signal détendancé et on trace le **spectre d'amplitude en fonction de la
période** (en heures, = inverse de la fréquence). Les pics doivent tomber **exactement** sur les
périodes injectées : 24 h, 12 h, 168 h.
<!-- #endregion -->

```python
FFT = np.fft.fft(y_detrend)
new_N = len(FFT) // 2
new_X = np.linspace(1e-12, fs_h / 2, new_N, endpoint=True)   # fréquence (cycles/h)
new_Xph = 1.0 / new_X                                        # période (h)
FFT_abs = np.abs(FFT)
fft_amp = 2 * FFT_abs[:new_N] / new_N                        # amplitude unilatérale


def plot_spectrum() -> plt.Figure:
    """Spectre d'amplitude en fonction de la période (h) — pics aux cycles injectés."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(new_Xph, fft_amp, color="black", lw=0.8)
    ax.set(xlabel="période (h)", ylabel="amplitude",
           title="Spectre de Fourier (méthode FFT)", xlim=(0, 200))
    for p in (12, 24, 168):
        ax.axvline(p, color=MOYEN, ls="--", alpha=0.6)
    fig.tight_layout()
    return fig


plot_spectrum();
```

<!-- #region -->
On peut **trier** les pics pour récupérer les périodes dominantes, exprimées en jours
(période / 24).
<!-- #endregion -->

```python
amp_df = pd.DataFrame({"Amplitude": fft_amp})
top_peaks = amp_df.sort_values("Amplitude", ascending=False).head(10)
print("Top 10 pics (index = bin de fréquence) :")
print(top_peaks)
print("\nPériodes correspondantes (h) :", np.round(new_Xph[top_peaks.index.values], 1))
```

<!-- #region -->
## 8. Fuites spectrales, fenêtrage et densité spectrale
<!-- #endregion -->

<!-- #region -->
La DFT suppose le signal **périodique** sur la fenêtre observée. Si une composante ne fait pas un
nombre **entier** de périodes dans la fenêtre, son énergie « **fuit** » sur les bins voisins (*spectral
leakage*) : un pic net devient étalé. Le remède est le **fenêtrage** : on multiplie le signal par une
fenêtre qui s'annule en douceur aux bords (Hann, Hamming, Blackman…), ce qui réduit drastiquement
les lobes secondaires.
<!-- #endregion -->

```python
def demo_windowing() -> plt.Figure:
    """Montre la réduction des fuites spectrales par fenêtrage de Hann."""
    fs, T = 200.0, 1.0
    t = np.arange(0, T, 1 / fs)
    # fréquence non entière en nombre de périodes -> fuite spectrale
    s = np.sin(2 * np.pi * 34.7 * t)
    win = sig.windows.hann(len(t))
    f = rfftfreq(len(t), 1 / fs)
    sp_rect = np.abs(rfft(s))
    sp_hann = np.abs(rfft(s * win))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, sp_rect + 1e-6, color=MAUVAIS, label="rectangulaire (fuite)")
    ax.semilogy(f, sp_hann + 1e-6, color=PRIMARY, label="fenêtre de Hann")
    ax.set(xlabel="fréquence (Hz)", ylabel="|X(f)| (log)",
           title="Fuite spectrale et fenêtrage", xlim=(0, 100))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


demo_windowing();
```

<!-- #region -->
Quand on s'intéresse à la **répartition de puissance** plutôt qu'aux amplitudes exactes, la méthode
de **Welch** est le standard : elle découpe le signal en segments fenêtrés se recouvrant, calcule un
périodogramme par segment et les **moyenne** → estimation de la **densité spectrale de puissance**
(PSD) bien moins bruitée qu'une FFT brute.
<!-- #endregion -->

```python
def demo_welch() -> plt.Figure:
    """Densité spectrale de puissance par la méthode de Welch (périodogramme moyenné)."""
    f, pxx = sig.welch(energy, fs=fs_h, nperseg=512)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(1 / np.maximum(f, 1e-9), pxx, color=PRIMARY)
    ax.set(xlabel="période (h)", ylabel="PSD", title="Welch PSD", xlim=(0, 200))
    for p in (12, 24, 168):
        ax.axvline(p, color=MOYEN, ls="--", alpha=0.6)
    fig.tight_layout()
    return fig


demo_welch();
```

<!-- #region -->
## 9. STFT / spectrogramme
<!-- #endregion -->

<!-- #region -->
La FFT donne le contenu fréquentiel **global** : elle ne dit pas *quand* chaque fréquence apparaît.
Pour un signal **non stationnaire** (la fréquence change au cours du temps), on utilise la
**transformée de Fourier à court terme** (STFT) : on glisse une fenêtre sur le signal et on calcule
une FFT locale à chaque position. Le résultat est un **spectrogramme** (temps × fréquence ×
puissance). Compromis incontournable (principe d'incertitude) : fenêtre courte → bonne résolution
**temporelle** mais mauvaise résolution **fréquentielle**, et inversement. Démonstration sur un
**chirp** (fréquence croissante de 10 à 200 Hz).
<!-- #endregion -->

```python
def demo_spectrogram() -> plt.Figure:
    """Spectrogramme d'un chirp (fréquence croissante) — analyse temps-fréquence."""
    fs = 1000.0
    t = np.arange(0, 2, 1 / fs)
    s = sig.chirp(t, f0=10, t1=2, f1=200, method="linear")
    f, tt, Sxx = sig.spectrogram(s, fs=fs, nperseg=128, noverlap=100)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, s, color=PRIMARY, lw=0.5)
    axes[0].set(title="chirp 10→200 Hz", xlabel="temps (s)")
    pcm = axes[1].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    fig.colorbar(pcm, ax=axes[1], label="dB")
    axes[1].set(xlabel="temps (s)", ylabel="fréquence (Hz)", title="spectrogramme (STFT)")
    fig.tight_layout()
    return fig


demo_spectrogram();
```

<!-- #region -->
## 10. Filtrage du bruit
<!-- #endregion -->

<!-- #region -->
### 10.1 Filtrage par seuillage FFT
<!-- #endregion -->

<!-- #region -->
Première approche, directe : on passe au domaine fréquentiel, on **met à zéro** les composantes dont
l'amplitude est sous un **seuil**, puis on revient au temporel par FFT inverse. On prend le pic le
plus élevé comme référence et on filtre tout ce qui est sous une fraction de ce pic.

⚠️ **Compromis** : seuil trop haut → on supprime aussi des composantes utiles ; seuil trop bas → on
garde le bruit. Visualisons d'abord l'effet sur le **spectre** à différents seuils absolus.
<!-- #endregion -->

```python
def fft_filter_amp(th: float) -> np.ndarray:
    """Spectre d'amplitude unilatéral après mise à zéro des composantes < th."""
    a = 2 * np.abs(FFT) / new_N
    a[a <= th] = 0
    return a[:new_N]


def fft_filter_perc(perc: float) -> np.ndarray:
    """Reconstruit le signal en gardant les pics > perc·(amplitude max)."""
    th = perc * fft_amp.max()
    fft_tof = FFT.copy()
    a = 2 * np.abs(fft_tof) / new_N
    fft_tof[a <= th] = 0
    return np.real(np.fft.ifft(fft_tof))


def plot_fft_threshold_spectrum() -> plt.Figure:
    """Spectre original vs filtré pour plusieurs seuils d'amplitude absolus."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, k in zip(axes.ravel(), [0.05, 0.2, 0.5, 1.0]):
        ax.plot(new_Xph, fft_amp, color=ACCENT_DARK, alpha=0.5, label="original")
        ax.plot(new_Xph, fft_filter_amp(k), color=MAUVAIS, label="filtré")
        ax.set(title=f"seuil={k}", xlim=(0, 200), xlabel="période (h)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


plot_fft_threshold_spectrum();
```

<!-- #region -->
Et l'effet sur le **signal reconstruit** : plus on monte le seuil, plus le signal devient « épuré »
(ne restent que les cycles dominants), au prix de la perte de structure fine.
<!-- #endregion -->

```python
def plot_fft_threshold_signal() -> plt.Figure:
    """Signal reconstruit pour différents pourcentages du pic max."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 7))
    colors = [ACCENT_DARK, PRIMARY, ACCENT, MAUVAIS]
    for ax, k, c in zip(axes, [0.0, 0.30, 0.60, 0.95], colors):
        ax.plot(idx, fft_filter_perc(k), color=c)
        ax.set(title=f"k={k:.2f} du maximum", ylabel="charge")
    axes[-1].set_xlabel("temps (échantillon)")
    fig.tight_layout()
    return fig


plot_fft_threshold_signal();
```

<!-- #region -->
### 10.2 Filtres numériques (Butterworth)
<!-- #endregion -->

<!-- #region -->
Le seuillage FFT brut est pédagogique mais grossier (il crée des artefacts de Gibbs et ne contrôle
pas la transition). En pratique on utilise des **filtres numériques** conçus proprement. Le filtre
de **Butterworth** est le plus courant : sa réponse en amplitude est **maximalement plate** dans la
bande passante. On l'applique avec `filtfilt` (filtrage **aller-retour** → déphasage nul). On peut
le déclarer **passe-bas**, **passe-haut** ou **passe-bande**.
<!-- #endregion -->

```python
def butter_filter(x: np.ndarray, cutoff, fs: float, btype: str, order: int = 4) -> np.ndarray:
    """Filtre Butterworth sans déphasage (filtfilt). cutoff en Hz (scalaire ou paire)."""
    nyq = fs / 2
    wn = np.atleast_1d(cutoff) / nyq
    b, a = sig.butter(order, wn if wn.size > 1 else wn[0], btype=btype)
    return sig.filtfilt(b, a, x)


def demo_butter() -> plt.Figure:
    """Filtre passe-bas / passe-haut / passe-bande sur un signal multi-tons bruité."""
    fs = 500.0
    t = np.arange(0, 1, 1 / fs)
    s = (np.sin(2 * np.pi * 5 * t) + 0.6 * np.sin(2 * np.pi * 40 * t)
         + 0.4 * np.sin(2 * np.pi * 120 * t))
    s_noisy = s + 0.3 * np.random.default_rng(0).normal(size=len(t))

    low = butter_filter(s_noisy, 10, fs, "low")
    high = butter_filter(s_noisy, 80, fs, "high")
    band = butter_filter(s_noisy, [20, 60], fs, "bandpass")

    fig, axes = plt.subplots(4, 1, figsize=(11, 8))
    axes[0].plot(t, s_noisy, color=ACCENT_DARK, lw=0.6); axes[0].set_title("signal bruité (5+40+120 Hz)")
    axes[1].plot(t, low, color=PRIMARY); axes[1].set_title("passe-bas 10 Hz → garde 5 Hz")
    axes[2].plot(t, high, color=MAUVAIS); axes[2].set_title("passe-haut 80 Hz → garde 120 Hz")
    axes[3].plot(t, band, color=ACCENT); axes[3].set_title("passe-bande 20-60 Hz → garde 40 Hz")
    for ax in axes:
        ax.set_xlim(0, 0.5)
    fig.tight_layout()
    return fig


demo_butter();
```

<!-- #region -->
On peut visualiser la **réponse en fréquence** $|H(f)|$ des filtres avec `freqz` : c'est le gain
appliqué à chaque fréquence (1 = laissée passer, 0 = coupée).
<!-- #endregion -->

```python
def demo_freq_response() -> plt.Figure:
    """Réponse en fréquence (gain) des filtres Butterworth d'ordre 4."""
    fs = 500.0
    fig, ax = plt.subplots(figsize=(9, 4))
    for cutoff, btype, c, lbl in [(10, "low", PRIMARY, "passe-bas 10 Hz"),
                                  (80, "high", MAUVAIS, "passe-haut 80 Hz")]:
        b, a = sig.butter(4, cutoff / (fs / 2), btype=btype)
        w, h = sig.freqz(b, a, worN=2048, fs=fs)
        ax.plot(w, np.abs(h), color=c, label=lbl)
    ax.set(xlabel="fréquence (Hz)", ylabel="gain |H(f)|",
           title="Réponse en fréquence (Butterworth ordre 4)", xlim=(0, 200))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


demo_freq_response();
```

<!-- #region -->
## 11. Transformée de Hilbert : enveloppe d'amplitude
<!-- #endregion -->

<!-- #region -->
On reprend le **signal analytique** $x_a(t)$ (§2.3). Son module donne l'enveloppe, sa phase la
fréquence instantanée :

![Définition du signal analytique via la transformée de Hilbert](images/tds_hilbert_analytic.png)

$$x_a = \mathcal{F}^{-1}\big(\mathcal{F}(x)\, 2U\big) = x + i\, y$$

où $\mathcal{F}$ est la transformée de Fourier, $U$ l'échelon unité (on **annule la moitié négative**
du spectre) et $y = \mathcal{H}\{x\}$. En pratique, `scipy.signal.hilbert` renvoie directement
$x_a$. Démo sur un **chirp** modulé en amplitude (fréquence 20→100 Hz, modulation à 3 Hz).
<!-- #endregion -->

```python
def demo_hilbert() -> plt.Figure:
    """Enveloppe d'amplitude et fréquence instantanée d'un chirp modulé via Hilbert."""
    duration, fs = 1.0, 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    s = sig.chirp(t, 20.0, t[-1], 100.0)
    s = s * (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))

    analytic = sig.hilbert(s)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(t, s, color=PRIMARY, lw=0.7, label="signal")
    axes[0].plot(t, envelope, color=MAUVAIS, lw=1.5, label="enveloppe")
    axes[0].set(xlabel="temps (s)"); axes[0].legend(fontsize=8)
    axes[1].plot(t[1:], inst_freq, color=ACCENT)
    axes[1].set(xlabel="temps (s)", ylabel="fréquence inst. (Hz)", ylim=(0, 120))
    fig.tight_layout()
    return fig


demo_hilbert();
```

<!-- #region -->
L'enveloppe (rouge) suit fidèlement la modulation d'amplitude à 3 Hz, et la fréquence instantanée
croît bien linéairement de 20 à 100 Hz : la transformée de Hilbert **démodule** le signal.
<!-- #endregion -->

<!-- #region -->
## 12. Ondelettes : débruitage et analyse temps-échelle
<!-- #endregion -->

<!-- #region -->
### 12.1 Décomposition DWT multi-niveaux
<!-- #endregion -->

<!-- #region -->
La **transformée en ondelettes discrète** (DWT) décompose le signal en cascades : à chaque niveau,
un filtre sépare une **approximation** (basses fréquences) et des **détails** (hautes fréquences),
puis on recommence sur l'approximation. Les premiers niveaux captent le bruit pur (haute fréquence) ;
plus on descend, plus on touche à la structure du signal. On applique cela au signal détendancé.
<!-- #endregion -->

```python
def wavelet_decompose(data: np.ndarray, wavelet: str, levels: int):
    """Décomposition DWT multi-niveaux. Renvoie (fig, COEFF_D, DATASET)."""
    dataset = np.asarray(data, dtype=float)
    fig, axarr = plt.subplots(nrows=levels, ncols=2, figsize=(14, 2.0 * levels))
    coeff_list, data_list = [], []
    for ii in range(levels):
        dataset, coeff_d = pywt.dwt(dataset, wavelet, mode="per")
        axarr[ii, 0].plot(dataset, color=PRIMARY)
        axarr[ii, 1].plot(coeff_d, color=MAUVAIS)
        axarr[ii, 0].set_ylabel(f"Niveau {ii + 1}", fontsize=11)
        axarr[ii, 0].set_yticklabels([]); axarr[ii, 1].set_yticklabels([])
        if ii == 0:
            axarr[ii, 0].set_title("coefficients d'approximation")
            axarr[ii, 1].set_title("coefficients de détail")
        coeff_list.append(np.repeat(coeff_d, 2 ** (ii + 1)))
        data_list.append(np.repeat(dataset, 2 ** (ii + 1)))
    fig.tight_layout()
    return fig, coeff_list, data_list


fig_dwt, _, _ = wavelet_decompose(y_detrend, "sym2", levels=6)
fig_dwt;
```

<!-- #region -->
### 12.2 Débruitage sur une série réelle (sunspots)
<!-- #endregion -->

<!-- #region -->
Appliquons-le à une vraie série : l'**activité solaire annuelle** (nombre de taches solaires,
1700-2008), chargée via `statsmodels`. L'approximation de niveau 1 donne une version « débruitée »
qui suit le cycle solaire de ~11 ans.

> Remplace le fichier `day.csv` (sunspots) de l'original, chargé depuis Google-Drive et perdu.
<!-- #endregion -->

```python
def load_sunspots() -> pd.Series:
    """Activité solaire annuelle (sunspots, 1700-2008) via statsmodels ; fallback synthétique."""
    try:
        import statsmodels.api as sm
        d = sm.datasets.sunspots.load_pandas().data
        return pd.Series(d["SUNACTIVITY"].to_numpy(float), index=d["YEAR"].astype(int),
                         name="sunspots")
    except Exception:  # noqa: BLE001 — fallback offline
        yr = np.arange(1700, 2009)
        v = 80 + 70 * np.sin(2 * np.pi * (yr - 1700) / 11) ** 2
        return pd.Series(np.clip(v, 0, None), index=yr, name="sunspots")


sun = load_sunspots()
print(f"sunspots : {len(sun)} ans, de {sun.index.min()} à {sun.index.max()}")

fig_sun, COEFF_D, DATASET = wavelet_decompose(sun.to_numpy(), "sym2", levels=4)
fig_sun;
```

<!-- #region -->
On superpose le signal brut et l'approximation de niveau 1 (re-normalisée), et on mesure la
corrélation entre les coefficients de détail (le « bruit » présumé) et le signal.
<!-- #endregion -->

```python
def plot_sun_reconstruction() -> plt.Figure:
    """Signal sunspots vs approximation niveau 1 (composante 'basse fréquence')."""
    ref = sun.to_numpy()
    approx = DATASET[0][: len(ref)]
    approx = approx * ref.max() / max(approx.max(), 1e-9)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sun.index, ref, color=MAUVAIS, lw=0.8, label="sunspots (brut)")
    ax.plot(sun.index, approx, color=PRIMARY, lw=1.5, label="approximation niveau 1")
    ax.set(xlabel="année", ylabel="activité", title="Débruitage par ondelettes (sunspots)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


plot_sun_reconstruction()

ref_sun = sun.to_numpy()
L = min(len(COEFF_D[0]), len(ref_sun))
corr_sun = np.corrcoef(COEFF_D[0][:L], ref_sun[:L])[0, 1]
print(f"Corrélation |détail niveau 1 ↔ signal| : {abs(corr_sun) * 100:.1f}%")
```

<!-- #region -->
### 12.3 Raffinement : choix du seuil par fit gaussien
<!-- #endregion -->

<!-- #region -->
Le bruit qu'on cherche à retirer est supposé **gaussien**. On peut le vérifier : l'histogramme des
**coefficients de détail** du premier niveau doit ressembler à une gaussienne centrée. On l'ajuste
(`curve_fit`), on en extrait $\sigma$, puis on **annule** tous les coefficients de détail dans
$\pm k\sigma$ (ce sont du bruit) et on reconstruit. On balaie $k$ et on mesure le RMSE **contre le
signal propre** — possible ici car on connaît la vérité terrain (signal synthétique).

> L'original débruitait des données climatiques de Drive (perdues) et son code de raffinement
> référençait une variable `test` jamais définie (bug). On le corrige en travaillant sur le signal
> synthétique, dont le bruit gaussien et le signal propre sont connus.
<!-- #endregion -->

```python
fig_ref, COEFF_S, DATA_S = wavelet_decompose(energy, "sym5", levels=4)
fig_ref;
```

```python
coeff0 = COEFF_S[0]
data0 = DATA_S[0]
Lc = min(len(coeff0), len(clean_sig), len(data0))
coeff0, data0, clean_ref = coeff0[:Lc], data0[:Lc], clean_sig[:Lc]


def gaus(x, a, x0, sigma):
    """Gaussienne a·exp(-(x-x0)²/(2σ²))."""
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fit_detail_gaussian() -> tuple:
    """Ajuste une gaussienne sur l'histogramme des coefficients de détail. Renvoie (fig, sigma)."""
    from scipy.optimize import curve_fit
    hist, edges = np.histogram(coeff0, bins=60)
    x_1 = edges[:-1] + np.diff(edges) / 2
    y_1 = hist.astype(float)
    val_medio = x_1[y_1.argmax()]
    p0 = [y_1.max(), val_medio, np.std(coeff0)]
    popt, _ = curve_fit(gaus, x_1, y_1, p0=p0, maxfev=10000)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_1, y_1, color=PRIMARY, label="histogramme détail")
    ax.plot(x_1, gaus(x_1, *popt), "--", color=MAUVAIS, label="fit gaussien")
    ax.set(xlabel="valeur du coefficient", ylabel="effectif",
           title="Hypothèse de bruit gaussien (coefficients de détail)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, abs(popt[2])


fig_fit, Sigma = fit_detail_gaussian()
print(f"sigma ajusté = {Sigma:.3f}")
fig_fit;
```

<!-- #region -->
$\sigma \approx 0.97$ : très proche de l'écart-type du bruit qu'on a injecté (1.0). On balaie le seuil
et on lit l'optimum (RMSE minimal vs signal propre).
<!-- #endregion -->

```python
def recons_from_threshold(threshold: float) -> np.ndarray:
    """Reconstruit le signal en annulant les coefficients de détail dans ±threshold·sigma."""
    lim = threshold * Sigma
    c0 = coeff0.copy()
    c0[(c0 < lim) & (c0 > -lim)] = 0
    level0 = c0 + data0
    return level0 * (clean_ref.max() / max(level0.max(), 1e-9))


def sweep_thresholds() -> plt.Figure:
    """Balaye le seuil de débruitage et mesure le RMSE vs signal propre (ground truth)."""
    ths = np.arange(0.0, 5.0, 0.25)
    rmse = [np.sqrt(np.mean((recons_from_threshold(t) - clean_ref) ** 2)) for t in ths]
    best = ths[int(np.argmin(rmse))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ths, rmse, "o-", color=PRIMARY)
    ax.axvline(best, color=MAUVAIS, ls="--", label=f"optimum = {best:.2f}σ")
    ax.set(xlabel="seuil (en σ)", ylabel="RMSE vs signal propre",
           title="Choix du seuil de débruitage par ondelettes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, best


fig_sweep, best_th = sweep_thresholds()
print(f"seuil optimal = {best_th:.2f}σ")
fig_sweep;
```

<!-- #region -->
### 12.4 Transformée en ondelettes continue (scaleogramme)
<!-- #endregion -->

<!-- #region -->
La **CWT** (continue) donne une vue **temps-fréquence** lisse, alternative au spectrogramme : on
corrèle le signal avec l'ondelette à toutes les **échelles** et toutes les **positions**. Sur un
signal qui passe de 10 Hz à 40 Hz, le scaleogramme localise nettement la transition.
<!-- #endregion -->

```python
def demo_cwt() -> plt.Figure:
    """Transformée en ondelettes continue (scaleogramme) d'un signal à 2 régimes."""
    fs = 200.0
    t = np.arange(0, 2, 1 / fs)
    s = np.piecewise(t, [t < 1, t >= 1],
                     [lambda tt: np.sin(2 * np.pi * 10 * tt),
                      lambda tt: np.sin(2 * np.pi * 40 * tt)])
    scales = np.arange(1, 128)
    coef, freqs = pywt.cwt(s, scales, "morl", sampling_period=1 / fs)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, s, color=PRIMARY, lw=0.6)
    axes[0].set(title="signal : 10 Hz puis 40 Hz", xlabel="temps (s)")
    pcm = axes[1].pcolormesh(t, freqs, np.abs(coef), shading="gouraud", cmap="viridis")
    fig.colorbar(pcm, ax=axes[1], label="|CWT|")
    axes[1].set(ylim=(0, 60), xlabel="temps (s)", ylabel="fréquence (Hz)",
                title="scaleogramme (CWT, ondelette de Morlet)")
    fig.tight_layout()
    return fig


demo_cwt();
```

<!-- #region -->
## 13. Applications audio (librosa)
<!-- #endregion -->

<!-- #region -->
Un signal audio n'est qu'un signal 1D échantillonné (ici à 22 050 Hz). **librosa** est la
bibliothèque de référence pour l'audio. On charge un extrait réel (une trompette) et on l'analyse :
forme d'onde, spectre global, puis les représentations temps-fréquence standard.

> Dataset **imposé** pour ce notebook (`00_datasets.md`) : échantillons `librosa`.
<!-- #endregion -->

```python
import librosa
import librosa.display

y_audio, sr_audio = librosa.load(librosa.example("trumpet"))
print(f"audio trumpet : {len(y_audio)} échantillons, sr={sr_audio} Hz, "
      f"durée={len(y_audio) / sr_audio:.2f}s")


def plot_waveform() -> plt.Figure:
    """Forme d'onde de l'audio (domaine temporel)."""
    fig, ax = plt.subplots(figsize=(11, 3))
    librosa.display.waveshow(y_audio, sr=sr_audio, color=PRIMARY, ax=ax)
    ax.set(title="forme d'onde — trompette", xlabel="temps (s)")
    fig.tight_layout()
    return fig


plot_waveform();
```

<!-- #region -->
Le **spectre FFT global** montre les harmoniques de la note jouée (un pic fondamental + ses
multiples).
<!-- #endregion -->

```python
def plot_audio_spectrum() -> plt.Figure:
    """Spectre FFT global de l'audio (domaine fréquentiel)."""
    Y = np.abs(rfft(y_audio))
    f = rfftfreq(len(y_audio), 1 / sr_audio)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(f, Y, color=ACCENT_DARK, lw=0.6)
    ax.set(title="spectre FFT global", xlabel="fréquence (Hz)", ylabel="|Y(f)|", xlim=(0, 4000))
    fig.tight_layout()
    return fig


plot_audio_spectrum();
```

<!-- #region -->
Le **spectrogramme STFT** (en dB) révèle l'évolution temporelle des harmoniques — on voit chaque
note attaquée puis tenue.
<!-- #endregion -->

```python
def plot_stft() -> plt.Figure:
    """Spectrogramme STFT en dB."""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(11, 4))
    img = librosa.display.specshow(D, sr=sr_audio, x_axis="time", y_axis="log", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="spectrogramme STFT (échelle log)")
    fig.tight_layout()
    return fig


plot_stft();
```

<!-- #region -->
Enfin, les **descripteurs** classiques utilisés en MIR (*music information retrieval*) et en parole :

- **Mel-spectrogramme** : spectrogramme sur une échelle de fréquences perceptuelle (mel).
- **MFCC** (*Mel-Frequency Cepstral Coefficients*) : compression du mel-spectrogramme, base
  historique de la reconnaissance vocale.
- **Chromagramme** : énergie repliée sur les 12 demi-tons (classe de hauteur) → utile pour
  l'harmonie/les accords.
<!-- #endregion -->

```python
def plot_audio_features() -> plt.Figure:
    """Mel-spectrogramme, MFCC et chromagramme."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y_audio, sr=sr_audio), ref=np.max)
    img0 = librosa.display.specshow(mel, sr=sr_audio, x_axis="time", y_axis="mel",
                                    ax=axes[0], cmap="magma")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB"); axes[0].set(title="mel-spectrogramme")
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr_audio, n_mfcc=13)
    img1 = librosa.display.specshow(mfcc, sr=sr_audio, x_axis="time", ax=axes[1], cmap="viridis")
    fig.colorbar(img1, ax=axes[1]); axes[1].set(title="MFCC (13 coefficients)")
    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr_audio)
    img2 = librosa.display.specshow(chroma, sr=sr_audio, x_axis="time", y_axis="chroma",
                                    ax=axes[2], cmap="coolwarm")
    fig.colorbar(img2, ax=axes[2]); axes[2].set(title="chromagramme")
    fig.tight_layout()
    return fig


plot_audio_features();
```

<!-- #region -->
## 14. Application image : FFT 2D
<!-- #endregion -->

<!-- #region -->
La transformée de Fourier se généralise en **2D** pour les images. Le spectre 2D (centré avec
`fftshift`) place les **basses fréquences au centre** (zones lisses, formes globales) et les **hautes
fréquences en périphérie** (contours, textures, détails). En masquant un disque central on réalise
un **passe-bas** (flou) ; son complément donne un **passe-haut** (détection de contours) — le principe
derrière la compression JPEG et de nombreux filtres image.
<!-- #endregion -->

```python
from skimage import data as skdata


def demo_fft2d() -> plt.Figure:
    """Spectre 2D d'une image + filtrage passe-bas / passe-haut dans le domaine fréquentiel."""
    img = skdata.camera().astype(float)
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))

    rows, cols = img.shape
    cr, cc = rows // 2, cols // 2
    yy, xx = np.ogrid[:rows, :cols]
    dist = np.sqrt((yy - cr) ** 2 + (xx - cc) ** 2)

    low_mask = dist <= 40
    img_low = np.real(np.fft.ifft2(np.fft.ifftshift(F * low_mask)))
    img_high = np.real(np.fft.ifft2(np.fft.ifftshift(F * (~low_mask))))
    vmax = np.percentile(np.abs(img_high), 99)  # contraste robuste pour les contours

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("image")
    axes[0, 1].imshow(mag, cmap="magma"); axes[0, 1].set_title("spectre 2D (log |F|)")
    axes[1, 0].imshow(img_low, cmap="gray"); axes[1, 0].set_title("passe-bas (flou)")
    axes[1, 1].imshow(img_high, cmap="gray", vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title("passe-haut (contours)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig


demo_fft2d();
```

<!-- #region -->
## 15. GAN pour la génération de signaux
<!-- #endregion -->

<!-- #region -->
Dernier volet, plus avancé : générer des signaux **réalistes** avec un **réseau antagoniste
génératif** (GAN). Un GAN oppose deux réseaux dans un jeu à somme nulle :

- le **générateur** $G$ transforme un vecteur de bruit latent en un faux signal ;
- le **discriminateur** $D$ apprend à distinguer vrais signaux et faux.

On optimise un objectif **min-max** :

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$$

À l'équilibre, $G$ produit des signaux indistinguables des vrais.
<!-- #endregion -->

<!-- #region -->
### 15.1 Mise en place de l'expérience
<!-- #endregion -->

<!-- #region -->
On définit notre jeu de données **contrôlé** : des signaux $A\sin(\omega x) + b$ additionnés d'un
bruit blanc gaussien.

![Mise en place : générateur de signaux contrôlés](images/tds_gan_experiment.png)

![Forme du signal](images/tds_gan_signal_formula.png) ![Ajout du bruit](images/tds_gan_noise_formula.png)

Où **A** est l'amplitude, **ω** la fréquence, **b** le biais. Le bruit blanc gaussien (moyenne 0,
écart-type fixe) « vit » à toutes les fréquences.

![Plages des paramètres](images/tds_gan_params.png)

- L'**amplitude** va de 0,1 à 10.
- Le **biais** va de 0,1 à 10.
- La **fréquence** varie de 1 à 2.
- L'**amplitude du bruit** est fixe (0,3).

> On corrige au passage le `plt.shoxw()` de l'original (faute de frappe → `plt.show()`).
<!-- #endregion -->

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
LENGTH = 100
LATENT = 3
X2 = np.linspace(-5, 5, LENGTH)


def sample_params(n: int, rng: np.random.Generator):
    """Tire n jeux de paramètres (amplitude, fréquence, biais) selon les plages choisies."""
    a = rng.uniform(0.1, 10, (n, 1))
    f = rng.uniform(1, 2, (n, 1))
    b = rng.uniform(0.1, 10, (n, 1))
    return a, f, b


def make_signals(n: int, rng: np.random.Generator) -> np.ndarray:
    """n signaux A·sin(ωx)+b + bruit gaussien (matrice n×LENGTH), vectorisé."""
    a, f, b = sample_params(n, rng)
    return a * np.sin(X2[None, :] * f) + b + 0.3 * rng.normal(size=(n, LENGTH))


# Statistiques globales pour standardiser (échelle réduite → apprentissage GAN stable)
_calib = make_signals(4000, np.random.default_rng(1))
MU, SD = float(_calib.mean()), float(_calib.std())


def generate_sample() -> plt.Figure:
    """Affiche un signal paramétré A·sin(ωx)+b + bruit (corrige le `plt.shoxw` de l'original)."""
    rng = np.random.default_rng(0)
    a, f, b = (v.ravel()[0] for v in sample_params(1, rng))
    s = a * np.sin(X2 * f) + b + 0.3 * rng.normal(size=LENGTH)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(X2, s, ".", color=PRIMARY)
    ax.set(title=f"Amplitude {a:.1f}  Fréquence {f:.2f}  Biais {b:.1f}", xlabel="x", ylabel="y")
    fig.tight_layout()
    return fig


generate_sample();
```

<!-- #region -->
### 15.2 Modèles générateur et discriminateur
<!-- #endregion -->

<!-- #region -->
Le **générateur** prend un vecteur latent (dimension 3) et produit un signal de longueur 100. Le
**discriminateur** prend un signal et renvoie un logit réel/faux.

![Le générateur : vecteur latent → signal](images/tds_gan_generator.png)

![Le discriminateur : signal → réel/faux](images/tds_gan_discriminator.png)

> L'original utilisait Keras/TensorFlow (LSTM). On le réimplémente en **PyTorch** (contrainte
> d'environnement : TF et torch ne coexistent pas dans le même process) avec des MLP compacts, et on
> **standardise** les signaux pour stabiliser l'apprentissage.
<!-- #endregion -->

```python
_rng_gan = np.random.default_rng(42)


def real_batch(n: int) -> torch.Tensor:
    """n vrais signaux standardisés (tenseur float32 n×LENGTH)."""
    out = (make_signals(n, _rng_gan) - MU) / SD
    return torch.from_numpy(out.astype(np.float32))


class Generator(nn.Module):
    """Générateur : vecteur latent → signal standardisé de longueur LENGTH."""

    def __init__(self, latent_dim: int = LATENT, length: int = LENGTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, length),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminateur : signal → logit réel/faux."""

    def __init__(self, length: int = LENGTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

<!-- #region -->
### 15.3 Entraînement
<!-- #endregion -->

<!-- #region -->
La boucle alterne : (1) mise à jour de $D$ sur un lot de vrais + un lot de faux (avec **label
smoothing** : on vise 0,9 au lieu de 1 pour les vrais, ce qui stabilise) ; (2) mise à jour de $G$
pour tromper $D$. On suit la progression toutes les 500 étapes.
<!-- #endregion -->

```python
def train_gan(steps: int = 2500, batch: int = 64):
    """Entraîne le GAN (boucle min-max, label smoothing) ; renvoie (générateur, fig de suivi)."""
    gen, disc = Generator(), Discriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    real_lbl = torch.full((batch, 1), 0.9)   # label smoothing
    fake_lbl = torch.zeros(batch, 1)
    snapshots = []
    for step in range(steps):
        x_real = real_batch(batch)
        z = torch.randn(batch, LATENT)
        x_fake = gen(z).detach()
        opt_d.zero_grad()
        loss_d = bce(disc(x_real), real_lbl) + bce(disc(x_fake), fake_lbl)
        loss_d.backward(); opt_d.step()

        z = torch.randn(batch, LATENT)
        opt_g.zero_grad()
        loss_g = bce(disc(gen(z)), torch.ones(batch, 1))
        loss_g.backward(); opt_g.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                snap = gen(torch.randn(1, LATENT)).numpy()[0] * SD + MU
            snapshots.append((step + 1, snap))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.2 * len(snapshots), 3))
    real_disp = real_batch(1).numpy()[0] * SD + MU
    for ax, (st, sample) in zip(np.atleast_1d(axes), snapshots):
        ax.plot(sample, ".", color=MAUVAIS, ms=4, label="généré")
        ax.plot(real_disp, ".", color=PRIMARY, ms=4, alpha=0.5, label="réel")
        ax.set_title(f"step {st}"); ax.legend(fontsize=7)
    fig.tight_layout()
    return gen, fig


gen_model, fig_train = train_gan()
fig_train;
```

<!-- #region -->
### 15.4 Résultats
<!-- #endregion -->

<!-- #region -->
Après entraînement, on génère une grille de signaux (dé-standardisés vers les unités d'origine) et
on les compare à de vrais signaux : le générateur a appris à produire des sinusoïdes bruitées
plausibles (amplitude, fréquence et biais variés).
<!-- #endregion -->

```python
def plot_gan_grid() -> plt.Figure:
    """Grille de signaux générés (dé-standardisés) vs réels après entraînement."""
    with torch.no_grad():
        fakes = gen_model(torch.randn(9, LATENT)).numpy() * SD + MU
    reals = real_batch(9).numpy() * SD + MU
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for k, ax in enumerate(axes.ravel()):
        ax.plot(fakes[k], ".", color=MAUVAIS, ms=3, label="généré")
        ax.plot(reals[k], ".", color=PRIMARY, ms=3, alpha=0.5, label="réel")
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Signaux générés (GAN) vs réels", fontsize=13)
    fig.tight_layout()
    return fig


plot_gan_grid();
```

<!-- #region -->
## 16. Récapitulatif et sources
<!-- #endregion -->

<!-- #region -->
**Quelle transformée / quel outil pour quel besoin ?**

| Besoin | Outil | Fonction |
|---|---|---|
| Contenu fréquentiel **global** d'un signal stationnaire | DFT / FFT | `np.fft.fft`, `scipy.fft.rfft` |
| Relier bins ↔ fréquences | — | `np.fft.fftfreq`, `fftshift` |
| Densité spectrale de puissance robuste | Welch | `scipy.signal.welch` |
| Réduire les **fuites** spectrales | Fenêtrage | `scipy.signal.windows.hann` |
| Contenu fréquentiel **qui évolue dans le temps** | STFT / spectrogramme | `scipy.signal.spectrogram`, `librosa.stft` |
| Analyse **temps-échelle** (transitoires, multi-résolution) | Ondelettes CWT/DWT | `pywt.cwt`, `pywt.dwt` |
| **Débruiter** proprement | Filtres numériques | `scipy.signal.butter` + `filtfilt` |
| **Enveloppe** / fréquence instantanée | Hilbert | `scipy.signal.hilbert` |
| **Audio** (mel, MFCC, chroma) | librosa | `librosa.feature.*` |
| Filtrage / analyse d'**images** | FFT 2D | `np.fft.fft2` |

**Repères pratiques**
- Toujours penser au **théorème de Nyquist** ($f_s > 2 f_{\max}$) et au filtre anti-repliement.
- **Détrend + fenêtrage** avant FFT pour des spectres lisibles.
- `filtfilt` (déphasage nul) plutôt qu'un `lfilter` simple quand la phase compte.
- Pour un signal **non stationnaire**, le spectrogramme ou la CWT, pas la FFT globale.

**2026 — au-delà des méthodes classiques.** Le traitement du signal « deep » est désormais courant :
modèles audio auto-supervisés (**wav2vec 2.0**, **HuBERT**), reconnaissance vocale (**Whisper**),
front-ends différentiables (**nnAudio**, `torchaudio.transforms`) qui rendent STFT/mel apprenables et
GPU. Mais Fourier, les ondelettes et les filtres restent le socle — et souvent la première brique de
ces pipelines.

**Sources**
- *NumPy / SciPy* : documentation `numpy.fft`, `scipy.fft`, `scipy.signal`.
- *PyWavelets* : <https://pywavelets.readthedocs.io>.
- *librosa* : McFee et al., <https://librosa.org>.
- *The Scientist and Engineer's Guide to DSP*, S. Smith (dspguide.com).
- Notebooks frères : `[[TS_Time_Series_Intro]]`, `[[TS_ARIMA]]`, `[[DL_PyTorch]]`.
<!-- #endregion -->
