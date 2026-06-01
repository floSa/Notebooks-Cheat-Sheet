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
# 📚 Structures de données — List / Tuple / Dict / NumPy / xarray
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet** : structures de données usuelles en Python (built-ins) et scientific (NumPy, xarray) pour le data work.

Couverture :

1. **List** — méthodes essentielles, opérations slicing avancées.
2. **Tuple** — immutable, unpacking, namedtuple.
3. **Dict** — itération, merge, dict-of-dicts, transformations.
4. **Set** — opérations ensemblistes.
5. **Collections** stdlib : `Counter`, `defaultdict`, `deque`, `OrderedDict`, `ChainMap`.
6. **NumPy** — création, reshape, broadcast, indexing avancé, ufuncs.
7. **xarray** — N-D arrays labellisés (multi-dim avec metadata).
<!-- #endregion -->

<!-- #region -->
## 1. List — opérations essentielles
<!-- #endregion -->

```python
xs = [3, 1, 4, 1, 5, 9, 2, 6]

# Slicing avancé
xs[::-1]                  # reverse
xs[1:6:2]                  # [1, 1, 9] (start:stop:step)
xs[-3:]                    # 3 derniers
xs[:3] + xs[-3:]           # premiers + derniers (concat)

# Modification
xs.append(0)
xs.extend([7, 8])          # vs +=, append d'un seul iterable
xs.insert(0, -1)
xs.remove(1)               # supprime PREMIÈRE occurrence
xs.pop(0)                  # supprime + renvoie 1er
xs.pop()                   # supprime + renvoie dernier
xs.sort()                  # in-place, stable, par défaut ascendant
sorted(xs, reverse=True)   # nouvelle liste

# Recherche
xs.count(1)                # nb d'occurrences
xs.index(4)                # 1ère position

# Tri custom
words = ["apple", "Bee", "carrot"]
sorted(words, key=str.lower)               # tri case-insensitive
sorted([(1, "b"), (2, "a")], key=lambda x: x[1])  # tri par 2e élément

# Aplatir une liste de listes
nested = [[1,2], [3,4], [5,6]]
[x for sublist in nested for x in sublist]   # [1,2,3,4,5,6]

# Ou via itertools
import itertools as it
list(it.chain.from_iterable(nested))
```

<!-- #region -->
## 2. Tuple — unpacking et namedtuple
<!-- #endregion -->

```python
# Tuple = liste immutable, hashable (peut être clé de dict)
t = (1, 2, 3)

# Unpacking
a, b, c = t
a, *rest = t        # rest = [2, 3]
*init, last = t     # init = [1, 2], last = 3
a, _, c = t         # ignorer avec _

# Swap (très Pythonic)
a, b = 10, 20
a, b = b, a

# Named tuple : tuple avec attributs nommés
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y, p._asdict())

# Better : dataclass(frozen=True) (3.7+)
from dataclasses import dataclass
@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

v = Vec2(1.0, 2.0)
```

<!-- #region -->
## 3. Dict — manipulations courantes
<!-- #endregion -->

```python
d = {"a": 1, "b": 2, "c": 3}

# Itération
for k in d: ...                # itère les clés
for k, v in d.items(): ...
for v in d.values(): ...

# Access safe
d.get("a", 0)                  # 0 si absent (vs d["a"] qui raise KeyError)
d.setdefault("z", 100)         # set si absent + return

# Merge (3.9+)
d1, d2 = {"a": 1, "b": 2}, {"b": 20, "c": 30}
merged = d1 | d2               # right wins → {'a': 1, 'b': 20, 'c': 30}

# Inverser key↔value
inverted = {v: k for k, v in d.items()}

# Filter
{k: v for k, v in d.items() if v > 1}

# Transform values
{k: v*2 for k, v in d.items()}

# Dict de dicts (nested)
nested = {"users": {"alice": {"age": 30}, "bob": {"age": 25}}}
nested["users"]["alice"]["age"]
```

<!-- #region -->
## 4. Set — opérations ensemblistes
<!-- #endregion -->

```python
a, b = {1, 2, 3, 4}, {3, 4, 5, 6}

print(a | b)        # union {1,2,3,4,5,6}
print(a & b)        # intersection {3,4}
print(a - b)        # différence {1,2}
print(a ^ b)        # diff symétrique {1,2,5,6}
print(a.issubset(b), a.isdisjoint(b))

# frozenset : set immutable (hashable)
fs = frozenset({1, 2, 3})
```

<!-- #region -->
## 5. `collections` — outils stdlib essentiels
<!-- #endregion -->

```python
from collections import Counter, defaultdict, deque, OrderedDict, ChainMap

# Counter : compteur d'occurrences
c = Counter("mississippi")
print(c.most_common(3))             # [('i', 4), ('s', 4), ('p', 2)]

# defaultdict : value par défaut à la création
words_by_len = defaultdict(list)
for w in ["apple", "bee", "car", "dog"]:
    words_by_len[len(w)].append(w)
print(dict(words_by_len))           # {5: ['apple'], 3: ['bee', 'car', 'dog']}

# deque : double-ended queue, append/pop O(1) des 2 côtés
dq = deque([1, 2, 3], maxlen=5)
dq.appendleft(0); dq.append(4); dq.append(5); dq.append(6)
print(dq)                            # rotation auto si > maxlen

# ChainMap : view sur plusieurs dicts (cherche dans l'ordre)
cm = ChainMap({"a": 1}, {"a": 2, "b": 3})   # {'a': 1, 'b': 3} (premier gagne)
```

<!-- #region -->
## 6. NumPy — fondamentaux
<!-- #endregion -->

```python
import numpy as np

# Création
a = np.array([[1,2,3], [4,5,6]])
np.zeros((3, 4))
np.ones((2, 3), dtype=np.float32)
np.eye(4)
np.arange(10, 20, 2)              # [10,12,14,16,18]
np.linspace(0, 1, 5)              # 5 valeurs régulières
np.random.default_rng(42).normal(0, 1, (3, 3))

# Shape / reshape
a.shape, a.size, a.ndim, a.dtype
a.reshape(3, 2)
a.flatten()                       # 1D copy
a.ravel()                         # 1D view (zero-copy si possible)
a.T                                # transpose
a[None, :, :]                     # ajouter un axe (1, 2, 3)
np.expand_dims(a, axis=0)         # idem
```

```python
# Indexing avancé
arr = np.arange(24).reshape(4, 6)

arr[1:3, 2:5]                     # sous-bloc
arr[:, ::2]                        # tous les 2 cols
arr[arr > 10]                     # masque booléen (1D output)
arr[[0, 2], [1, 3]]               # fancy indexing : (arr[0,1], arr[2,3])
arr[arr.sum(axis=1) > 50]         # filter rows par condition
np.where(arr > 10, arr, -1)       # ternaire vectorisé
```

```python
# Broadcasting (la superpuissance NumPy)
m = np.arange(12).reshape(3, 4)   # (3, 4)
v = np.array([10, 20, 30, 40])    # (4,) → broadcast en (1, 4)
print(m + v)                       # (3, 4) + (4,) → (3, 4)

col = np.array([[100], [200], [300]])  # (3, 1)
print(m + col)                          # (3, 4) + (3, 1) → (3, 4)

# Ufuncs : appliquées élément par élément
np.sin(arr), np.exp(arr), np.log1p(arr)

# Axis : pour les agrégations
arr.sum(axis=0)    # somme par colonne
arr.mean(axis=1)   # moyenne par ligne
arr.std()
```

<!-- #region -->
## 7. xarray — N-D arrays labellisés
<!-- #endregion -->

<!-- #region -->
**xarray** = pandas + numpy : tableaux multi-dim avec dimensions nommées + coordinates + metadata. Standard pour les données scientifiques (météo, ocean, neurosciences, ML 3D-4D).

```python
"""
import xarray as xr
import numpy as np

# Création depuis numpy avec dimensions nommées
da = xr.DataArray(
    np.random.randn(3, 4, 5),
    dims=["time", "lat", "lon"],
    coords={
        "time": pd.date_range("2026-01-01", periods=3),
        "lat": [-90, -45, 0, 45],
        "lon": [0, 90, 180, 270, 360],
    },
    name="temperature",
)

# Sélection PAR LABEL (et non par index, contrairement à numpy)
da.sel(time="2026-01-01", lat=0)
da.sel(lat=slice(-45, 45))   # range labels

# Agrégations conservant les labels
da.mean(dim="time")           # moyenne sur la dim time, conserve lat/lon
da.groupby("lat").mean()      # group by lat values
"""
```

xarray est très puissant pour :

- Données géospatiales / climatiques.
- Time series multi-variables.
- Tensor data avec axes interprétables (image RGB-A avec dim "channel", batch d'images avec dim "sample").
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [Python collections docs](https://docs.python.org/3/library/collections.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [xarray docs](https://docs.xarray.dev/)
- Notebooks liés : `Structure_Python`, `Structures_DataFrame`, `Structures_Preprocessing`.
<!-- #endregion -->
