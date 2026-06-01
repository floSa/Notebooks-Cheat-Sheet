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
# 🐍 Python — Patterns avancés (Cheat-sheet)
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet** des patterns Python utiles en data/ML : itération, fonctionnel, async, typage, performance.

Public visé : data scientist/ML engineer qui veut écrire du Python idiomatique 2026.

Couverture :

1. Comprehensions, generators, `itertools`.
2. Fonctionnel : `lambda`, `map`, `filter`, `reduce`, `partial`, `@functools.cache`.
3. Decorators : `@property`, `@classmethod`, `@staticmethod`, `@dataclass`, custom decorators.
4. f-strings avancées (formatting, debug `f"{x=}"`).
5. **Type hinting moderne** (3.10+ : `|`, `TypeAlias`, `ParamSpec`, generics).
6. Pattern matching (`match/case`).
7. Context managers (`with`, `contextlib`).
8. Exception handling (`try/except/else/finally`, exceptions custom).
9. `asyncio` (concurrence I/O).
10. `multiprocessing` (parallèle CPU).
11. Pickle / dataclasses / `attrs` / Pydantic — sérialisation.
<!-- #endregion -->

<!-- #region -->
## 1. Comprehensions et générateurs
<!-- #endregion -->

```python
# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]
print(squares)

# Dict / Set comprehension
char_count = {c: "Hello".count(c) for c in set("Hello")}
print(char_count)

# Generator (lazy, économe en mémoire)
gen = (x**2 for x in range(10**6))
# Pas de mémoire allouée tant qu'on n'itère pas
print(sum(gen))   # somme à la volée
```

```python
import itertools as it

# Combinaisons utiles
list(it.combinations([1,2,3,4], 2))       # combinaisons 2 parmi 4
list(it.permutations([1,2,3], 2))         # permutations
list(it.product([0,1], repeat=3))         # produit cartésien (binaire 3 bits)
list(it.chain([1,2], [3,4], [5,6]))       # concat de plusieurs iterables
list(it.groupby("aaabbbcc"))              # group consécutifs
list(it.accumulate([1,2,3,4]))            # somme cumulée [1,3,6,10]
list(it.islice(range(100), 5, 10, 2))     # slice [5, 7, 9]
list(it.zip_longest([1,2,3], "abcd", fillvalue="-"))  # zip qui pad
```

<!-- #region -->
## 2. Fonctionnel
<!-- #endregion -->

```python
from functools import reduce, partial, cache

# lambda + map + filter (moins idiomatique en 2026 — préfère comprehensions)
list(map(lambda x: x*2, [1,2,3]))       # [2, 4, 6]
list(filter(lambda x: x > 1, [0,1,2,3]))  # [2, 3]
reduce(lambda a, b: a*b, [1,2,3,4])      # 24

# partial : currying
add = lambda x, y: x + y
add5 = partial(add, 5)
print(add5(3))   # 8

# cache : memoization
@cache
def fib(n):
    return fib(n-1) + fib(n-2) if n > 1 else n

print(fib(50))   # rapide grâce au cache
```

<!-- #region -->
## 3. Decorators
<!-- #endregion -->

```python
from functools import wraps
import time

def timing(fn):
    """Decorator qui logge le temps d'exécution."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.perf_counter() - t0:.3f}s")
        return result
    return wrapper


@timing
def slow_sum(n):
    return sum(range(n))


slow_sum(1_000_000)
```

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    label: str = "origin"
    tags: list[str] = field(default_factory=list)   # mutable default — utiliser field

    def distance(self, other: "Point") -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5


p1 = Point(0, 0)
p2 = Point(3, 4)
print(p2.distance(p1))   # 5.0
print(p1)                 # auto-repr
```

<!-- #region -->
## 4. f-strings avancées
<!-- #endregion -->

```python
x, y = 3.14159, 42

# Format spec
f"{x:.2f}"          # '3.14'
f"{x:>10.2f}"       # '      3.14' (right-align)
f"{x:,.2f}"         # '3.14' (séparateurs de milliers : '1,234.56')
f"{y:b}"            # '101010' (binaire)
f"{y:#x}"           # '0x2a' (hex avec préfixe)
f"{y:08d}"          # '00000042' (zero-padded)

# Debug (3.8+) : affiche aussi le nom de la variable
print(f"{x=}, {y=}")

# Format datetime
from datetime import datetime
now = datetime.now()
print(f"{now:%Y-%m-%d %H:%M:%S}")
```

<!-- #region -->
## 5. Type hinting moderne (3.10+)
<!-- #endregion -->

```python
from typing import TypeAlias, Callable, Any
# Plus besoin de `from typing import List, Dict, Optional, Union` en 3.10+

# Syntaxe moderne (PEP 604 union types)
def parse(value: int | str | None) -> float:
    if value is None:
        return 0.0
    return float(value)


# TypeAlias pour les types complexes
Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]
RGB: TypeAlias = tuple[int, int, int]


def dot(v1: Vector, v2: Vector) -> float:
    return sum(a*b for a, b in zip(v1, v2))


# Callable
def apply(fn: Callable[[int], int], values: list[int]) -> list[int]:
    return [fn(v) for v in values]


print(apply(lambda x: x**2, [1, 2, 3]))
```

```python
# Generics — 3.12+ syntax
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()


stack_int = Stack[int]()
stack_int.push(1); stack_int.push(2)
print(stack_int.pop())
```

<!-- #region -->
## 6. Pattern matching (3.10+)
<!-- #endregion -->

```python
def classify(value):
    match value:
        case 0:
            return "zero"
        case int() if value < 0:
            return "negative int"
        case int():
            return "positive int"
        case float():
            return "float"
        case str() if value.startswith("http"):
            return "url"
        case list() if len(value) == 0:
            return "empty list"
        case [first, *rest]:
            return f"list starting with {first}"
        case {"name": name, "age": age}:
            return f"dict person {name} aged {age}"
        case _:
            return "unknown"


for v in [0, -5, 3.14, "https://x.com", [], [1,2,3], {"name": "Marie", "age": 30}]:
    print(f"{str(v):30s} -> {classify(v)}")
```

<!-- #region -->
## 7. Context managers
<!-- #endregion -->

```python
from contextlib import contextmanager
import time

# Manager via classe
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        print(f"Elapsed : {self.elapsed:.3f}s")


with Timer():
    time.sleep(0.1)


# Manager via décorateur (plus concis)
@contextmanager
def temporary_value(obj, attr, new_value):
    old = getattr(obj, attr)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        setattr(obj, attr, old)
```

<!-- #region -->
## 8. Exception handling
<!-- #endregion -->

```python
class ConfigError(Exception):
    """Custom exception."""

def parse_int(s: str) -> int:
    try:
        return int(s)
    except ValueError as e:
        raise ConfigError(f"Cannot parse {s!r} as int") from e
    else:
        print("parse OK")
    finally:
        print("done")


# Exception groups (3.11+)
try:
    raise ExceptionGroup("multiple errors", [ValueError("a"), KeyError("b")])
except* ValueError as eg:
    print("Got ValueError(s):", eg.exceptions)
except* KeyError as eg:
    print("Got KeyError(s):", eg.exceptions)
```

<!-- #region -->
## 9. asyncio (concurrence I/O)
<!-- #endregion -->

```python
import asyncio

async def fetch_url(url: str, delay: float) -> str:
    """Simule un appel I/O concurrent."""
    await asyncio.sleep(delay)
    return f"data from {url}"


async def main():
    # Lance 3 fetchs en parallèle, attend tous
    results = await asyncio.gather(
        fetch_url("a", 0.1),
        fetch_url("b", 0.1),
        fetch_url("c", 0.1),
    )
    return results


# Dans un notebook Jupyter, asyncio est déjà running, donc on utilise await directement
# Sinon : asyncio.run(main())
print(asyncio.run(main()))
```

<!-- #region -->
## 10. multiprocessing (parallèle CPU)
<!-- #endregion -->

```python
# Pour CPU-bound, le GIL Python empêche le vrai parallélisme avec threads
# → utiliser multiprocessing (ou concurrent.futures.ProcessPoolExecutor)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def slow_square(x):
    time.sleep(0.05)  # simule du calcul lourd
    return x * x


# ThreadPool : OK pour I/O-bound
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(slow_square, range(8)))
print(f"Thread : {results}")


# ProcessPool : pour CPU-bound (mais overhead de spawn)
# Décommenter pour test ; sur Windows + Jupyter, peut être délicat
# with ProcessPoolExecutor(max_workers=4) as ex:
#     results = list(ex.map(slow_square, range(8)))
```

<!-- #region -->
## 11. Sérialisation : pickle, dataclasses, Pydantic
<!-- #endregion -->

```python
import pickle

# Pickle — pratique mais NON sécurisé (n'unpickle JAMAIS de la data non fiable)
data = {"x": [1, 2, 3], "y": "hello"}
pickle_bytes = pickle.dumps(data)
restored = pickle.loads(pickle_bytes)
print(restored)
```

```python
# Pydantic — validation + sérialisation production-grade (2026 standard)
# pip install pydantic
"""
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=120)
    email: EmailStr

u = User(name="Marie", age=30, email="m@x.com")  # validation auto
print(u.model_dump_json())
"""
```

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Python docs — Language Reference](https://docs.python.org/3/reference/)
- [Real Python](https://realpython.com/)
- [Fluent Python — Luciano Ramalho (livre)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
- [PEPs récents (3.10-3.13)](https://peps.python.org/)
- Notebooks liés : `Structures_L_T_D_Cheat_Sheet`, `Structures_Preprocessing`.
<!-- #endregion -->
