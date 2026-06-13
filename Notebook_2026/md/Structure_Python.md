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
# 🐍 Structure Python — Constructions du langage
<!-- #endregion -->

<!-- #region -->
Cheat-sheet **et** wiki technique des constructions Python utiles au quotidien data/ML : fonctions, introspection, arguments, fonctionnel, décorateurs, typage moderne, pattern matching, exceptions, context managers, modules, sérialisation et parallélisme.

Public visé : data scientist / ML engineer qui veut écrire du Python idiomatique en 2026 (testé sous **Python 3.12**).

Couverture :

1. Fonctions & fonctions d'ordre supérieur
2. Introspection d'une fonction (`__code__`, `inspect`)
3. Arguments : `*args`, `**kwargs`, keyword/positional-only
4. Lambda & programmation fonctionnelle
5. Décorateurs (dont paramétrés)
6. f-strings
7. Tuple unpacking
8. Compréhensions, générateurs & `itertools`
9. `yield` & générateurs
10. Type hinting moderne (3.10+ / 3.12)
11. Pattern matching (`match`/`case`)
12. Contrôle de flux : `break` / `continue` / `pass` / `for-else`
13. Exceptions (dont `except*`)
14. Context managers
15. Modules & imports
16. Sérialisation : pickle, dataclasses, Pydantic
17. Parallélisme : multiprocessing, futures, asyncio
<!-- #endregion -->

<!-- #region -->
## 1. Fonctions & fonctions d'ordre supérieur
<!-- #endregion -->

<!-- #region -->
Une **fonction d'ordre supérieur** prend une fonction en argument (ou en renvoie une). `apply_all` reçoit `func` et l'applique à chaque élément — brique de base du style fonctionnel. On type les signatures avec `collections.abc.Callable`.
<!-- #endregion -->

```python
from collections.abc import Callable


def add10(x: int) -> int:
    """Renvoie x augmenté de 10."""
    return x + 10


def apply_all(values: list[int], func: Callable[[int], int]) -> list[int]:
    """Applique `func` à chaque élément de `values` (fonction d'ordre supérieur)."""
    return [func(v) for v in values]


print(apply_all([1, 2, 3, 4, 5], add10))
```

<!-- #region -->
Passer le comportement (`add10`) en paramètre évite de réécrire la boucle pour chaque transformation : on découple *l'itération* de *l'opération*.
<!-- #endregion -->

<!-- #region -->
## 2. Introspection d'une fonction
<!-- #endregion -->

<!-- #region -->
Tout objet fonction expose son bytecode via `__code__`. Champs utiles :

- `co_argcount` — nombre d'arguments positionnels
- `co_name` — nom de la fonction
- `co_varnames` — tuple des variables locales
- `co_firstlineno` — numéro de la première ligne de la `def`
<!-- #endregion -->

```python
def myfunction(a: int, b: int) -> int:
    return a + b


print(myfunction.__code__.co_argcount)     # 2  -> nombre d'arguments
print(myfunction.__code__.co_name)         # myfunction -> nom
print(myfunction.__code__.co_varnames)     # ('a', 'b') -> variables locales
print(myfunction.__code__.co_firstlineno)  # ligne de la 1re ligne de la def
```

<!-- #region -->
`__code__` est bas niveau. En pratique on préfère `inspect.signature`, qui reconstruit une signature lisible (paramètres, défauts, annotation de retour) — idéal pour introspecter une API ou écrire un décorateur générique.
<!-- #endregion -->

```python
import inspect

sig = inspect.signature(myfunction)
print(sig)                       # (a: int, b: int) -> int
print(list(sig.parameters))      # ['a', 'b']
print(sig.return_annotation)     # <class 'int'>
```

<!-- #region -->
`inspect` alimente énormément d'outils (debuggers, frameworks de DI, `functools.wraps`) : c'est la voie standard pour raisonner sur du code à l'exécution.
<!-- #endregion -->

<!-- #region -->
## 3. Arguments : *args, **kwargs, keyword/positional-only
<!-- #endregion -->

<!-- #region -->
- `*args` capture les arguments **positionnels** surnuméraires dans un **tuple**.
- `**kwargs` capture les arguments **nommés** surnuméraires dans un **dict**.

C'est ce qui permet d'écrire des fonctions à arité variable (et des décorateurs transparents).
<!-- #endregion -->

```python
def collect_args(*args: int) -> tuple[int, ...]:
    """*args capture les positionnels surnuméraires dans un tuple."""
    return args


print(collect_args())          # ()
print(collect_args(1))         # (1,)
print(collect_args(1, 2, 3))   # (1, 2, 3)


def collect_kwargs(**kwargs: object) -> dict[str, object]:
    """**kwargs capture les nommés surnuméraires dans un dict."""
    return kwargs


print(collect_kwargs())                       # {}
print(collect_kwargs(a=1, b=2, apple="pie"))  # {'a': 1, 'b': 2, 'apple': 'pie'}
```

<!-- #region -->
Une signature complète peut combiner toutes les formes : `/` marque la fin des **positional-only**, `*` le début des **keyword-only**. Symétriquement, `*` et `**` servent aussi **à l'appel** pour déplier une séquence / un dict en arguments.
<!-- #endregion -->

```python
def full_signature(pos_only, /, normal, *args, kw_only, **kwargs):
    return pos_only, normal, args, kw_only, kwargs


print(full_signature(1, 2, 3, 4, kw_only=5, extra=6))

# Unpacking à l'appel : * déplie une liste, ** déplie un dict
nums = [1, 2, 3]
params = {"kw_only": 9}
print(full_signature(0, 10, *nums, **params))
```

<!-- #region -->
Le positional-only (`/`) protège les noms internes (on peut les renommer sans casser les appels) ; le keyword-only (`*`) force des appels explicites et lisibles (`func(timeout=5)` plutôt que `func(5)`).
<!-- #endregion -->

<!-- #region -->
## 4. Lambda & programmation fonctionnelle
<!-- #endregion -->

<!-- #region -->
Une **lambda** est une fonction anonyme d'une expression. Pratique comme petit argument jetable d'une fonction d'ordre supérieur ; au-delà d'une ligne, préférer un `def` nommé (plus lisible et debuggable).
<!-- #endregion -->

```python
add = lambda x, y: x + y          # noqa: E731 — pédagogique
print("add(1, 1) :", add(1, 1))


def higher_order(x: int, func: Callable[[int], int]) -> int:
    return func(x)


print("lambda:", higher_order(100, lambda x: x + 10))
```

<!-- #region -->
La boîte à outils fonctionnelle standard : `map`/`filter` (transformer/filtrer paresseusement), `functools.reduce` (replier en une valeur), `functools.partial` (figer des arguments — *currying*).
<!-- #endregion -->

```python
from functools import reduce, partial

print(list(map(lambda x: x * 2, [1, 2, 3])))        # [2, 4, 6]
print(list(filter(lambda x: x > 1, [0, 1, 2, 3])))  # [2, 3]
print(reduce(lambda a, b: a * b, [1, 2, 3, 4]))     # 24

add5 = partial(add, 5)
print("add5(3) :", add5(3))                         # 8
```

<!-- #region -->
En 2026, pour `map`/`filter` simples on préfère souvent une **compréhension** (cf. §8), plus lisible. `partial` et `reduce` restent précieux pour composer des pipelines et configurer des callbacks.
<!-- #endregion -->

<!-- #region -->
## 5. Décorateurs
<!-- #endregion -->

<!-- #region -->
Un **décorateur** est une fonction qui prend une fonction et en renvoie une version augmentée, **sans modifier le code source** de la cible. Cas d'usage : logging, timing, cache, retry, contrôle d'accès. `functools.wraps` recopie le nom/la docstring de la fonction d'origine sur le wrapper (sinon `greet.__name__` deviendrait `'wrapper'`).
<!-- #endregion -->

```python
from functools import wraps


def logger(func: Callable) -> Callable:
    """Décorateur : trace chaque appel. @wraps préserve nom/docstring de `func`."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Appel de la fonction", func.__name__)
        return func(*args, **kwargs)
    return wrapper


@logger
def greet(name: str) -> None:
    print("Bonjour", name)


greet("John")
print(greet.__name__)  # 'greet' grâce à @wraps (sinon 'wrapper')
```

<!-- #region -->
Pour **paramétrer** un décorateur, on ajoute un niveau : une fabrique qui reçoit les paramètres et renvoie le décorateur. Ici `@repeat(3)` exécute la fonction 3 fois.
<!-- #endregion -->

```python
def repeat(n: int) -> Callable:
    """Fabrique de décorateur : exécute la fonction `n` fois."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


@repeat(3)
def ping() -> str:
    print("ping")
    return "done"


ping()
```

<!-- #region -->
Le décorateur paramétré est le motif derrière `@app.route("/")` (Flask/FastAPI), `@pytest.mark.parametrize(...)`, `@lru_cache(maxsize=128)`, etc.
<!-- #endregion -->

<!-- #region -->
## 6. f-strings
<!-- #endregion -->

<!-- #region -->
Les **f-strings** interpolent des expressions directement dans la chaîne. Le suffixe `=` (3.8+) affiche *à la fois* l'expression et sa valeur — parfait pour le debug rapide.
<!-- #endregion -->

```python
name = "tom"
age = 5
print(f"my name is {name} and my age is {age}")

# Debug `=` : affiche le nom ET la valeur
height = 1.44
print(f"{name=} {age=} {height=}")
```

<!-- #region -->
La **mini-grammaire de format** après `:` contrôle l'affichage : précision (`.2f`), alignement/largeur (`>10`), séparateurs (`,`), base (`b`, `x`), remplissage (`08d`), et même les dates (`%Y-%m-%d`).
<!-- #endregion -->

```python
pi = 3.14159265
print(f"pi is {pi:.2f}")            # 2 décimales
print(f"{pi:>10.2f}")               # aligné à droite sur 10 colonnes
print(f"{1234567:,}")               # séparateurs de milliers
print(f"{42:b} {42:#x} {42:08d}")   # binaire, hexa préfixé, zero-padded

from datetime import datetime
moment = datetime(2026, 6, 2, 14, 30, 0)
print(f"{moment:%Y-%m-%d %H:%M:%S}")
```

<!-- #region -->
Les f-strings sont évaluées à l'exécution et plus rapides que `str.format` ou la concaténation : c'est le mode de formatage par défaut en Python moderne.
<!-- #endregion -->

<!-- #region -->
## 7. Tuple unpacking
<!-- #endregion -->

<!-- #region -->
Le **dépaquetage** affecte plusieurs variables en une fois depuis un itérable. Cas emblématique : l'échange `a, b = b, a` sans variable temporaire (le membre de droite est évalué d'abord, en tuple).
<!-- #endregion -->

```python
person = ["tom", 5, "male"]
nom, age2, genre = person
print(nom, age2, genre)

# Swap sans variable temporaire
a, b = 1, 2
a, b = b, a
print(a, b)
```

<!-- #region -->
L'étoile `*` capture « le reste » dans une liste. Elle peut être en fin **ou au milieu** du motif (un seul `*` par dépaquetage).
<!-- #endregion -->

```python
letters = ["A", "B", "C", "D", "E", "F", "G"]

first, second, *others = letters
print(first, second, others)     # A B ['C', 'D', 'E', 'F', 'G']

head, *middle, tail = letters    # * au milieu
print(head, middle, tail)
```

<!-- #region -->
Le star-unpacking sert aussi à séparer « tête / corps / queue » d'une séquence sans calculer d'indices, et se généralise au pattern matching (§11).
<!-- #endregion -->

<!-- #region -->
## 8. Compréhensions, générateurs & itertools
<!-- #endregion -->

<!-- #region -->
Les **compréhensions** (list/dict/set) construisent une collection en une expression lisible. Le **générateur** `(... for ...)` est sa version *paresseuse* : il ne matérialise rien en mémoire (coût mémoire $O(1)$) et produit les valeurs à la demande — idéal pour les gros flux.
<!-- #endregion -->

```python
squares = [x**2 for x in range(10) if x % 2 == 0]
print(squares)

char_count = {c: "hello".count(c) for c in set("hello")}
print(char_count)

# Générateur paresseux : rien n'est calculé tant qu'on n'itère pas
gen = (x**2 for x in range(10**6))
print(sum(gen))  # somme à la volée, mémoire O(1)
```

<!-- #region -->
`itertools` fournit des briques d'itération combinatoires et paresseuses très efficaces : combinaisons, produit cartésien, concaténation, accumulation, slicing paresseux.
<!-- #endregion -->

```python
import itertools as it

print(list(it.combinations([1, 2, 3, 4], 2)))   # 2 parmi 4
print(list(it.product([0, 1], repeat=3)))       # produit cartésien (3 bits)
print(list(it.chain([1, 2], [3, 4])))           # concat d'itérables
print(list(it.accumulate([1, 2, 3, 4])))        # somme cumulée [1,3,6,10]
print(list(it.islice(range(100), 5, 10, 2)))    # slice paresseux [5,7,9]
```

<!-- #region -->
Règle d'or : si on n'a besoin que d'**itérer une fois**, préférer un générateur à une liste — on économise mémoire et temps de construction.
<!-- #endregion -->

<!-- #region -->
## 9. yield & générateurs
<!-- #endregion -->

<!-- #region -->
Le mot-clé **`yield`** transforme une fonction en **générateur** : à chaque `yield`, l'exécution se suspend et rend une valeur, puis reprend là où elle s'était arrêtée. On obtient une séquence potentiellement infinie, calculée pas à pas.
<!-- #endregion -->

```python
from collections.abc import Iterator


def fibonacci(n: int) -> Iterator[int]:
    """Génère les `n` premiers termes de Fibonacci (lazy via yield)."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


print(list(fibonacci(10)))
```

<!-- #region -->
`yield from` délègue l'itération à un sous-itérable : pratique pour aplatir une structure ou chaîner des générateurs sans boucle imbriquée explicite.
<!-- #endregion -->

```python
def flatten(nested: list[list[int]]) -> Iterator[int]:
    """yield from : délègue l'itération à un sous-générateur."""
    for sub in nested:
        yield from sub


print(list(flatten([[1, 2], [3, 4], [5]])))
```

<!-- #region -->
Les générateurs sont la base du *streaming* de données (lecture de fichiers ligne à ligne, pipelines ETL) : on traite des volumes plus grands que la RAM.
<!-- #endregion -->

<!-- #region -->
## 10. Type hinting moderne
<!-- #endregion -->

<!-- #region -->
Les **annotations de type** documentent les signatures et alimentent les vérificateurs statiques (`mypy`, `pyright`). ⚠️ Elles ne sont **pas vérifiées à l'exécution** : `add_typed("a", "b")` ne lèverait pas d'erreur de type côté Python. Syntaxe moderne : union avec `|` (PEP 604, remplace `Optional`/`Union`) et alias via `TypeAlias`.
<!-- #endregion -->

```python
from typing import TypeAlias


def add_typed(x: int, y: int) -> int:
    """Les annotations documentent mais ne sont PAS vérifiées à l'exécution."""
    return x + y


print(add_typed(2, 1))


def parse(value: int | str | None) -> float:
    if value is None:
        return 0.0
    return float(value)


print(parse("3.5"), parse(None))

Vector: TypeAlias = list[float]


def dot(v1: Vector, v2: Vector) -> float:
    return sum(a * b for a, b in zip(v1, v2))


print(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
```

<!-- #region -->
Pour le code générique : la syntaxe **3.12** (PEP 695) déclare un paramètre de type directement (`class Stack[T]`), et un **`Protocol`** décrit un type *structurel* (duck typing) — n'importe quel objet ayant les bonnes méthodes convient, sans héritage explicite.
<!-- #endregion -->

```python
class Stack[T]:
    """Pile générique typée (syntaxe 3.12)."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()


stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())


from typing import Protocol


class SupportsLen(Protocol):
    def __len__(self) -> int: ...


def total_len(items: list[SupportsLen]) -> int:
    return sum(len(x) for x in items)


print(total_len(["abc", [1, 2], {"k": "v"}]))
```

<!-- #region -->
Bonne pratique 2026 : typer les frontières publiques (API, modèles de données) et lancer `mypy`/`pyright` en CI. Le typage améliore l'autocomplétion et documente les contrats sans surcoût d'exécution.
<!-- #endregion -->

<!-- #region -->
## 11. Pattern matching (match/case)
<!-- #endregion -->

<!-- #region -->
`match`/`case` (3.10+) fait du **filtrage structurel** : il déconstruit la valeur selon sa forme (littéral, séquence `[x, y]`, mapping `{"k": v}`, classe) et peut ajouter des **gardes** (`if ...`). Le `case _` final est le fourre-tout. Plus lisible qu'une cascade de `if/elif` quand on discrimine sur la *structure*.
<!-- #endregion -->

```python
def process_data(data: object) -> str:
    match data:
        case 0:
            return "zéro"
        case int() if data < 0:
            return "entier négatif"
        case [x, y]:
            return f"liste de 2 : {x}, {y}"
        case {"name": name, "age": age}:
            return f"dict personne : {name}, {age}"
        case str() as s:
            return f"chaîne : {s}"
        case _:
            return "autre chose"


for d in [0, -3, [1, 2], {"name": "John", "age": 25}, "Hello", 3.14]:
    print(process_data(d))
```

<!-- #region -->
Les cas sont testés **dans l'ordre** : placer le plus spécifique avant le plus général. Le matching de classes (`Point(x=0, y=0)`) en fait un outil puissant pour traiter des AST, des messages, des événements.
<!-- #endregion -->

<!-- #region -->
## 12. Contrôle de flux : break / continue / pass / for-else
<!-- #endregion -->

<!-- #region -->
- **`break`** interrompt toute la boucle.
- **`continue`** saute l'itération courante et passe à la suivante.
- **`pass`** ne fait *rien* (placeholder syntaxique quand un bloc est obligatoire mais vide).
<!-- #endregion -->

```python
# break : sort de toute la boucle
for i in [1, 2, 3, 4, 5]:
    if i == 3:
        break
    print("break:", i)

# continue : saute UNE itération
for i in [1, 2, 3, 4, 5]:
    if i == 3:
        continue
    print("continue:", i)

# pass : ne fait rien (la boucle continue normalement)
for i in [1, 2, 3, 4, 5]:
    if i == 3:
        pass  # branche volontairement vide
    print("pass:", i)
```

<!-- #region -->
La clause **`for ... else`** (souvent méconnue) exécute le `else` **si la boucle s'est terminée sans `break`** — idiome propre pour « rechercher, sinon traiter l'échec ».
<!-- #endregion -->

```python
for i in range(5):
    if i == 99:
        break
else:
    print("boucle terminée sans break")
```

<!-- #region -->
Différence clé : `pass` laisse la boucle continuer **et** exécuter le reste du corps ; `continue` saute le reste du corps pour cette itération.
<!-- #endregion -->

<!-- #region -->
## 13. Exceptions
<!-- #endregion -->

<!-- #region -->
Structure complète : **`try`** (code risqué) / **`except`** (gestion, on peut ré-encapsuler avec `raise ... from e` pour garder la cause) / **`else`** (s'exécute si aucune exception) / **`finally`** (toujours exécuté, p. ex. libération de ressource). Définir une exception métier = sous-classer `Exception`.
<!-- #endregion -->

```python
class ConfigError(Exception):
    """Exception métier personnalisée."""


def parse_int(s: str) -> int:
    """try/except/else/finally complet."""
    try:
        value = int(s)
    except ValueError as e:
        raise ConfigError(f"Impossible de parser {s!r}") from e
    else:
        print("parse OK")   # uniquement si pas d'exception
        return value
    finally:
        print("fin (toujours exécuté)")


print(parse_int("42"))
try:
    parse_int("abc")
except ConfigError as e:
    print("Capturé:", e)
```

<!-- #region -->
Deux compléments : un *pattern* clavier robuste basé sur `input()` (ici **défini mais non appelé** — pas de stdin en exécution non interactive) ; et les **exception groups** (3.11+) qui agrègent plusieurs erreurs, gérées sélectivement avec `except*`.
<!-- #endregion -->

```python
def lire_entier_securise() -> int:
    """Redemande tant que l'entrée n'est pas un entier (nécessite un stdin)."""
    while True:
        try:
            return int(input("Entrez un nombre : "))
        except ValueError:
            print("Veuillez entrer un nombre valide.")


# Exception groups (3.11+) : plusieurs erreurs gérées avec except*
try:
    raise ExceptionGroup("erreurs multiples", [ValueError("a"), KeyError("b")])
except* ValueError as eg:
    print("ValueError(s):", eg.exceptions)
except* KeyError as eg:
    print("KeyError(s):", eg.exceptions)
```

<!-- #region -->
Bonnes pratiques : capturer l'exception la **plus spécifique** possible, ne jamais avaler une erreur silencieusement (`except: pass`), et préférer `raise ... from e` pour conserver la traçabilité de la cause.
<!-- #endregion -->

<!-- #region -->
## 14. Context managers
<!-- #endregion -->

<!-- #region -->
Un **context manager** (`with`) garantit qu'une ressource est libérée même en cas d'exception. Version par **classe** : implémenter `__enter__` (entrée) et `__exit__` (sortie, toujours appelé). Exemple : un chronomètre.
<!-- #endregion -->

```python
import time


class Timer:
    """Context manager par classe : __enter__ / __exit__."""

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.t0
        print(f"Durée : {self.elapsed:.4f}s")


with Timer():
    sum(range(100_000))
```

<!-- #region -->
Version plus concise : le décorateur **`@contextmanager`** transforme un générateur en context manager. Le code avant `yield` = entrée, après = sortie (à mettre dans un `finally` pour garantir le nettoyage).
<!-- #endregion -->

```python
from contextlib import contextmanager


@contextmanager
def temporary_attr(obj: object, attr: str, new_value: object):
    """Modifie temporairement un attribut, puis le restaure à la sortie."""
    old = getattr(obj, attr)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


class Cfg:
    debug = False


cfg = Cfg()
with temporary_attr(cfg, "debug", True):
    print("dans le bloc, debug =", cfg.debug)
print("après le bloc, debug =", cfg.debug)
```

<!-- #region -->
On retrouve ce motif partout : `open(...)` (fermeture du fichier), `torch.no_grad()`, verrous (`threading.Lock`), transactions DB. Écrire ses propres managers fiabilise la gestion de ressources.
<!-- #endregion -->

<!-- #region -->
## 15. Modules & imports
<!-- #endregion -->

<!-- #region -->
Importer une fonction depuis un autre fichier suppose que ce fichier (module) soit accessible via `sys.path`. Pour rendre la démo **exécutable de bout en bout**, on écrit ici un vrai `helper.py` dans un dossier temporaire, qu'on ajoute au `sys.path`, puis qu'on importe dynamiquement (équivalent de `import helper`).
<!-- #endregion -->

```python
import importlib
import sys
import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
(tmpdir / "helper.py").write_text(
    "def myfunction():\n"
    "    return 'myfunction is called'\n\n"
    "if __name__ == '__main__':\n"
    "    # ce bloc ne tourne QUE si on exécute `python helper.py`\n"
    "    print('exécuté en script')\n",
    encoding="utf-8",
)

sys.path.insert(0, str(tmpdir))
helper = importlib.import_module("helper")  # équivalent de `import helper`
print(helper.myfunction())
```

<!-- #region -->
Une fois le module sur le `sys.path`, on peut aussi importer un nom précis (`from helper import myfunction`). La variable spéciale **`__name__`** vaut le nom du module à l'import, et `'__main__'` quand le fichier est lancé directement — d'où l'idiome `if __name__ == '__main__':` pour séparer code importable et code de script.
<!-- #endregion -->

```python
from helper import myfunction as mf
print(mf())

print("__name__ ici :", __name__)
```

<!-- #region -->
En projet réel, on structure le code en **packages** (dossiers avec `pyproject.toml`) et on importe par chemin qualifié (`from mypkg.utils import myfunction`) plutôt que de manipuler `sys.path` à la main.
<!-- #endregion -->

<!-- #region -->
## 16. Sérialisation : pickle, dataclasses, Pydantic
<!-- #endregion -->

<!-- #region -->
**`pickle`** sérialise des objets Python arbitraires en binaire (`dump`/`load`). ⚠️ **Ne jamais** dé-sérialiser des données non fiables : `pickle.load` peut exécuter du code arbitraire. Pour de la donnée inter-langages, préférer `json` ; pour des modèles ML/numpy, `joblib`.
<!-- #endregion -->

```python
import pickle

lis = [4, 5, 6]
pickle_path = tmpdir / "test.sav"

with open(pickle_path, "wb") as f:
    pickle.dump(lis, f)              # store

with open(pickle_path, "rb") as f:
    restored = pickle.load(f)        # load

print("pickle:", restored)
```

<!-- #region -->
Une **`dataclass`** génère automatiquement `__init__`, `__repr__`, `__eq__` à partir des annotations — parfait pour des structures de données légères. Astuce : pour un défaut **mutable** (liste, dict), utiliser `field(default_factory=...)` (sinon il serait partagé entre instances).
<!-- #endregion -->

```python
from dataclasses import dataclass, field


@dataclass
class Point:
    """dataclass : __init__/__repr__/__eq__ générés automatiquement."""

    x: float
    y: float
    label: str = "origin"
    tags: list[str] = field(default_factory=list)  # défaut mutable -> field

    def distance(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


p1, p2 = Point(0, 0), Point(3, 4)
print(p2.distance(p1), "|", p1)
```

<!-- #region -->
Pour la **validation** à la frontière (API, config, données externes), **Pydantic v2** est le standard 2026 : les contraintes (`Field(ge=0, le=120)`, `EmailStr`) sont vérifiées **à la construction**, et `model_dump_json()` sérialise. Une entrée invalide lève `ValidationError` avec le détail des erreurs.
<!-- #endregion -->

```python
from pydantic import BaseModel, Field, EmailStr, ValidationError


class User(BaseModel):
    """Pydantic v2 : validation + (dé)sérialisation production-grade."""

    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=120)
    email: EmailStr


u = User(name="Marie", age=30, email="marie@example.com")
print(u.model_dump_json())

try:
    User(name="", age=200, email="not-an-email")
except ValidationError as e:
    print("Pydantic a rejeté :", len(e.errors()), "erreurs")
```

<!-- #region -->
Règle de choix : `dataclass` pour une structure interne simple ; **Pydantic** dès qu'il faut valider/parser de la donnée venue de l'extérieur ; `pickle`/`joblib` pour persister des objets Python (en confiance) ; `json` pour l'interopérabilité.
<!-- #endregion -->

<!-- #region -->
## 17. Parallélisme : multiprocessing, futures, asyncio
<!-- #endregion -->

<!-- #region -->
Pour le **CPU-bound**, le *GIL* empêche les threads de s'exécuter vraiment en parallèle → on utilise **plusieurs processus** (`multiprocessing.Pool`). Chaque processus a son propre interpréteur. ⚠️ Le code de lancement doit être sous `if __name__ == "__main__":` (sérialisation des tâches). Ici la taille est réduite pour un run rapide.
<!-- #endregion -->

```python
import math
import multiprocessing as mp


def cpu_task(x: int) -> float:
    """Tâche CPU-bound (réutilise math.log)."""
    return x ** 2 - x * math.log(x + 1)


def bench_multiprocessing() -> None:
    n = 200_000

    t0 = time.perf_counter()
    for x in range(n):
        cpu_task(x)
    print(f"Sans multiprocessing : {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    with mp.Pool(4) as pool:
        pool.map(cpu_task, range(n))
    print(f"Avec multiprocessing : {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    bench_multiprocessing()
```

<!-- #region -->
`concurrent.futures` offre une API plus haut niveau. Pour de l'**I/O-bound** (réseau, disque), un `ThreadPoolExecutor` suffit : pendant l'attente I/O le GIL est relâché, donc les threads progressent réellement en parallèle.
<!-- #endregion -->

```python
from concurrent.futures import ThreadPoolExecutor


def io_task(x: int) -> int:
    time.sleep(0.01)  # simule une latence I/O
    return x * x


with ThreadPoolExecutor(max_workers=4) as ex:
    print("threads:", list(ex.map(io_task, range(8))))
```

<!-- #region -->
Pour beaucoup d'I/O concurrentes, **`asyncio`** est encore plus efficace : un seul thread, des coroutines (`async`/`await`) multiplexées sur une boucle d'événements. `asyncio.gather` lance plusieurs coroutines et attend leurs résultats.
<!-- #endregion -->

```python
import asyncio


async def fetch(url: str, delay: float) -> str:
    """Simule un appel réseau non bloquant."""
    await asyncio.sleep(delay)
    return f"data from {url}"


async def gather_all() -> list[str]:
    return await asyncio.gather(
        fetch("a", 0.05), fetch("b", 0.05), fetch("c", 0.05)
    )


print(asyncio.run(gather_all()))
```

<!-- #region -->
Mémo de choix : **`asyncio`** pour énormément d'I/O concurrentes (web, réseau) ; **threads** pour de l'I/O modérée ou des libs bloquantes ; **processus** pour le CPU-bound. Note : dans un notebook Jupyter une boucle asyncio tourne déjà — on peut alors `await` directement au lieu de `asyncio.run`.
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [Python docs — The Python Language Reference](https://docs.python.org/3/reference/)
- [Python docs — `functools`](https://docs.python.org/3/library/functools.html) · [`itertools`](https://docs.python.org/3/library/itertools.html) · [`inspect`](https://docs.python.org/3/library/inspect.html)
- [PEP 604 (union `|`)](https://peps.python.org/pep-0604/) · [PEP 634 (match)](https://peps.python.org/pep-0634/) · [PEP 695 (generics 3.12)](https://peps.python.org/pep-0695/)
- [Pydantic v2 — documentation](https://docs.pydantic.dev/latest/)
- [Fluent Python — Luciano Ramalho (2e éd.)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
- Notebooks liés : `Structures_L_T_D_Cheat_Sheet`, `Structures_Preprocessing`, `Structures_DataFrame`.
<!-- #endregion -->
