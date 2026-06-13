---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="At9TzJyGiUVg" -->
### Apply function
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XP3noEUf2-Yq" executionInfo={"status": "ok", "timestamp": 1702566327307, "user_tz": -60, "elapsed": 4, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="57e4e49a-93dd-4d90-e1f5-1fa138baf97d"
def add10(x):
  return x + 10

def apply_all(list1, function):
  out = []
  for i in list1:
    out.append(function(i))
  return out

print(apply_all([1,2,3,4,5], add10))
```

<!-- #region id="0UrQLx_X6Bhz" -->
- co_argcount — the number of arguments  
- co_name — the function name  
- co_varnames — a tuple containing all variable names  
- co_firstlineno — the line number of the FIRST line of the function
<!-- #endregion -->

<!-- #region id="bVoIzCHoidRG" -->
### Catch Informations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dN1ZPttm3Esk" executionInfo={"status": "ok", "timestamp": 1702567081675, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="20639c52-b90e-4210-e6eb-5f4a235f3218"
def myfunction(a, b):
    return a + b

print(myfunction.__code__.co_argcount)      # 2
print(myfunction.__code__.co_name)          # myfunction
print(myfunction.__code__.co_varnames)      # ('a', 'b')
print(myfunction.__code__.co_firstlineno)   # 1
```

<!-- #region id="xqHR1OAyijZ3" -->
**Arguments**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KOAhZQux5xmb" executionInfo={"status": "ok", "timestamp": 1702567204754, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fd998188-5c9b-4f2d-8942-675ece1e3f39"
def test(*args):
  print(args)

test()         # ()
test(1)        # (1,)
test(1,2)      # (1, 2)
test(1,2,3)    # (1, 2, 3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Rv2WrekV6dQa" executionInfo={"status": "ok", "timestamp": 1702567414670, "user_tz": -60, "elapsed": 264, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="093e5d39-d28f-47df-a260-e8fe0a041d5a"
def test(**kwargs):
  print(kwargs)

test()                         # {}
test(a=1)                      # {'a':1}
test(a=1, b=2)                 # {'a':1, 'b':2}
test(a=1, b=2, apple='pie')    # {'a':1, 'b':2, 'apple':'pie'}
```

<!-- #region id="QEladqkQ6yuT" -->
###Décorateur
<!-- #endregion -->

<!-- #region id="KrQ9nkEL7B37" -->
- En Python, un décorateur est une fonction qui prend en paramètre une autre fonction et qui retourne une nouvelle fonction. La nouvelle fonction est une version modifiée de la fonction d'origine, avec des fonctionnalités supplémentaires ajoutées par le décorateur.

- Les décorateurs sont utiles pour ajouter des fonctionnalités supplémentaires à des fonctions existantes, sans avoir à modifier leur code source. Ils sont également utilisés pour ajouter des fonctionnalités communes à plusieurs fonctions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bGCcwt9A6hDB" executionInfo={"status": "ok", "timestamp": 1702567780532, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="32e498b4-caa4-4cff-ee2d-6a4f30b8129e"
def logger(function):
    def wrapper(*args, **kwargs):
        print("Appel de la fonction", function.__name__)
        return function(*args, **kwargs)
    return wrapper

@logger
def greet(name):
    print("Bonjour", name)

greet("John")
```

<!-- #region id="Su8JTkNc9cYZ" -->
### f-strings (formatted strings)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="jhtUMiv-65Yp" executionInfo={"status": "ok", "timestamp": 1702568001116, "user_tz": -60, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="cfb96c9e-2baf-42ea-f3a5-dd7e092096cc"
# using f-strings
name = 'tom'
age = 5

s = f'my name is {name} and my age is {age}'
s
```

```python colab={"base_uri": "https://localhost:8080/"} id="O0AW5o7J9fuo" executionInfo={"status": "ok", "timestamp": 1702568059952, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="daa023fd-fbf5-45b8-f74f-793772cb861e"
name = 'tom'
age = 5
height = 1.44

s = f'{name=} {age=} {height=}'
print(s)

pi = 3.14159265
s = f'pi is {pi:.2f}'
print(s)
```

<!-- #region id="lW6Ud6YZ-IvQ" -->
###Tuple Unpacking (with & without *)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OIvGD24D9qnY" executionInfo={"status": "ok", "timestamp": 1702568223314, "user_tz": -60, "elapsed": 272, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="29aa9c82-5567-4e74-932d-3bcf7d46db8d"
person = ['tom', 5, 'male']

name, age, gender = person
print(name, age, gender )
```

```python colab={"base_uri": "https://localhost:8080/"} id="Gu3CtaLj-Qbv" executionInfo={"status": "ok", "timestamp": 1702568286341, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="6d98bd81-8e29-43f6-b257-87949fac8b94"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
first, second, *others = letters
print(first)
print(second)
print(*others)
```

<!-- #region id="xKUaycRDiBCs" -->
### Importer des fichier comme fonction d'un autre fichier
<!-- #endregion -->

<!-- #region id="M7HpG8jhjtCu" -->
/!\ Il est important d'avoir helper.py et main.py dans le même repertoire
<!-- #endregion -->

```python id="rCmPu9Vt-fB6"
'''helper.py'''

def myfunction():
  print('myfunction is called')

if __name__ == '__main__':
  print('hahahahaha')
  # this block only runs when helper.py itself is run
  # eg. python helper.py
```

```python id="RQjLU64BiJpi"
'''main.py'''

from helper import *

myfunction() # myfunction is called
```

<!-- #region id="ktQxUWz7j_ow" -->
**ou**
<!-- #endregion -->

```python id="Xo-yce9niL0g"
'''main.py'''

from helper import myfunction

myfunction()    # hi
```

<!-- #region id="ZVIOkjX7kC_Q" -->
### Lambda
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zICnqSC9iL3u" executionInfo={"status": "ok", "timestamp": 1703148920975, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4a2ed5cd-62a8-43db-92bd-0a917682ec2f"
def add(x, y):
  return x + y

add = lambda x,y : x+y
print("add(1,1) : ",  add(1,1) )

##########################################

def avg(a, b, c):
  return (a+b+c)/3

avg = lambda a,b,c : (a+b+c)/3
print("avg(1,1,1) : ",  avg(1,1,1) )
```

```python colab={"base_uri": "https://localhost:8080/"} id="dMQtvEzzlMH8" executionInfo={"status": "ok", "timestamp": 1703149083350, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4221cbde-00be-4fb2-e424-b36bc28ec3c7"
def higher_order_function(x, myfunction):
  return x + 10

# CASE 1: not using lambda functions
def myfunction(x):
  return x + 10

print("higher_order_function:", higher_order_function(100, myfunction) )

# CASE 2: using lambda functions
print( "lambda:",  higher_order_function(100, lambda x:x+10))
```

<!-- #region id="6A0VPbG0mrMw" -->
### Type Hinting & Enforcement
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ibxkVOEAll8i" executionInfo={"status": "ok", "timestamp": 1703149754108, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="16242f7a-9e31-4992-9353-d5e1754045a8"
def add(x:int, y:int) -> int:
  return x + y

print(add(2,1))
```

<!-- #region id="Hu2h5fpbozEO" -->
### Try Exept Pass Continue Finally
<!-- #endregion -->

```python id="m8eD3RqNmugX"
try:
  # code risqué qui pourrait causer des exceptions

except :
  # ce bloc s'exécute si une exception se produit dans le bloc try

finally :
  # ce qui se trouve ici sera TOUJOURS exécuté
```

```python colab={"base_uri": "https://localhost:8080/"} id="UEMpi8BE0vn6" executionInfo={"status": "ok", "timestamp": 1703152923142, "user_tz": -60, "elapsed": 2568, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3582ef53-ddfe-4b3d-80eb-ec6e5d02eb0a"
def lire_clavier():
    while True:
        try:
            num = int(input("Entrez un nombre : "))
            return num
        except ValueError:
            print("Veuillez entrer un nombre valide.")
        finally:
            print("Fin de la boucle.")

num = lire_clavier()
print(num)
```

<!-- #region id="kWujPW8MpC52" -->
### Break Stop Continue
<!-- #endregion -->

<!-- #region id="D1GDVD26pPwC" -->
- **break** arrête toute la boucle for/while.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U9WAJYLkpF5G" executionInfo={"status": "ok", "timestamp": 1703149903362, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f77ca2ef-fde7-4fff-967c-c272e020b7d4"
for i in [1,2,3,4,5]:
  if i == 3:
    break
  print(i)
```

<!-- #region id="s_2OQCwGpTpb" -->
- **continue** permet de sauter UNE itération. L'itération suivante aura quand même lieu.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0rj2r9SRpRnH" executionInfo={"status": "ok", "timestamp": 1703149923914, "user_tz": -60, "elapsed": 227, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ccc78635-3773-4dfb-bb9b-7c78f1787e41"
for i in [1,2,3,4,5]:
  if i == 3:
    continue
  print(i)
```

<!-- #region id="RYS_c6QkqFJL" -->
- **pass** ne fait absolument rien.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JtXbedEXpWrg" executionInfo={"status": "ok", "timestamp": 1703150280899, "user_tz": -60, "elapsed": 277, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7e7f20fd-fa19-47fa-e8f7-ad29da86c94e"
for i in [1,2,3,4,5]:
    if i == 3:
        pass
        print(i)
    else :
        print(i)
```

<!-- #region id="z_u_DDrKs3Ro" -->
## yield
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sZtZeIaaqU2D" executionInfo={"status": "ok", "timestamp": 1703151171458, "user_tz": -60, "elapsed": 268, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8cf90ef2-ee67-4ce0-dc4a-4565b45e3cbd"
def fibonacci():
    a, b = 0, 1
    i = 0
    while i < 10:
        yield a
        i += 1
        a, b = b, a + b

for num in fibonacci():
    print(num)
```

<!-- #region id="n_ihTBq913RP" -->
### Pickl
<!-- #endregion -->

<!-- #region id="hVk2shWf2AyE" -->
- **Store**
<!-- #endregion -->

```python id="1ADs-RImtzZs"
# saving a list into a .sav file
import pickle

lis = [4,5,6]
with open('test.sav', 'wb') as f:
  pickle.dump(lis, f)

'''
this will actually create a new file test.sav
this test.sav file will contain the list [4,5,6]
'''
```

<!-- #region id="h3EpFl2g17BO" -->
- **Load**
<!-- #endregion -->

```python id="S8fUJJGq17Vk"
# loading data from test.sav
import pickle

with open('test.sav', 'rb') as f:
  x = pickle.load(f)

print(x)    # [4,5,6]
```

<!-- #region id="_mpseaPP5Atm" -->
### correspondance de motifs,
simplifie une logique conditionnelle complexe avec différents types de structures de données :
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="40_Oky6f5A1m" executionInfo={"status": "ok", "timestamp": 1703154825797, "user_tz": -60, "elapsed": 257, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="76c7ad3e-d9ac-454c-d641-bec6e424bea5"
def process_data(data):
    match data:
        case 0:
            print("Received zero")
        case [x, y]:
            print(f"Received a list: {x}, {y}")
        case {"name": name, "age": age}:
            print(f"Received a dictionary: {name}, {age}")
        case _:
            print("Received something else")
print("cas 1")
process_data(0)         # Output: Received zero
print("cas 2")
process_data( [1, 2])   # Output: Received a list: 1, 2
print("cas 3")
process_data({"name": "John", "age": 25})  # Output: Received a dictionary: John, 25
print("cas 4")
process_data("Hello")   # Output: Received something else
```

```python colab={"base_uri": "https://localhost:8080/"} id="YwYjibpELNEX" executionInfo={"status": "ok", "timestamp": 1703242687506, "user_tz": -60, "elapsed": 429, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4cf1de74-225b-45f0-d6e8-cc73969309c8"
math.log(10)
```

```python id="Jg1YQB895Bag" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1703242899723, "user_tz": -60, "elapsed": 16863, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="030b4057-d35c-4397-cdd7-5245824bd2b3"
import multiprocessing
import time
import math

range_run = 10**7

def f(x):
    return x**2 - x*math.log(x+1)

def main():
    # Sans multiprocessing
    start_time = time.time()
    for x in range(range_run):
        f(x)
    end_time = time.time()
    print("Temps d'exécution sans multiprocessing :", end_time - start_time)

    # Avec multiprocessing
    start_time = time.time()
    with multiprocessing.Pool(4) as pool:
        results = pool.map(f, range(range_run))
    end_time = time.time()
    print("Temps d'exécution avec multiprocessing :", end_time - start_time)

if __name__ == "__main__":
    main()
```

```python id="BOycnh0KJUOT"

```
