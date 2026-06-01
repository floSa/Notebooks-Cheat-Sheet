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

```python id="751GaOsSwheb"

```

<!-- #region id="VQo9EMN_wnxr" -->
Les Kolmogorov-Arnold Networks (KAN) sont un cadre théorique pour l'approximation de fonctions continues sur un intervalle donné. Elles se basent sur les travaux de Kolmogorov et Arnold, qui ont démontré des résultats fondamentaux sur la représentation des fonctions continues.

### Théorème de Kolmogorov

Le théorème de Kolmogorov énonce que toute fonction continue multivariée peut être représentée comme une superposition de fonctions continues univariées. Plus formellement, il affirme que pour toute fonction continue $ f: [0,1]^n→R $, il existe des fonctions continues univariées $ϕ_i$  et $ψ_{ij}$ telles que :

$f(x_1,x_2,...,x_n)=\sum_{i=1}^{2n+1}​ϕ_i (\sum_{j=1}^{n} ​ψ_{ij}(x_j))$

### Contribution de Arnold

Arnold a amélioré ce résultat en montrant que les fonctions $ψ_{ij}$​ peuvent être choisies de manière plus structurée et avec moins de dépendances.

### Application aux réseaux de neurones

Les Kolmogorov-Arnold Networks (KAN) sont des réseaux de neurones qui exploitent ce théorème pour construire des modèles capables d'approximer des fonctions continues. La structure d'un KAN est généralement composée de trois couches :

1.  **Entrée** : La couche d'entrée prend les variables $f(x_1,x_2,...,x_n)$​.
2.  **Transformation intermédiaire** : Chaque entrée est transformée par un ensemble de fonctions univariées $​ψ_{ij} $​.
3.  **Superposition** : Les sorties des transformations intermédiaires sont combinées par des fonctions univariées $ϕ_i$​ pour produire la sortie finale.

### Avantages des KAN

*   **Théoriquement fondé** : Les KAN reposent sur un théorème mathématique robuste, garantissant une approximation universelle pour des fonctions continues.
*   **Simplicité des transformations univariées** : La décomposition en transformations univariées simplifie l'analyse et la formation du réseau.

### Limites des KAN

*   **Complexité de construction** : Trouver les fonctions $ϕ_i$ ​ et $ψ_{ij}$​ adaptées peut être complexe et non trivial dans la pratique.
*   **Scalabilité** : Pour des problèmes de grande dimension, la mise en œuvre des KAN peut devenir impraticable en raison de la croissance rapide du nombre de fonctions nécessaires.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CY0tr8cq0V3w" executionInfo={"status": "ok", "timestamp": 1721398285815, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4d81553d-8291-4b52-d4a9-d38120f96272"
import numpy as np

def psi(x, j):
    """Fonction de transformation univariée ψ_ij"""
    return np.sin(x + j)

def phi(y, i):
    """Fonction de superposition univariée φ_i"""
    return np.sum(y) + i

def kolmogorov_arnold_network(X):
    """Réseau Kolmogorov-Arnold approchant une fonction continue"""
    n = X.shape[1]
    output = 0
    for i in range(2 * n + 1):
        sum_psi = np.zeros(X.shape[0])
        for j in range(n):
            sum_psi += psi(X[:, j], j)
        output += phi(sum_psi, i)
    return output

# Exemple d'utilisation
X = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
result = kolmogorov_arnold_network(X)
print(result)

```

```python id="mv_UWaAUVS3z"

```
