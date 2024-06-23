---
title: Measuring Divergence
layout: home
---

{% include latex_template.html %}
<div class="text-center">

Measuring Divergence And Presenting Code Example

</div>
{: .fs-5 .fw-700 .text-delta}


### 1. Squared Euclidean Distance (SED)

**Definition:**
The Squared Euclidean Distance between two points $ p $ and $ q $ in Euclidean space is the sum of the squared differences between corresponding coordinates.

**Formula:**
$$ \text{SED}(p, q) = \sum_{i} (p_i - q_i)^2 $$

**Python Code Example:**
```python
import numpy as np

def squared_euclidean_distance(p, q):
    return np.sum((p - q) ** 2)

# Example usage:
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
print("Squared Euclidean Distance:", squared_euclidean_distance(p, q))
```

### 2. Kullback-Leibler Divergence (Relative Entropy)

**Definition:**
The Kullback-Leibler Divergence measures how one probability distribution diverges from a second, expected probability distribution.

**Formula:**
$$ D_{\text{KL}}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} $$

**Python Code Example:**
```python
import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

# Example usage:
p = np.array([0.1, 0.4, 0.5])
q = np.array([0.2, 0.2, 0.6])
print("Kullback-Leibler Divergence:", kl_divergence(p, q))
```

### 3. f-Divergences

**Definition:**
f-Divergences are a class of divergences that generalize many common divergences, defined using a convex function $ f $.

**General Formula:**
$$ D_f(P \| Q) = \sum_{i} Q(i) f\left( \frac{P(i)}{Q(i)} \right) $$

**Python Code Example:**
```python
import numpy as np

def f_divergence(p, q, f):
    return np.sum(q * f(p / q))

# Example usage with a specific f (for example, the function used in KL divergence, f(t) = t * log(t)):
f = lambda t: t * np.log(t)
p = np.array([0.1, 0.4, 0.5])
q = np.array([0.2, 0.2, 0.6])
print("f-Divergence:", f_divergence(p, q, f))
```

### 4. Bregman Divergences

**Definition:**
Bregman Divergences are a class of divergences defined using a convex function $ \phi $.

**General Formula:**
$$ D_{\phi}(p, q) = \phi(p) - \phi(q) - \nabla \phi(q) \cdot (p - q) $$

**Python Code Example:**
```python
import numpy as np

def bregman_divergence(p, q, phi, grad_phi):
    return phi(p) - phi(q) - np.dot(grad_phi(q), (p - q))

# Example usage with a specific phi (for example, the squared Euclidean function):
phi = lambda x: np.sum(x ** 2)
grad_phi = lambda x: 2 * x
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
print("Bregman Divergence:", bregman_divergence(p, q, phi, grad_phi))
```

### Explanation:

1. **Squared Euclidean Distance (SED):**
   - Measures the sum of the squared differences between two points in space.
   - Useful for understanding the basic concept of divergence in a simple Euclidean space.

2. **Kullback-Leibler Divergence (Relative Entropy):**
   - Measures how one probability distribution differs from another.
   - Central to information theory and often used in machine learning to compare distributions.

3. **f-Divergences:**
   - A general class of divergences that includes many specific divergences.
   - Allows flexibility by using different functions $ f $ to define the divergence.

4. **Bregman Divergences:**
   - Another general class of divergences, defined using a convex function $ \phi $.
   - Used in various fields, including optimization and machine learning, for their nice mathematical properties.

These code examples provide a basic framework for understanding and calculating these divergences. Adjustments can be made depending on specific applications and data types.
