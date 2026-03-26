# 📋 Coding Style Guide

## 1. General Principles

- **Consistency:** Follow a single style across all languages and scripts.
- **Clarity over cleverness:** Code should be readable and self-explanatory.
- **Modularity:** Functions, classes, and components should have a single responsibility.
- **Commenting & Docstrings:** Explain *why*, not *what*.
- **Version control friendly:** Commit small, meaningful changes with clear messages.

---

## 2. Python (Backend & AI Scripts)

**Formatting:** Follow PEP 8

- 4-space indentation, max 79 characters per line.
- Blank lines to separate functions and classes.

**Naming conventions:**

| Type | Style |
| --- | --- |
| Functions / Variables | `snake_case` |
| Classes | `CamelCase` |
| Constants | `ALL_CAPS` |

**Type hints for clarity:**

```python
def compute_speed(position: tuple[float, float], time: float) -> float:
    """Compute speed given position (x, y) and time delta."""
    ...
```

**Imports:** Standard → Third-party → Local

```python
# Standard library
import os
import math

# Third-party
import torch
import numpy as np

# Local
from utils.metrics import compute_mae
```

**Configuration:** Use YAML or JSON files instead of hardcoding paths or hyperparameters.

---

## 3. JavaScript / TypeScript (Frontend)

- **Variables / Functions:** `camelCase`
- **Classes / Components:** `PascalCase`
- **Formatting:** Use Prettier (2-space indent)
- **ES6+ features:** `const`/`let`, arrow functions, destructuring
- **Component Guidelines:** Keep components <200 lines; clear props and state

Example:

```tsx
const calculateSpeed = (distance: number, time: number): number => {
  return distance / time;
};

export default function SpeedDisplay({ speed }: { speed: number }) {
  return <div>Current speed: {speed}</div>;
}
```

---

## 4. Kotlin (Android)

- **Naming:** `camelCase` for functions/variables, `PascalCase` for classes
- **Line Length:** Max 120 characters
- **Immutability:** Prefer `val` over `var`
- **Organize imports automatically**
- **Use extension functions** for cleaner code

Example:

```kotlin
fun Double.toKmH(): Double = this * 3.6

val speed = 10.0.toKmH()
```

---

## 5. AI / ML Scripts

- Separate **preprocessing, model definition, training, and evaluation** into modules.
- Clear file structure:

```
models/
  osnet_multi_branch.py
data/
  preprocess_tracking.py
utils/
  metrics.py
```

- Use config files (YAML/JSON) for all hyperparameters.

---

## 6. Linters & Formatters

| Language | Linter | Formatter |
| --- | --- | --- |
| Python | flake8 | black |
| JS/TS | eslint | prettier |
| Kotlin | ktlint | IDE formatter |

---

## 7. Recommended Directory Structure

```
courtvision_project/
├─ backend/
│  ├─ models/
│  ├─ utils/
│  ├─ main.py
├─ frontend/
│  ├─ components/
│  ├─ pages/
├─ android_app/
│  ├─ src/
│  ├─ build.gradle
├─ configs/
│  ├─ default.yaml
├─ tests/
```

---

## 8. Docstrings & Comments

- **Python:** Triple-quoted docstrings (Google style)

```python
def compute_speed(position: tuple[float, float], time: float) -> float:
    """
    Compute speed given position and time delta.

    Args:
        position (tuple[float, float]): (x, y) coordinates
        time (float): time delta in seconds

    Returns:
        float: speed in meters per second
    """
    ...
```

- **JS/TS:** `/** ... */` JSDoc style
- **Kotlin:** `/** ... */` for functions/classes
