# SMA CoFI Example Implementation Plan

## Executive Summary
This document outlines Stage 2 of the SMA integration: creating an example notebook demonstrating the Slime Mould Algorithm within the CoFI examples repository, specifically in the `test_functions_for_optimization` directory.

**Prerequisites**: Stage 1 (SMA core integration) must be completed and merged.

## Objective

Create `slime_mould_algorithm_optimization.ipynb` in the `test_functions_for_optimization` directory, following the pattern established by `modified_himmelblau.ipynb`.

## GitHub Workflow

### 1. Repository Setup
```bash
# Fork cofi-examples repository (after Stage 1 is merged)
# Navigate to https://github.com/inlab-geo/cofi-examples → Fork

# Clone your fork
git clone https://github.com/YOUR_USERNAME/cofi-examples.git
cd cofi-examples

# Add upstream remote
git remote add upstream https://github.com/inlab-geo/cofi-examples.git

# Sync with upstream
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Create feature branch
git checkout -b feature/add-sma-test-function-example
```

### 2. Environment Setup
```bash
# Install CoFI with SMA support (after Stage 1 merge)
pip install git+https://github.com/inlab-geo/cofi.git

# Or if using local development version:
pip install -e /path/to/cofi

# Install additional requirements
pip install -r envs/requirements.txt

# Verify SMA is available
python -c "
import cofi
options = cofi.InversionOptions()
options.set_tool('mealpy.sma')
print('SMA available!')
"
```

### 3. Notebook Development

#### 3.1 Study Existing Pattern
```bash
cd examples/test_functions_for_optimization

# Study the existing example structure
ls -la
# modified_himmelblau.ipynb
# meta.yml

# Examine the notebook structure
jupyter notebook modified_himmelblau.ipynb
```

#### 3.2 Create SMA Example Notebook
Create `slime_mould_algorithm_optimization.ipynb` following this structure:

**Cell 1: Title and Setup**
```markdown
# Slime Mould Algorithm Optimization with CoFI

[![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/test_functions_for_optimization/slime_mould_algorithm_optimization.ipynb)

> If you are running this notebook locally, make sure you've followed [steps here](https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally)
to set up the environment.
```

**Cell 2: Environment Setup**
```python
# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi
```

**Cell 3: Imports**
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)
```

**Cell 4: Theory Section**
```python
# Display theory on SMA
from IPython.display import display, Markdown

theory_content = """
## Slime Mould Algorithm (SMA)

The Slime Mould Algorithm is a bio-inspired metaheuristic optimizer that mimics the oscillation mode of slime mould in nature. It uses adaptive weights to model positive and negative feedback of propagation waves based on bio-oscillator patterns.

### Key Characteristics:
- **Nature-inspired**: Based on slime mould foraging behavior
- **Gradient-free**: Only requires objective function evaluations
- **Adaptive**: Uses vibration parameter for exploration/exploitation balance
- **Versatile**: Effective for continuous optimization problems

### CoFI Integration:
SMA integrates seamlessly with CoFI's architecture, operating on flattened parameter vectors while supporting spatial regularization at the utility level.
"""

display(Markdown(theory_content))
```

**Cell 5-7: Test Function Definitions**
```python
def modified_himmelblau(x): 
    """Modified Himmelblau function with additional regularization term"""
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2+((x[0]-3)**2+(x[1]-2)**2)

def rosenbrock(x):
    """Classic Rosenbrock function for benchmarking"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    """Rastrigin function - highly multimodal test function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
```

**Cell 8: Visualization Function**
```python
def plot_2d_function(func, xlim=(-6, 6), ylim=(-6, 6), title=""):
    """Plot 2D test function"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    
    X = np.arange(xlim[0], xlim[1], 0.1)
    Y = np.arange(ylim[0], ylim[1], 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = func([X, Y])
    
    im = ax.pcolor(X, Y, Z, norm=colors.LogNorm(vmin=10**-2, vmax=Z.max()))
    ax.scatter(3, 2, color='red', label="Global minimum", marker='*', s=100)
    ax.set_title(title)
    ax.legend()
    fig.colorbar(im)
    plt.show()

# Plot the test function
plot_2d_function(modified_himmelblau, title="Modified Himmelblau Function")
```

**Cell 9-11: SMA Optimization Examples**
```python
# Example 1: Modified Himmelblau with SMA
print("=== SMA on Modified Himmelblau Function ===")

# Define CoFI problem
problem_himmelblau = BaseProblem()
problem_himmelblau.name = "Modified Himmelblau Function"
problem_himmelblau.set_objective(modified_himmelblau)
problem_himmelblau.set_model_shape((2,))
problem_himmelblau.set_bounds((-6, 6))

# Define inversion options
options_sma = InversionOptions()
options_sma.set_tool("mealpy.sma")
options_sma.set_params(
    algorithm="OriginalSMA",
    epoch=100,
    pop_size=30,
    pr=0.03
)

# Run inversion
inv_sma = Inversion(problem_himmelblau, options_sma)
result_sma = inv_sma.run()

print(f"SMA Result: {result_sma.model}")
print(f"SMA Objective: {result_sma.objective}")
print(f"True minimum: [3, 2]")
print(f"Distance from true minimum: {np.linalg.norm(result_sma.model - [3, 2])}")
```

**Cell 12-14: Comparison with Other Methods**
```python
# Compare SMA with Border Collie Optimization
print("\n=== Comparison: SMA vs Border Collie ===")

# Border Collie setup
options_bc = InversionOptions()
options_bc.set_tool("cofi.border_collie_optimization")
options_bc.set_params(number_of_iterations=100)

inv_bc = Inversion(problem_himmelblau, options_bc)
result_bc = inv_bc.run()

# Compare results
results_comparison = {
    "SMA": {"model": result_sma.model, "objective": result_sma.objective},
    "Border Collie": {"model": result_bc.model, "objective": result_bc.objective}
}

for method, res in results_comparison.items():
    distance = np.linalg.norm(res["model"] - [3, 2])
    print(f"{method:15} | Model: [{res['model'][0]:6.3f}, {res['model'][1]:6.3f}] | "
          f"Obj: {res['objective']:8.3f} | Distance: {distance:.3f}")
```

**Cell 15-16: High-Dimensional Test**
```python
# Example 2: High-dimensional Rosenbrock
print("\n=== SMA on High-Dimensional Rosenbrock (10D) ===")

problem_rosenbrock = BaseProblem()
problem_rosenbrock.set_objective(rosenbrock)
problem_rosenbrock.set_model_shape((10,))
problem_rosenbrock.set_bounds((-5, 10))

options_sma_hd = InversionOptions()
options_sma_hd.set_tool("mealpy.sma")
options_sma_hd.set_params(
    algorithm="BaseSMA",
    epoch=200,
    pop_size=50
)

inv_sma_hd = Inversion(problem_rosenbrock, options_sma_hd)
result_sma_hd = inv_sma_hd.run()

print(f"10D Rosenbrock Result: {result_sma_hd.model}")
print(f"Distance from optimum (all ones): {np.linalg.norm(result_sma_hd.model - 1.0)}")
print(f"Objective value: {result_sma_hd.objective}")
```

**Cell 17-18: Algorithm Variants Comparison**
```python
# Compare SMA algorithm variants
print("\n=== SMA Algorithm Variants Comparison ===")

variants = ["OriginalSMA", "BaseSMA"]
variant_results = {}

for variant in variants:
    options_variant = InversionOptions()
    options_variant.set_tool("mealpy.sma")
    options_variant.set_params(
        algorithm=variant,
        epoch=50,
        pop_size=20
    )
    
    inv_variant = Inversion(problem_himmelblau, options_variant)
    result_variant = inv_variant.run()
    variant_results[variant] = result_variant
    
    distance = np.linalg.norm(result_variant.model - [3, 2])
    print(f"{variant:12} | Model: [{result_variant.model[0]:6.3f}, {result_variant.model[1]:6.3f}] | "
          f"Obj: {result_variant.objective:8.3f} | Distance: {distance:.3f}")
```

**Cell 19: CoFI Architecture Demonstration**
```python
# Demonstrate CoFI's "define once, solve many ways" principle
print("\n=== CoFI Architecture: Same Problem, Multiple Solvers ===")

solvers = [
    ("mealpy.sma", {"algorithm": "OriginalSMA", "epoch": 50, "pop_size": 20}),
    ("cofi.border_collie_optimization", {"number_of_iterations": 50}),
    ("scipy.optimize.minimize", {"method": "Powell"})
]

for solver_name, solver_params in solvers:
    try:
        options = InversionOptions()
        options.set_tool(solver_name)
        options.set_params(**solver_params)
        
        inv = Inversion(problem_himmelblau, options)  # Same problem definition!
        result = inv.run()
        
        distance = np.linalg.norm(result.model - [3, 2])
        print(f"{solver_name:25} | Model: [{result.model[0]:6.3f}, {result.model[1]:6.3f}] | "
              f"Distance: {distance:.3f}")
    except Exception as e:
        print(f"{solver_name:25} | Error: {str(e)[:50]}")
```

**Cell 20: Performance Analysis**
```python
# Convergence analysis
if hasattr(result_sma, 'history') and result_sma.history:
    plt.figure(figsize=(10, 6))
    plt.plot(result_sma.history['objective'])
    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Value')
    plt.title('SMA Convergence History')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
else:
    print("Convergence history not available for this SMA implementation")
```

**Cell 21: Parallel Execution Demo**
```python
# Demonstrate parallel execution capabilities
print("\n=== Parallel Execution Demo ===")

options_parallel = InversionOptions()
options_parallel.set_tool("mealpy.sma")
options_parallel.set_params(
    algorithm="OriginalSMA",
    epoch=30,
    pop_size=20,
    mode="thread",
    n_workers=2
)

import time
start_time = time.time()
inv_parallel = Inversion(problem_himmelblau, options_parallel)
result_parallel = inv_parallel.run()
parallel_time = time.time() - start_time

print(f"Parallel execution time: {parallel_time:.2f} seconds")
print(f"Result: {result_parallel.model}")
```

**Cell 22: Watermark**
```python
# Watermark
watermark_list = ["cofi", "numpy", "matplotlib", "mealpy"]
for pkg in watermark_list:
    try:
        pkg_var = __import__(pkg)
        print(pkg, getattr(pkg_var, "__version__"))
    except ImportError:
        print(f"{pkg}: not available")
```

#### 3.3 Create Metadata File
Create `meta.yml` (if needed, following existing pattern):
```yaml
title: "Slime Mould Algorithm Optimization"
description: "Demonstrates SMA optimization on test functions using CoFI"
author: "CoFI Team"
level: "beginner"
tags: ["optimization", "metaheuristic", "test-functions", "sma"]
```

### 4. Testing and Validation

#### 4.1 Test Notebook Execution
```bash
# Test notebook runs without errors
jupyter nbconvert --to notebook --execute slime_mould_algorithm_optimization.ipynb

# Check output notebook
ls slime_mould_algorithm_optimization.nbconvert.ipynb
```

#### 4.2 Verify Results
- SMA finds near-optimal solutions for test functions
- Comparisons with other methods work correctly
- Visualizations display properly
- No errors in any cells

### 5. Commit and Push

```bash
git add examples/test_functions_for_optimization/slime_mould_algorithm_optimization.ipynb

# Add meta.yml if created
git add examples/test_functions_for_optimization/meta.yml

git commit -m "feat: add SMA optimization example for test functions

- Create slime_mould_algorithm_optimization.ipynb
- Demonstrate SMA on Himmelblau and Rosenbrock functions
- Compare with Border Collie and scipy optimizers
- Showcase CoFI's 'define once, solve many ways' principle
- Include high-dimensional and parallel execution examples"

git push origin feature/add-sma-test-function-example
```

### 6. Pull Request

#### 6.1 Create Pull Request
```bash
# Navigate to https://github.com/YOUR_USERNAME/cofi-examples
# Create PR targeting: inlab-geo/cofi-examples:main
```

**PR Title**: `Add SMA optimization example to test_functions_for_optimization`

**PR Description**:
```markdown
## Summary
Adds a comprehensive example notebook demonstrating the Slime Mould Algorithm (SMA) on optimization test functions, following the pattern established by `modified_himmelblau.ipynb`.

## Contents
- **File**: `examples/test_functions_for_optimization/slime_mould_algorithm_optimization.ipynb`
- **Test Functions**: Modified Himmelblau, Rosenbrock, Rastrigin
- **Comparisons**: SMA vs Border Collie vs scipy optimizers
- **Features**: Algorithm variants, high-dimensional problems, parallel execution

## Key Demonstrations
- ✅ SMA optimization on 2D visualization problems
- ✅ High-dimensional optimization (10D Rosenbrock)
- ✅ Algorithm variants (OriginalSMA vs BaseSMA)
- ✅ CoFI architecture: "define once, solve many ways"
- ✅ Performance comparison with existing CoFI tools
- ✅ Parallel execution capabilities

## Dependencies
- Requires CoFI with SMA integration (merged in main repo)
- Uses existing cofi-examples environment

## Testing
- [x] Notebook executes without errors
- [x] All visualizations display correctly
- [x] SMA finds near-optimal solutions
- [x] Comparisons work as expected
- [x] Follows cofi-examples style guidelines

## Integration
- Placed in `test_functions_for_optimization` as suggested
- Follows pattern of existing examples in that directory
- Compatible with existing environment setup
```

## Expected Outcomes

After successful merge:

1. **Example Location**: `examples/test_functions_for_optimization/slime_mould_algorithm_optimization.ipynb`
2. **Accessibility**: Available through CoFI examples documentation
3. **Educational Value**: Demonstrates SMA usage patterns for new users
4. **Integration**: Shows how SMA fits within CoFI's ecosystem

## Success Criteria

- [ ] Notebook executes successfully in clean environment
- [ ] SMA demonstrates competitive performance on test functions
- [ ] Clear educational progression from simple to complex examples
- [ ] Proper integration with existing examples directory
- [ ] PR approved and merged
- [ ] Example appears in CoFI examples documentation

## Timeline Estimate

- **Notebook Development**: 3-4 hours
- **Testing & Refinement**: 1-2 hours
- **Documentation & PR**: 1 hour
- **Total**: 5-7 hours

This Stage 2 implementation provides users with practical guidance on using SMA while showcasing its integration within CoFI's architecture.