# CoFI Model Handling Architecture and Spatial Regularization Design

## Executive Summary

This document explains how CoFI handles model vectors and the sophisticated separation of concerns between the optimization framework and spatial regularization utilities. While CoFI generally treats models as abstract vectors, spatial regularization operators represent a crucial exception that demonstrates the framework's elegant design.

## Core Principle: Vector Abstraction with Spatial Awareness

### The General Rule: Models as Abstract Vectors

CoFI follows a **"define once, solve many ways"** philosophy where:

1. **Optimization tools** (scipy, emcee, SMA, Border Collie, etc.) only see flat parameter vectors
2. **Model vectors** are passed without semantic meaning about what elements represent
3. **Forward solvers/objective functions** are responsible for interpreting physical meaning
4. **CoFI framework** manages optimization without understanding physics

#### Evidence from Border Collie Implementation

```python
# cofi/src/cofi/tools/_cofi_border_collie_optimization.py
# Line 60: Converts model_shape to flat size
self.mod_size = np.prod(inv_problem.model_shape)

# Lines 185, 190: Passes position vectors directly to objective
self.pack_fit[i] = self.inv_problem.objective(self.pack[i].pos)
self.flock_fit[i] = self.inv_problem.objective(self.flock[i].pos)
```

#### Pattern in All Optimizers

```python
# CoFI sees this:
model_vector = [0.1, 0.5, 2.3, 1.7, 0.9]  # Just numbers!

# Forward solver interprets meaning:
def forward_model(model_vec):
    # User's responsibility to give meaning:
    velocity = model_vec.reshape((nx, nz))  # 2D velocity field
    # OR
    density, resistivity = model_vec[:3], model_vec[3:]  # Different properties
    # OR  
    layer_thicknesses = model_vec  # 1D layered model
    
    # Forward solver computes predictions based on physics
    predictions = physics_engine(velocity)
    return predictions
```

### The Exception: Spatial Regularization

**Regularization operators DO need spatial context** to create meaningful smoothness constraints.

#### Evidence from FMM Tomography Example

From `cofi-examples/examples/fmm_tomography/fmm_tomography_regularization_discussion.ipynb`:

```python
# Spatial regularization setup
model_shape = fmm.model_shape  # 2D spatial grids (e.g., (20, 20))

reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
    model_shape=model_shape,    # THIS IS THE KEY!
    weighting_matrix="smoothing"
)
```

#### How Spatial Operators Are Created

From `cofi/src/cofi/utils/_reg_lp_norm.py`:

```python
# Lines ~238-250: CoFI examines model_shape to create spatial operators
if np.size(self.model_shape) == 2 and np.ndim(self.model_shape) == 1:  # 2D model
    nx = self.model_shape[0]  # Grid points in x direction  
    ny = self.model_shape[1]  # Grid points in y direction
    
    # Create 2D finite difference operators using findiff
    d_dx = findiff.FinDiff(0, 1, order)  # x direction
    d_dy = findiff.FinDiff(1, 1, order)  # y direction
    
    # Build spatial derivative matrix that operates on flattened vectors
    self._weighting_matrix = d_dx.matrix((nx, ny)) + d_dy.matrix((nx, ny))
elif np.size(self.model_shape) == 1:  # 1D model
    d_dx = findiff.FinDiff(0, 1, order)
    self._weighting_matrix = d_dx.matrix((self.model_size,))
```

## Two-Level Architecture Design

CoFI maintains the **"define once, solve many ways"** principle through a sophisticated two-level architecture:

### Level 1: Framework Level (Vector Agnostic)
- **Optimizers**: Only see flat parameter vectors
- **Model passing**: Always as `np.array([param1, param2, ..., paramN])`
- **Bounds handling**: Applied per parameter, no spatial awareness
- **Convergence**: Based on parameter changes, not spatial patterns

### Level 2: Utility Level (Spatially Aware)
- **Regularization operators**: Understand spatial structure via `model_shape`
- **Matrix construction**: Create spatial derivative operators
- **Vector operations**: Still operate on flattened vectors but with spatial meaning

### The Brilliant Interface

```python
# User defines problem with spatial context:
model_shape = (nx, ny)  # 2D velocity grid
smoothing_reg = QuadraticReg(model_shape=model_shape, weighting_matrix="smoothing")

# CoFI's regularization utility builds spatial operator:
# - Knows that model_vector[i*ny + j] corresponds to grid point (i,j)
# - Creates matrix that computes spatial derivatives across neighbors
# - BUT STILL OPERATES ON FLATTENED VECTORS!

# Optimizer (like SMA, Border Collie) still only sees:
model_vector = [v00, v01, v02, ..., v10, v11, v12, ...]  # Flattened

# When regularization is applied:
reg_value = smoothing_reg(model_vector)  # Operates on flat vector
# Internally: uses spatial derivative matrix, returns scalar
```

## Example: Tomography with Spatial Regularization

From the FMM tomography example:

```python
def objective_func(slowness, sigma, reg):
    # Forward modeling (user understands physics)
    ttimes = forward_solver(slowness.reshape(model_shape))
    
    # Data misfit (vector operations)
    residual = obstimes - ttimes
    data_misfit = residual.T @ residual / sigma**2
    
    # Spatial regularization (operates on flat vector with spatial awareness)
    model_reg = reg(slowness)  # reg knows spatial structure
    
    return data_misfit + model_reg

# CoFI setup
problem = BaseProblem()
problem.set_objective(objective_func, args=[sigma, reg_smoothing])
problem.set_model_shape(model_shape)  # Informs regularization

# Any optimizer can solve this
options.set_tool("scipy.optimize.minimize")  # Uses gradients
# OR
options.set_tool("cofi.border_collie_optimization")  # Direct search
# OR  
options.set_tool("mealpy.sma")  # Our new SMA implementation
```

## Design Philosophy: Separation of Concerns

### What Each Component Knows

1. **Forward Solver/Objective Function**:
   - Physics: what model parameters represent
   - Coordinate systems: how to interpret spatial layouts
   - Units and scaling: physical meaning of parameter values

2. **Regularization Utilities**:
   - Spatial relationships: neighboring grid points, smoothness
   - Mathematical operators: derivatives, norms, covariances
   - Matrix construction: efficient sparse matrix operations

3. **Optimization Framework**:
   - Parameter spaces: bounds, constraints, transformations
   - Search strategies: gradients, metaheuristics, sampling
   - Convergence: when to stop, how to report results

4. **Individual Optimizers**:
   - Algorithm specifics: update rules, population management
   - Hyperparameters: learning rates, population sizes
   - Nothing about physics or spatial structure!

## Implications for New Tool Development

### For Direct Search Methods (like SMA)

```python
class MealpySma(BaseInferenceTool):
    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "model_shape"}  # model_shape for dimensionality only
    
    def _objective_wrapper(self, solution):
        # Reshape to give back spatial structure for user's functions
        model = solution.reshape(self.inv_problem.model_shape)
        return self.inv_problem.objective(model)  # User handles everything else
```

### For Gradient-Based Methods

```python
class GradientOptimizer(BaseInferenceTool):
    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "gradient", "model_shape"}
    
    def _gradient_wrapper(self, solution):
        model = solution.reshape(self.inv_problem.model_shape)
        grad = self.inv_problem.gradient(model)
        return grad.flatten()  # Optimizer needs flat gradient vector
```

## Key Insights

1. **Model vectors are always flattened** for optimization algorithms
2. **Spatial context exists at the utility level** through `model_shape`
3. **Regularization operators bridge** between spatial awareness and vector operations
4. **Optimizers remain completely agnostic** about physical meaning
5. **The reshape/flatten pattern** is crucial for maintaining this separation

## Benefits of This Design

1. **Optimizer Portability**: Any optimization algorithm can solve any problem
2. **Spatial Flexibility**: Same regularization works for 1D, 2D, 3D problems
3. **Physical Generality**: Framework doesn't lock in specific physics
4. **Mathematical Consistency**: All operations ultimately on flat vectors
5. **User Simplicity**: Define problem once, try multiple solvers

## Future Considerations

When implementing new inference tools or regularization utilities:

- **Maintain vector abstraction** at optimizer level
- **Leverage model_shape** for spatial operations in utilities only
- **Always reshape appropriately** when passing to user functions
- **Document spatial assumptions** clearly in regularization operators
- **Test with different model_shape configurations** (1D, 2D, 3D)

This architecture enables CoFI's powerful "define once, solve many ways" capability while supporting sophisticated spatial regularization for geophysical inverse problems.