# Slime Mould Algorithm Integration Plan for CoFI

## Executive Summary
This document outlines the comprehensive plan for integrating the Slime Mould Algorithm (SMA) into the CoFI framework. After evaluating multiple implementations, we recommend using the MEALPY library as the backend for this integration.

## Background Research

### Algorithm Overview
The Slime Mould Algorithm (SMA) is a bio-inspired stochastic optimizer proposed by Li et al. (2020) in *Future Generation Computer Systems*. It simulates the oscillation mode of slime mould in nature, using adaptive weights to model positive and negative feedback of propagation waves based on bio-oscillator patterns.

### Key Characteristics
- **Nature-inspired**: Based on slime mould foraging behavior
- **Stochastic**: Global optimization without gradient requirements
- **Adaptive**: Uses vibration parameter for exploration/exploitation balance
- **Versatile**: Effective for continuous, discrete, and constrained optimization

### CoFI Compatibility Analysis
SMA aligns perfectly with CoFI's model handling architecture:
- **Vector-based**: SMA operates on flat parameter vectors (no spatial awareness needed)
- **Gradient-free**: Only requires scalar objective function evaluations
- **Model-agnostic**: Can handle any problem where CoFI's objective function is defined
- **Spatial regularization compatible**: Works seamlessly with CoFI's spatial regularization utilities since they operate at the utility level, not optimizer level

## Implementation Options Analysis

### Option 1: MEALPY Library (RECOMMENDED)
**Repository**: https://github.com/thieu1995/mealpy
- **Stars**: 1.1k
- **Last Updated**: July 2025
- **License**: MIT
- **Pros**: 
  - Mature, well-maintained implementation
  - Contains both OriginalSMA and improved BaseSMA
  - Built-in parallel execution support
  - Comprehensive documentation
  - Pure Python with NumPy
- **Cons**: 
  - Additional dependency for CoFI

### Option 2: Original Authors' Implementation
**Repository**: https://github.com/aliasgharheidaricom/Slime-Mould-Algorithm-A-New-Method-for-Stochastic-Optimization-
- **Language**: MATLAB
- **Pros**: Reference implementation
- **Cons**: Would require Python port

### Option 3: Custom Implementation
- **Pros**: No external dependencies
- **Cons**: Significant development effort, maintenance burden

## Recommended Approach: MEALPY Integration

### Rationale
1. **Consistency with CoFI philosophy**: Wrap established, well-tested libraries
2. **Reduced maintenance**: Leverage community-maintained implementation
3. **Feature-rich**: Access to both original and improved variants
4. **Performance**: Built-in parallelization support
5. **Documentation**: Well-documented with examples

## Implementation Plan

### Phase 1: Setup and Code Generation

#### 1.1 Git Workflow
```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR_USERNAME/cofi.git
cd cofi
git checkout -b feature/add-slime-mould-algorithm
```

#### 1.2 Generate Tool Template
```bash
python tools/new_inference_tool.py mealpy_sma
```
This creates: `src/cofi/tools/_mealpy_sma.py`

### Phase 2: Core Implementation

#### 2.1 Tool Class Structure
```python
# src/cofi/tools/_mealpy_sma.py

import numpy as np
from . import BaseInferenceTool, error_handler

class MealpySma(BaseInferenceTool):
    """Wrapper for Slime Mould Algorithm from mealpy library
    
    The Slime Mould Algorithm (SMA) is a bio-inspired optimizer that 
    mimics the oscillation mode of slime mould in nature. It uses 
    adaptive weights to simulate positive and negative feedback of 
    the propagation wave.
    """
    
    documentation_links = [
        "https://mealpy.readthedocs.io/",
        "https://doi.org/10.1016/j.future.2020.03.055",
        "https://github.com/thieu1995/mealpy"
    ]
    
    short_description = (
        "Slime Mould Algorithm - bio-inspired optimization based on "
        "slime mould foraging behavior (via mealpy)"
    )
    
    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "model_shape"}
    
    @classmethod
    def optional_in_problem(cls) -> dict:
        return {
            "bounds": None,
            "initial_model": None,
        }
    
    @classmethod
    def required_in_options(cls) -> set:
        return set()
    
    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "algorithm": "OriginalSMA",  # or "BaseSMA"
            "epoch": 100,
            "pop_size": 50,
            "pr": 0.03,
            "mode": "single",  # "thread", "process", "swarm"
            "n_workers": None,
            "seed": None,
            "verbose": False,
        }
    
    @classmethod
    def available_algorithms(cls) -> set:
        return {"OriginalSMA", "BaseSMA"}
    
    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._setup_problem_dict()
        self._setup_optimizer()
    
    def _setup_problem_dict(self):
        """Convert CoFI problem to mealpy format
        
        Following CoFI's vector abstraction principle:
        - model_shape used only for dimensionality (np.prod)
        - Bounds handling supports both uniform and per-parameter formats
        - No spatial awareness at optimizer level
        """
        from mealpy import FloatVar
        
        # Get problem dimensions - CoFI's vector abstraction
        model_shape = self.inv_problem.model_shape
        n_dims = np.prod(model_shape)
        
        # Setup bounds - handle multiple CoFI formats
        if self.inv_problem.bounds_defined():
            bounds = self.inv_problem.bounds
            
            # Handle different bound formats from CoFI
            if len(bounds) == 2 and np.isscalar(bounds[0]):
                # Uniform bounds: (-10, 10) for all parameters
                lb = [bounds[0]] * n_dims
                ub = [bounds[1]] * n_dims
            elif len(bounds) == n_dims and all(len(b) == 2 for b in bounds):
                # Per-parameter bounds: [(-10, 10), (-5, 15), ...]
                lb = [b[0] for b in bounds]
                ub = [b[1] for b in bounds]
            else:
                # Legacy format: bounds[0] = lowers, bounds[1] = uppers
                lb = bounds[0] if isinstance(bounds[0], (list, np.ndarray)) else [bounds[0]] * n_dims
                ub = bounds[1] if isinstance(bounds[1], (list, np.ndarray)) else [bounds[1]] * n_dims
        else:
            # Default wide bounds if not specified
            lb = [-100.0] * n_dims
            ub = [100.0] * n_dims
        
        # Create MEALPY problem dictionary
        self._problem_dict = {
            "obj_func": self._objective_wrapper,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min",
        }
    
    def _objective_wrapper(self, solution):
        """Wrapper to reshape solution for CoFI objective
        
        Following CoFI's model handling architecture:
        - SMA operates on flattened vectors (MEALPY requirement)
        - CoFI objective functions expect original model_shape
        - This wrapper bridges between the two representations
        - Only extracts scalar objective value (SMA is gradient-free)
        """
        model = solution.reshape(self.inv_problem.model_shape)
        obj_value = self.inv_problem.objective(model)
        
        # Ensure scalar return even if objective returns (value, gradient) tuple
        # This handles cases where users reuse gradient-based objective functions
        if isinstance(obj_value, (tuple, list)):
            return obj_value[0]
        return obj_value
    
    def _setup_optimizer(self):
        """Initialize the mealpy optimizer"""
        from mealpy.bio_based import SMA
        
        algorithm = self._params.get("algorithm", "OriginalSMA")
        
        if algorithm == "OriginalSMA":
            self._optimizer = SMA.OriginalSMA(
                epoch=self._params.get("epoch", 100),
                pop_size=self._params.get("pop_size", 50),
                pr=self._params.get("pr", 0.03),
            )
        elif algorithm == "BaseSMA":
            self._optimizer = SMA.BaseSMA(
                epoch=self._params.get("epoch", 100),
                pop_size=self._params.get("pop_size", 50),
                pr=self._params.get("pr", 0.03),
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @error_handler(
        when="solving optimization problem with Slime Mould Algorithm",
        context="calling mealpy SMA optimizer",
    )
    def __call__(self) -> dict:
        """Run the optimization"""
        mode = self._params.get("mode", "single")
        n_workers = self._params.get("n_workers", None)
        
        # Run optimization
        if mode == "single":
            g_best = self._optimizer.solve(self._problem_dict)
        else:
            g_best = self._optimizer.solve(
                self._problem_dict, 
                mode=mode, 
                n_workers=n_workers
            )
        
        # Format results
        result = {
            "success": True,  # SMA doesn't provide explicit success flag
            "model": g_best.solution.reshape(self.inv_problem.model_shape),
            "objective": g_best.target.fitness,
            "n_iterations": self._params.get("epoch", 100),
            "history": {
                "objective": self._optimizer.history.list_global_best_fit,
            } if hasattr(self._optimizer, "history") else None,
        }
        
        return result

# Register in CoFI tree
# CoFI -> Parameter estimation -> Optimization -> Gradient-free optimizers -> mealpy -> SMA
# description: Bio-inspired optimization mimicking slime mould foraging behavior
# documentation: https://mealpy.readthedocs.io/
```

### Phase 3: Integration

#### 3.1 Update `src/cofi/tools/__init__.py`
```python
# Add import
from ._mealpy_sma import MealpySma

# Add to __all__ list
__all__ = [
    # ... existing entries ...
    "MealpySma",
]

# Add to inference_tools_table
inference_tools_table = {
    # ... existing entries ...
    "mealpy.sma": MealpySma,
    "mealpy.slime_mould": MealpySma,  # alias
}
```

#### 3.2 Update Dependencies
Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "mealpy>=3.0.0",
]
```

### Phase 4: Testing

#### 4.1 Test File Structure
Create `tests/cofi_tools/test_mealpy_sma.py`:

```python
import numpy as np
import pytest
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.tools import MealpySma
from cofi.utils import QuadraticReg

class TestMealpySma:
    """Test suite for Slime Mould Algorithm integration
    
    Tests verify SMA's compatibility with CoFI's model handling architecture,
    including vector abstraction and spatial regularization support.
    """
    
    def setup_method(self):
        """Setup test fixtures"""
        # Simple quadratic problem
        self.problem = BaseProblem()
        self.problem.set_objective(lambda x: np.sum(x**2))
        self.problem.set_model_shape((5,))
        self.problem.set_bounds((-10, 10))
        
        self.options = InversionOptions()
        self.options.set_tool("mealpy.sma")
    
    def test_basic_optimization(self):
        """Test basic optimization works"""
        self.options.set_params(epoch=20, pop_size=10)
        inv = Inversion(self.problem, self.options)
        result = inv.run()
        
        assert result.success
        assert "model" in result
        assert "objective" in result
        assert result.objective < 1.0  # Should find near-optimal
    
    def test_algorithm_variants(self):
        """Test both OriginalSMA and BaseSMA"""
        for algo in ["OriginalSMA", "BaseSMA"]:
            self.options.set_params(algorithm=algo, epoch=10)
            inv = Inversion(self.problem, self.options)
            result = inv.run()
            assert result.success
    
    def test_objective_with_multiple_returns(self):
        """Test SMA handles objectives that return extra values (like gradients)
        
        Verifies CoFI's design principle: gradient-free optimizers should handle
        cases where users accidentally provide gradient-returning functions.
        """
        def objective_with_grad(x):
            # Function originally written for gradient-based methods
            return np.sum(x**2), 2*x  # Returns (objective, gradient)
        
        problem = BaseProblem()
        problem.set_objective(objective_with_grad)
        problem.set_model_shape((3,))
        problem.set_bounds((-5, 5))
        
        self.options.set_params(epoch=10, pop_size=10)
        inv = Inversion(problem, self.options)
        result = inv.run()  # Should work by extracting only scalar part
        assert result.success
        assert "model" in result
    
    def test_spatial_regularization_compatibility(self):
        """Test SMA works with spatial regularization (2D problem)
        
        Verifies CoFI's two-level architecture: spatial regularization operates
        at utility level while SMA remains vector-agnostic at optimizer level.
        """
        def objective_with_spatial_reg(slowness):
            # Simulate a 2D tomography problem with spatial smoothing
            data_misfit = np.sum((slowness - 0.5)**2)  # Simple data term
            reg_value = spatial_reg(slowness)  # Spatial regularization
            return data_misfit + reg_value
        
        # Setup 2D problem with spatial regularization
        model_shape = (10, 8)  # 2D grid
        spatial_reg = QuadraticReg(
            model_shape=model_shape,
            weighting_matrix="smoothing"
        )
        
        problem = BaseProblem()
        problem.set_objective(objective_with_spatial_reg)
        problem.set_model_shape(model_shape)
        problem.set_bounds((0.1, 1.0))
        
        self.options.set_params(epoch=20, pop_size=15)
        inv = Inversion(problem, self.options)
        result = inv.run()
        
        assert result.success
        assert result.model.shape == model_shape  # Proper reshaping
        # Spatial regularization should encourage smoothness
        assert np.std(result.model) < 0.5  # Relatively smooth solution
    
    def test_parallel_modes(self):
        """Test parallel execution modes"""
        self.options.set_params(mode="thread", n_workers=2, epoch=10)
        inv = Inversion(self.problem, self.options)
        result = inv.run()
        assert result.success
    
    def test_without_bounds(self):
        """Test optimization without explicit bounds"""
        problem_no_bounds = BaseProblem()
        problem_no_bounds.set_objective(lambda x: np.sum(x**2))
        problem_no_bounds.set_model_shape((3,))
        
        inv = Inversion(problem_no_bounds, self.options)
        result = inv.run()
        assert result.success
    
    def test_high_dimensional(self):
        """Test with high-dimensional problem"""
        problem_hd = BaseProblem()
        problem_hd.set_objective(lambda x: np.sum(x**2))
        problem_hd.set_model_shape((50,))
        problem_hd.set_bounds((-5, 5))
        
        self.options.set_params(epoch=50, pop_size=30)
        inv = Inversion(problem_hd, self.options)
        result = inv.run()
        assert result.success
        assert result.objective < 10.0
    
    def test_rosenbrock_benchmark(self):
        """Test on Rosenbrock function benchmark"""
        def rosenbrock(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        problem = BaseProblem()
        problem.set_objective(rosenbrock)
        problem.set_model_shape((10,))
        problem.set_bounds((-5, 10))
        
        self.options.set_params(epoch=100, pop_size=50)
        inv = Inversion(problem, self.options)
        result = inv.run()
        
        assert result.success
        # Check if close to global optimum (all ones)
        assert np.allclose(result.model, 1.0, atol=0.5)
    
    def test_bounds_format_compatibility(self):
        """Test different bounds formats supported by CoFI"""
        # Test uniform bounds
        problem1 = BaseProblem()
        problem1.set_objective(lambda x: np.sum(x**2))
        problem1.set_model_shape((3,))
        problem1.set_bounds((-2, 2))  # Uniform bounds
        
        inv1 = Inversion(problem1, self.options)
        result1 = inv1.run()
        assert result1.success
        
        # Test per-parameter bounds
        problem2 = BaseProblem()
        problem2.set_objective(lambda x: np.sum(x**2))
        problem2.set_model_shape((3,))
        problem2.set_bounds([(-1, 1), (-2, 2), (-3, 3)])  # Per-parameter bounds
        
        inv2 = Inversion(problem2, self.options)
        result2 = inv2.run()
        assert result2.success
```

#### 4.2 Testing Commands
```bash
# Run tests
pytest tests/cofi_tools/test_mealpy_sma.py -v

# With coverage
coverage run -m pytest tests/cofi_tools/test_mealpy_sma.py
coverage report
coverage xml

# Format check
black --check src/cofi/tools/_mealpy_sma.py tests/cofi_tools/test_mealpy_sma.py

# Linting
pylint src/cofi/tools/_mealpy_sma.py
```

### Phase 5: Documentation and Examples

#### 5.1 Example Notebook
Create `cofi-examples/examples/optimization_with_sma/slime_mould_optimization.ipynb`

Key sections demonstrating CoFI's model handling architecture:
1. **Introduction to SMA** - Bio-inspired optimization principles
2. **Simple 2D visualization** - Himmelblau function example showing vector abstraction
3. **Spatial regularization demo** - 2D tomography with smoothing regularization
4. **Comparison with other methods** - Border Collie, scipy optimizers on same problems  
5. **CoFI architecture showcase** - How SMA integrates with CoFI's "define once, solve many ways"
6. **Parallel execution** - MEALPY's built-in parallelization features
7. **Real-world geophysical example** - Seismic tomography or EM inversion

#### 5.2 Documentation Updates
- Add SMA to the inference tools documentation
- Include in the optimization methods comparison table
- Add to the CoFI gallery if applicable

### Phase 6: Quality Assurance

#### 6.1 Pre-commit Checklist
- [ ] All tests pass
- [ ] Code formatted with black
- [ ] Pylint score > 8.0
- [ ] Coverage > 80%
- [ ] Documentation complete
- [ ] Example notebook runs successfully

#### 6.2 Pull Request Structure
1. **Main PR**: Core implementation + tests
   - Target branch: `main` or `develop`
   - Include: Tool implementation, integration, tests
   
2. **Examples PR**: To `cofi-examples` repo
   - Example notebook
   - Data files if needed

### Phase 7: Maintenance Plan

#### 7.1 Version Compatibility
- Pin mealpy version in requirements
- Monitor mealpy updates for breaking changes
- Maintain compatibility matrix

#### 7.2 Performance Monitoring
- Benchmark against other gradient-free methods
- Track convergence rates on standard problems
- Document performance characteristics

## Risk Mitigation

### Potential Issues and Solutions

1. **Dependency Conflicts**
   - Solution: Use version pinning, test in isolated environments

2. **Performance Issues**
   - Solution: Leverage parallel modes, optimize problem formulation

3. **API Changes in mealpy**
   - Solution: Pin version, maintain compatibility layer if needed

4. **Numerical Stability**
   - Solution: Add input validation, bounds checking

## Success Criteria

- [ ] Integration passes all CoFI test suite
- [ ] Documentation is complete and clear
- [ ] Example demonstrates clear use cases
- [ ] Performance is competitive with other gradient-free methods
- [ ] Code follows CoFI style guidelines
- [ ] PR approved by maintainers

## Timeline Estimate

- Phase 1-2 (Core Implementation): 2-3 hours
- Phase 3 (Integration): 1 hour
- Phase 4 (Testing): 2-3 hours
- Phase 5 (Documentation): 2 hours
- Phase 6-7 (QA & Review): 1-2 hours

**Total Estimated Time**: 8-12 hours of development work

## Architectural Insights Integration

This SMA integration demonstrates key CoFI design principles:

### Vector Abstraction Principle
- SMA operates purely on flattened parameter vectors
- `model_shape` used only for dimensionality (`np.prod`)
- No spatial awareness at optimizer level
- Seamless compatibility with any problem where CoFI objective is defined

### Separation of Concerns
- **Optimizer Level**: SMA manages population, explores parameter space
- **Utility Level**: Spatial regularization operates via `QuadraticReg` 
- **User Level**: Forward solvers interpret physical meaning
- **Framework Level**: CoFI coordinates between levels

### "Define Once, Solve Many Ways"
- Same problem definition works with SMA, Border Collie, scipy methods
- User defines physics once, can try multiple optimization strategies
- Spatial regularization transparent to optimization algorithm choice

### Testing Strategy Validates Architecture
- Spatial regularization compatibility test verifies two-level design
- Multiple bounds format test confirms vector abstraction
- Objective wrapper test ensures proper CoFI integration

## References

1. Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020). Slime mould algorithm: A new method for stochastic optimization. *Future Generation Computer Systems*, 111, 300-323.
2. MEALPY Documentation: https://mealpy.readthedocs.io/
3. CoFI Developer Guide: https://cofi.readthedocs.io/en/latest/developer/new_tool.html
4. CoFI Model Handling Architecture: `../knowledge_base/cofi_model_handling_and_spatial_regularization.md`