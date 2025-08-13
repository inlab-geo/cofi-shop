# Slime Mould Algorithm Integration into CoFI Core

## Executive Summary
This document outlines the plan for integrating the Slime Mould Algorithm (SMA) into the CoFI framework core. This is Stage 1 of a two-stage implementation strategy, focusing solely on the solver implementation, registration, and testing within the main CoFI repository.

## Background and Rationale

### Algorithm Overview
The Slime Mould Algorithm (SMA) is a bio-inspired stochastic optimizer proposed by Li et al. (2020) in *Future Generation Computer Systems*. It simulates the oscillation mode of slime mould in nature, using adaptive weights to model positive and negative feedback of propagation waves based on bio-oscillator patterns.

### CoFI Compatibility Analysis
SMA aligns perfectly with CoFI's model handling architecture:
- **Vector-based**: SMA operates on flat parameter vectors (no spatial awareness needed)
- **Gradient-free**: Only requires scalar objective function evaluations
- **Model-agnostic**: Can handle any problem where CoFI's objective function is defined
- **Spatial regularization compatible**: Works seamlessly with CoFI's spatial regularization utilities

### Implementation Choice: MEALPY Library
- **Repository**: https://github.com/thieu1995/mealpy
- **License**: MIT
- **Rationale**: Mature, well-maintained, pure Python implementation with both OriginalSMA and BaseSMA variants

## GitHub Workflow

### 1. Repository Setup
```bash
# Fork CoFI repository
# Navigate to https://github.com/inlab-geo/cofi → Fork

# Clone your fork
git clone https://github.com/YOUR_USERNAME/cofi.git
cd cofi

# Add upstream remote
git remote add upstream https://github.com/inlab-geo/cofi.git

# Sync with upstream
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Create feature branch
git checkout -b feature/add-mealpy-sma-solver
```

### 2. Tool Generation and Implementation

#### 2.1 Generate Tool Template
```bash
# Use CoFI's official helper script
python tools/new_inference_tool.py mealpy_sma

# Verify creation
ls src/cofi/tools/_mealpy_sma.py
```

#### 2.2 Core Implementation
File: `src/cofi/tools/_mealpy_sma.py`

```python
import numpy as np
from . import BaseInferenceTool, error_handler

class MealpySma(BaseInferenceTool):
    """Wrapper for Slime Mould Algorithm from mealpy library
    
    The Slime Mould Algorithm (SMA) is a bio-inspired optimizer that 
    mimics the oscillation mode of slime mould in nature. Following CoFI's
    vector abstraction principle, this tool operates purely on flattened
    parameter vectors while supporting spatial regularization at the utility level.
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
        - Bounds handling supports multiple CoFI formats
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
        
        optimizer_params = {
            "epoch": self._params.get("epoch", 100),
            "pop_size": self._params.get("pop_size", 50),
            "pr": self._params.get("pr", 0.03),
        }
        
        # Add initial population if initial_model provided
        if self.inv_problem.initial_model_defined():
            initial_flat = self.inv_problem.initial_model.flatten()
            optimizer_params["starting_positions"] = [initial_flat]
        
        if algorithm == "OriginalSMA":
            self._optimizer = SMA.OriginalSMA(**optimizer_params)
        elif algorithm == "BaseSMA":
            self._optimizer = SMA.BaseSMA(**optimizer_params)
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
        
        # Format results following CoFI conventions
        result = {
            "success": True,  # SMA doesn't provide explicit success flag
            "model": g_best.solution.reshape(self.inv_problem.model_shape),
            "objective": g_best.target.fitness,
            "n_iterations": self._params.get("epoch", 100),
        }
        
        # Add optimization history if available
        if hasattr(self._optimizer, "history"):
            result["history"] = {
                "objective": self._optimizer.history.list_global_best_fit,
            }
        
        return result

# CoFI tree registration comment
# CoFI -> Parameter estimation -> Optimization -> Gradient-free optimizers -> mealpy -> SMA
# description: Bio-inspired optimization mimicking slime mould foraging behavior
# documentation: https://mealpy.readthedocs.io/
```

### 3. Registration in CoFI

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

### 4. Testing Implementation

#### 4.1 Create Test File
Create `tests/cofi_tools/test_mealpy_sma.py` (copy from `_template.py` as recommended):

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
        assert result.objective < 1.0
    
    def test_algorithm_variants(self):
        """Test both OriginalSMA and BaseSMA"""
        for algo in ["OriginalSMA", "BaseSMA"]:
            self.options.set_params(algorithm=algo, epoch=10)
            inv = Inversion(self.problem, self.options)
            result = inv.run()
            assert result.success
    
    def test_objective_with_multiple_returns(self):
        """Test SMA handles objectives that return extra values"""
        def objective_with_grad(x):
            return np.sum(x**2), 2*x  # Returns (objective, gradient)
        
        problem = BaseProblem()
        problem.set_objective(objective_with_grad)
        problem.set_model_shape((3,))
        problem.set_bounds((-5, 5))
        
        self.options.set_params(epoch=10, pop_size=10)
        inv = Inversion(problem, self.options)
        result = inv.run()
        assert result.success
    
    def test_spatial_regularization_compatibility(self):
        """Test SMA works with spatial regularization"""
        def objective_with_spatial_reg(slowness):
            data_misfit = np.sum((slowness - 0.5)**2)
            reg_value = spatial_reg(slowness)
            return data_misfit + reg_value
        
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
        assert result.model.shape == model_shape
    
    def test_bounds_format_compatibility(self):
        """Test different bounds formats supported by CoFI"""
        # Test uniform bounds
        problem1 = BaseProblem()
        problem1.set_objective(lambda x: np.sum(x**2))
        problem1.set_model_shape((3,))
        problem1.set_bounds((-2, 2))
        
        inv1 = Inversion(problem1, self.options)
        result1 = inv1.run()
        assert result1.success
    
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
```

### 5. Quality Assurance

#### 5.1 Testing Commands
```bash
# Run SMA tests
pytest tests/cofi_tools/test_mealpy_sma.py -v

# Run full test suite
pytest tests/

# Generate coverage
coverage run -m pytest tests/cofi_tools/test_mealpy_sma.py
coverage report
coverage xml
```

#### 5.2 Code Quality
```bash
# Format code
black src/cofi/tools/_mealpy_sma.py tests/cofi_tools/test_mealpy_sma.py

# Check formatting
black --check .

# Linting
pylint src/cofi/tools/_mealpy_sma.py
```

### 6. Commit Strategy

```bash
# Implementation
git add src/cofi/tools/_mealpy_sma.py
git commit -m "feat: implement MealpySma inference tool

- Subclass BaseInferenceTool following CoFI guidelines
- Support OriginalSMA and BaseSMA variants
- Handle multiple CoFI bounds formats
- Vector abstraction with spatial regularization compatibility"

# Registration
git add src/cofi/tools/__init__.py
git commit -m "feat: register MealpySma in CoFI tools

- Import MealpySma class
- Add to inference_tools_table with aliases
- Register in CoFI tool hierarchy"

# Dependencies
git add pyproject.toml
git commit -m "deps: add mealpy>=3.0.0 for SMA support"

# Tests
git add tests/cofi_tools/test_mealpy_sma.py
git commit -m "test: add comprehensive MealpySma tests

- Copy from _template.py as recommended
- Test vector abstraction compatibility
- Test spatial regularization integration
- Test multiple bounds formats and edge cases"

# Quality fixes
git add .
git commit -m "style: apply formatting and address linting"
```

### 7. Pull Request

#### 7.1 Pre-PR Checklist
- [ ] All tests pass locally
- [ ] Code formatted with black
- [ ] Pylint score > 8.0
- [ ] Coverage > 80%
- [ ] Follows CoFI developer guide

#### 7.2 Create Pull Request
```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin feature/add-mealpy-sma-solver

# Create PR at GitHub
# Target: inlab-geo/cofi:main
# Source: YOUR_USERNAME/cofi:feature/add-mealpy-sma-solver
```

**PR Title**: `Add Slime Mould Algorithm (SMA) solver via MEALPY`

**PR Description**:
```markdown
## Summary
Integrates the Slime Mould Algorithm (SMA) into CoFI following the official developer guide. SMA is a bio-inspired metaheuristic optimizer providing gradient-free optimization for any CoFI problem.

## Implementation
- **Tool**: `MealpySma` subclassing `BaseInferenceTool`
- **Registration**: `mealpy.sma` and `mealpy.slime_mould` identifiers
- **Backend**: MEALPY library (OriginalSMA, BaseSMA variants)
- **Architecture**: Follows CoFI's vector abstraction principle

## Key Features
- ✅ Gradient-free optimization
- ✅ Spatial regularization compatibility
- ✅ Multiple bounds format support
- ✅ Algorithm variants (OriginalSMA, BaseSMA)
- ✅ Parallel execution modes

## Testing
- [x] Basic optimization functionality
- [x] Algorithm variants
- [x] Spatial regularization integration
- [x] Multiple bounds formats
- [x] Edge cases and error handling

## Dependencies
- Adds `mealpy>=3.0.0`

## Breaking Changes
None - purely additive.

## References
- Paper: Li et al. (2020) Future Generation Computer Systems
- MEALPY: https://github.com/thieu1995/mealpy
- CoFI Developer Guide: https://cofi.readthedocs.io/en/latest/developer/new_tool.html
```

## Success Criteria

- [ ] PR approved and merged into CoFI main
- [ ] All CoFI tests pass with new solver
- [ ] Tool appears in CoFI tool list
- [ ] Users can run: `options.set_tool("mealpy.sma")`
- [ ] Documentation builds successfully
- [ ] No breaking changes to existing functionality

## Timeline Estimate

- **Implementation**: 4-6 hours
- **Testing**: 2-3 hours  
- **QA & Review**: 2-3 hours
- **Total**: 8-12 hours

This Stage 1 implementation provides the foundation for Stage 2 (examples) while following CoFI's official development guidelines.