# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the CoFI Shop monorepo containing three main projects:
1. **cofi** - Core CoFI (Common Framework for Inference) library for geophysical inversions
2. **cofi-examples** - Example notebooks and scripts demonstrating CoFI usage  
3. **cofi-resources** - Documentation and educational resources

## Key Development Commands

### Testing and Quality Checks
```bash
# Run tests (from cofi/ directory)
pytest

# Run tests with coverage
coverage run -m pytest; coverage report; coverage xml

# Format code with black
black --check .  # Check formatting
black .           # Apply formatting

# Run linting
pylint cofi
```

### Documentation
```bash
# Build documentation (from cofi/docs/)
make html         # Build HTML docs
make clean_all    # Clean all generated docs
```

### Installation and Environment
```bash
# Create conda environment with all dependencies (from cofi-examples/)
mamba env create -f envs/environment.yml
mamba activate cofi_env

# Install cofi for development (from cofi/)
pip install -e .
```

### Running Examples
```bash
# Launch Jupyter Lab for interactive examples (from cofi-examples/)
jupyter-lab
```

## Architecture

### Core Components (cofi/)

The main CoFI library is structured around these key classes in `src/cofi/`:

- **BaseProblem** (`_base_problem.py`) - Defines the inversion problem with functions like objective, data_misfit, forward model, jacobian, etc.
- **InversionOptions** (`_inversion_options.py`) - Configures which inference tool/algorithm to use and its parameters
- **Inversion** (`_inversion.py`) - Orchestrates the inversion by combining BaseProblem and InversionOptions
- **InversionResult** - Stores and provides access to inversion results

### Inference Tools (cofi/tools/)

Multiple backend solvers are supported:
- scipy (lstsq, optimize.minimize, sparse.lstsq)
- PyTorch optimizers
- emcee (MCMC sampler)
- BayesBay, NeighPy (specialized geophysical tools)
- Custom CoFI implementations (simple Newton, border collie optimization)

### Utilities (cofi/utils/)

- Regularization classes (L2, Lp norm, model covariance)
- Multiple runs support (InversionPool)
- Prior distributions

### Examples Structure

Examples are organized in `cofi-examples/examples/` with:
- Jupyter notebooks demonstrating concepts
- Python scripts for command-line execution
- Supporting libraries (`*_lib.py` files)
- Data files in `data/` subdirectories

## Development Workflow

1. **Making Changes**: Follow existing code style, use type hints where appropriate
2. **Testing**: Write tests in `tests/` mirroring source structure
3. **Documentation**: Update docstrings and relevant example notebooks
4. **Formatting**: Run `black` before committing
5. **Building**: Package uses setuptools with `pyproject.toml` configuration

## Important Patterns

- Problems are defined by setting functions/properties on BaseProblem instances
- The framework automatically derives related functions (e.g., objective from data_misfit + regularization)
- Inference tools are accessed via string identifiers (e.g., "scipy.optimize.minimize")
- Results are standardized across different backend solvers

## Technical Knowledge Base

For detailed technical information, algorithm implementations, and research notes that extend beyond this overview, refer to the `knowledge_base/` directory. This contains:
- Algorithm integration plans and implementation details
- Research findings and technical deep-dives
- Architecture decisions and design patterns
- Performance optimization notes

### Key Documents
- `cofi_model_handling_and_spatial_regularization.md` - Comprehensive analysis of how CoFI handles model vectors, the separation between optimization and spatial regularization, and the two-level architecture design
- `claude_conversation_mealpy_logging_fix.md` - Documentation of mealpy integration and logging improvements