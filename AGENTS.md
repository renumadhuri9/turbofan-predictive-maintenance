# Project Conventions & Guidelines
- Tech Stack: Python 3.10+, scikit-learn, pandas, numpy, PyTorch (or sklearn MLP), Streamlit, joblib.
- Code Style: Write modular, production-grade Python scripts inside `src/`. Avoid placing core logic inside Jupyter Notebooks.
- Typing & Docstrings: Every function must have Python type hints and NumPy-style docstrings.
- Reproducibility: Fix all random seeds to `42`.
- Data Target: Predict 3 operational classes based on piecewise-clipped Remaining Useful Life (RUL):
  - Healthy: RUL > 30
  - Warning: 15 < RUL <= 30
  - Critical: RUL <= 15

