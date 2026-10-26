import pytest
import pandas as pd
import numpy as np
from src.models import map_rul_to_class, preprocess_data

def test_map_rul_to_class():
    # Healthy: RUL > 30 (mapped to 0)
    assert map_rul_to_class(31) == 0
    assert map_rul_to_class(100) == 0

    # Warning: 15 < RUL <= 30 (mapped to 1)
    assert map_rul_to_class(30) == 1
    assert map_rul_to_class(16) == 1

    # Critical: RUL <= 15 (mapped to 2)
    assert map_rul_to_class(15) == 2
    assert map_rul_to_class(0) == 2

def test_preprocess_data():
    # Create dummy train df
    # Engine 1 runs for 3 cycles
    # Engine 2 runs for 2 cycles
    train_data = {
        "unit_number": [1, 1, 1, 2, 2],
        "time_in_cycles": [1, 2, 3, 1, 2],
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    train_df = pd.DataFrame(train_data)

    # Create dummy test df
    # Engine 1 runs for 2 cycles (true RUL = 5)
    # Engine 2 runs for 1 cycle (true RUL = 10)
    test_data = {
        "unit_number": [1, 1, 2],
        "time_in_cycles": [1, 2, 1],
        "feature_1": [0.6, 0.7, 0.8]
    }
    test_df = pd.DataFrame(test_data)

    # True RUL series
    true_rul = pd.Series([5, 10])

    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df, true_rul)

    # For train_df:
    # unit 1 max_cycles = 3. RULs: 2, 1, 0 -> classes: 2, 2, 2
    # unit 2 max_cycles = 2. RULs: 1, 0 -> classes: 2, 2
    np.testing.assert_array_equal(y_train, np.array([2, 2, 2, 2, 2]))

    # For test_df:
    # unit 1 max_cycles = 2. true_rul = 5. cycle 1 RUL: 5 + (2 - 1) = 6. cycle 2 RUL: 5 + (2 - 2) = 5.
    # cycle 1 RUL=6 -> class 2. cycle 2 RUL=5 -> class 2.
    # unit 2 max_cycles = 1. true_rul = 10. cycle 1 RUL: 10 + (1 - 1) = 10.
    # cycle 1 RUL=10 -> class 2.
    np.testing.assert_array_equal(y_test, np.array([2, 2, 2]))

    # Check features shapes
    assert X_train.shape == (5, 1) # only feature_1 is left
    assert X_test.shape == (3, 1)
