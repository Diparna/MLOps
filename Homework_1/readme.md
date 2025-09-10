## Model Versions

| Version | File Name             | Description                     |
|---------|------------------------|---------------------------------|
| v1      | model_v1_single.py     | Simple Linear Regression using 1 feature (`x4`) |
| v2      | model_v2_dual.py       | Dual-feature Linear Regression using `x2`, `x4` |
| current | model_v2_dual.py       | Best-performing model so far    |

## Model Comparison

The current model (`model_v2_dual.py`) uses two predictor variables, `x2` and `x4`, and significantly performs better than the previous single-feature model (`model_v1_single.py`) that used only `x4`. While the first model achieved a test R² of 0.262 and RMSE of 8.80, the dual-feature model improved the R² to 0.485 and reduced RMSE to 7.35. This confirms that combining both `x2` and `x4` captures more variance in the target variable `y`.


### Run the latest model:
```bash
python Homework_1/model_v2_dual.py
