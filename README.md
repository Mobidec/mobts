# mobts

> Python package for preprocessing and imputing urban mobility time series data.

Designed for transport datasets such as bike counts, traffic loops, and station-based observations.

---

## What this package does

- Clean time series data:
 + Detects measurement errors
 + Flags and removes them 

- Imputes missing/invalid data based on a multi-tier method

---

## Installation

pip install mobts

---

## Example of running the code

from mobts import preprocess
from mobts import impute

# Step 1: clean raw data
pp = preprocess()
df_clean = pp.run(df)

# Step 2: impute missing values
imp = impute()
df_imputed = imp.run(df_clean)

---

### Functional examples

Full step-by-step examples are available in:
 - notebooks/demo_preprocessing_imputation.ipynb

## License

This project is licensed under the MIT License, which means it is freely usable for personal and commercial purposes. The MIT License is one of the most permissive open source licenses. It allows you to do almost anything with the source code, as long as you retain the original license notice and copyright information when redistributing the software or substantial portions of it. This license comes without any warranties, so the software is provided "as is." For more details, please refer to the included LICENSE file.