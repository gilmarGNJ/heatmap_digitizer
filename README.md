# heatmap_digitizer
heatmap_digitizer is a python package for extracting numerical data from heatmap images.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install heatmap_digitizer.

```bash
pip install heatmap_digitizer
```

## Usage
On terminal

```bash
# Pops up a window. After following instructions within the window, creates a csv file.
python3 -m heatmap_digitizer file_name.png

# Same as above, but at the end read and plot the csv file.
python3 -m heatmap_digitizer --plot file_name.png  # -p or --plot

# Uses an example image and plot the csv file at the end.
python3 -m heatmap_digitizer -e -p   # -e or --example

# Help option
python3 -m heatmap_digitizer --help
```
On Python
```python3
from heatmap_digitizer import HeatmapDigitizer

# Help function
print(help(HeatmapDigitizer))

# Pops up a window. After following instructions within the window, creates a csv file.
example = HeatmapDigitizer("file_name.png")
example.connect()

# Same as above, but at the end read and plot the csv file.
example = HeatmapDigitizer("file_name.png")
example.heatmap_plot = True
example.connect()

# Uses an example image and plot the csv file at the end.
example = HeatmapDigitizer("example")
example.connect()
```
After generating the csv file, you can read and plot it with Python.
```python3
# An example of how to read the generated csv file as a dataframe:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

fig, ax = plt.subplots(1, figsize=(800 / 96, 800 / 96), dpi=96)
example = pd.read_csv("file_name.csv", index_col=0)

example.columns = np.around(example.columns.astype(float), decimals=2)
example.index = np.around(example.index, decimals=2)

average = example.mean().mean()
heat_map = sb.heatmap(example, center=average, annot=False, cbar=True,
                      cbar_kws={'label': 'values'}, ax=ax, square=True)

ax.set_xlabel('x coordinate')
ax.set_ylabel('y coordinate')
ax.set_title('An example plot using the generated csv file.')
plt.show()
```

## License
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)