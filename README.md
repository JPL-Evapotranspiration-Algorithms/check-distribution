# `check-distribution`

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Python utility for analyzing and logging the statistical distribution of values in arrays and raster images during model runs. Particularly useful for debugging geospatial data processing pipelines and monitoring intermediate variables in scientific computing workflows.

## Author

**Gregory H. Halverson** (they/them)  
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)  
NASA Jet Propulsion Laboratory 329G

## Features

- 📊 **Statistical Analysis**: Automatically computes min, max, mean, and NaN proportion
- 🎨 **Color-Coded Logging**: Uses colored output to highlight warnings and important values
- 🗺️ **Raster Support**: Native support for `Raster` objects from the `rasters` package
- ⚠️ **Blank Detection**: Optionally raises errors when outputs are completely blank
- 📈 **Smart Output**: Detailed distribution for categorical data, statistics for continuous data
- 🕒 **Temporal Tracking**: Optional date and location parameters for tracking values over time

## Installation

### From PyPI

```bash
pip install check-distribution
```

### From Source

```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/check-distribution.git
cd check-distribution
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from check_distribution import check_distribution

# Analyze a simple array
data = np.random.rand(100, 100)
check_distribution(data, "temperature")
```

## Detailed Usage

### Basic Usage

The `check_distribution` function analyzes array or raster data and logs statistical information:

```python
from check_distribution import check_distribution
import numpy as np

# Create sample data
temperature = np.random.normal(25, 5, (100, 100))

# Basic check
check_distribution(temperature, "temperature")
```

**Output:**
```
variable temperature min: 10.234 mean: 25.123 max: 39.876 nan: 0.00% (nan)
```

### Working with Categorical Data

For arrays with fewer than 10 unique values, the function displays counts for each value:

```python
# Land cover classification
land_cover = np.array([
    [1, 1, 2, 3],
    [1, 2, 2, 3],
    [2, 2, 3, 3],
    [3, 3, 3, 0]
])

check_distribution(land_cover, "land_cover")
```

**Output:**
```
variable land_cover (int64) with 4 unique values
* 0: 1
* 1: 3
* 2: 5
* 3: 7
```

### Adding Temporal Context

Track variable distributions over time:

```python
from datetime import date

# Daily temperature data
for day in range(1, 8):
    temp = np.random.normal(20 + day, 3, (100, 100))
    check_distribution(
        temp, 
        "temperature",
        date_UTC=date(2025, 1, day)
    )
```

**Output:**
```
variable temperature on 2025-01-01 min: 15.234 mean: 21.123 max: 27.876 nan: 0.00% (nan)
variable temperature on 2025-01-02 min: 16.123 mean: 22.456 max: 28.234 nan: 0.00% (nan)
...
```

### Adding Spatial Context

Include location information for spatial data:

```python
# Process multiple tiles
tiles = ["h08v05", "h09v05", "h08v06"]

for tile in tiles:
    ndvi = np.random.rand(2400, 2400)
    check_distribution(
        ndvi,
        "NDVI",
        date_UTC=date(2025, 6, 15),
        target=tile
    )
```

**Output:**
```
variable NDVI on 2025-06-15 at h08v05 min: 0.001 mean: 0.512 max: 0.999 nan: 0.00% (nan)
variable NDVI on 2025-06-15 at h09v05 min: 0.003 mean: 0.498 max: 0.997 nan: 2.34% (nan)
...
```

### Working with Raster Objects

The function natively supports `Raster` objects from the `rasters` package:

```python
from rasters import Raster

# Load a raster
dem = Raster("elevation.tif")

# Check distribution (automatically uses raster's nodata value)
check_distribution(dem, "elevation")
```

### Detecting Blank Outputs

By default, the function raises `BlankOutputError` if an array is completely NaN:

```python
from check_distribution import check_distribution, BlankOutputError

# This will raise an error
try:
    blank_data = np.full((100, 100), np.nan)
    check_distribution(blank_data, "missing_data")
except BlankOutputError as e:
    print(f"Error: {e}")
```

**Output:**
```
Error: variable missing_data is a blank image
```

To allow blank outputs:

```python
blank_data = np.full((100, 100), np.nan)
check_distribution(blank_data, "missing_data", allow_blank=True)
```

### Handling Arrays with NaN Values

The function automatically detects and reports NaN proportions:

```python
# Create data with NaN values
data = np.random.rand(100, 100)
data[data < 0.2] = np.nan

check_distribution(data, "partial_data")
```

**Output:**
```
variable partial_data min: 0.201 mean: 0.612 max: 0.999 nan: 20.15% (nan)
```

High NaN proportions (>50%) are highlighted in yellow, and 100% NaN triggers a red warning.

### Detecting All-Zero Arrays

Arrays with all zeros are automatically flagged:

```python
zeros = np.zeros((100, 100))
check_distribution(zeros, "empty_result", allow_blank=True)
```

**Output (warning):**
```
variable empty_result all zeros min: 0.000 mean: 0.000 max: 0.000 nan: 0.00% (nan)
```

### Complete Example: Processing Pipeline

```python
import numpy as np
from datetime import date
from check_distribution import check_distribution, BlankOutputError

def process_satellite_image(tile_id, acquisition_date):
    """Example processing pipeline with distribution checks."""
    
    # Load raw data
    raw_dn = np.random.randint(0, 16384, (2400, 2400))
    check_distribution(raw_dn, "raw_DN", acquisition_date, tile_id)
    
    # Convert to reflectance
    reflectance = raw_dn * 0.0001
    check_distribution(reflectance, "reflectance", acquisition_date, tile_id)
    
    # Calculate NDVI
    nir = reflectance * np.random.uniform(0.8, 1.2, reflectance.shape)
    red = reflectance * np.random.uniform(0.1, 0.3, reflectance.shape)
    ndvi = (nir - red) / (nir + red + 1e-10)
    
    try:
        check_distribution(ndvi, "NDVI", acquisition_date, tile_id)
    except BlankOutputError:
        print(f"Warning: NDVI calculation failed for {tile_id}")
        return None
    
    # Apply cloud mask
    cloud_mask = np.random.choice([0, 1], reflectance.shape, p=[0.15, 0.85])
    check_distribution(cloud_mask, "cloud_mask", acquisition_date, tile_id)
    
    ndvi_masked = ndvi.copy()
    ndvi_masked[cloud_mask == 1] = np.nan
    check_distribution(ndvi_masked, "NDVI_masked", acquisition_date, tile_id)
    
    return ndvi_masked

# Run pipeline
result = process_satellite_image("h08v05", date(2025, 6, 15))
```

## API Reference

### `check_distribution()`

```python
check_distribution(
    image: Union[Raster, np.ndarray],
    variable: str,
    date_UTC: Union[date, str] = None,
    target: str = None,
    allow_blank: bool = False
)
```

**Parameters:**

- **`image`** (Raster or np.ndarray): The array or raster to analyze
- **`variable`** (str): Name of the variable for logging purposes
- **`date_UTC`** (date or str, optional): Date associated with the data
- **`target`** (str, optional): Location or tile identifier
- **`allow_blank`** (bool, optional): If False (default), raises `BlankOutputError` for completely NaN arrays

**Raises:**

- **`BlankOutputError`**: When the array is completely NaN and `allow_blank=False`

**Behavior:**

- **< 10 unique values**: Lists each unique value with its count
- **≥ 10 unique values**: Shows min, mean, max, and NaN percentage
- **All zeros**: Logs a warning
- **Negative minimums**: Highlighted in red
- **Non-positive maximums**: Highlighted in red
- **High NaN proportion (>50%)**: Highlighted in yellow
- **Complete NaN (100%)**: Highlighted in red and may raise error

### `BlankOutputError`

Exception raised when an array is completely blank (all NaN values).

```python
from check_distribution import BlankOutputError
```

## Dependencies

- **`numpy`**: Array operations and statistics
- **`colored-logging`**: Colored console output
- **`rasters`**: Raster data support

## Use Cases

### Scientific Computing
- Monitor intermediate variables in physics simulations
- Validate numerical solver outputs
- Track convergence in iterative algorithms

### Geospatial Processing
- Verify satellite image calibration
- Monitor vegetation indices calculation
- Validate atmospheric correction results
- Track data quality across tiles and dates

### Machine Learning
- Inspect feature distributions during preprocessing
- Monitor activation values in neural networks
- Validate data augmentation pipelines

### Data Quality Control
- Detect data loading errors
- Identify processing failures
- Monitor data completeness over time

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=check_distribution tests/

# Run specific test file
pytest tests/test_check_distribution.py
```

### Building

```bash
# Build distribution
make build

# Install locally
pip install -e .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0

## Citation

If you use this software in your research, please cite:

```bibtex
@software{halverson2025checkdistribution,
  author = {Halverson, Gregory H.},
  title = {check-distribution: A Python utility for analyzing array distributions},
  year = {2025},
  url = {https://github.com/JPL-Evapotranspiration-Algorithms/check-distribution}
}
```

## Related Projects

- **[rasters](https://github.com/JPL-Evapotranspiration-Algorithms/rasters)**: Geospatial raster processing
- **[colored-logging](https://github.com/gregory-halverson/colored-logging)**: Enhanced logging with colors
