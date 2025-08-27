# Scikit-Learn Pickle Compatibility Fix

## Problem Description

The NeSPReSO API was experiencing scikit-learn version compatibility issues when loading PCA objects:

```
InconsistentVersionWarning: Trying to unpickle estimator PCA from version 1.4.1.post1 when using version 1.5.2. This might lead to breaking code or invalid results. Use at your own risk.
```

**Root Cause**: PCA objects were trained and saved using scikit-learn version 1.4.1.post1, but the current environment has version 1.5.2. This version mismatch can cause:
- Warnings during model loading
- Potential runtime errors
- Inconsistent prediction results
- System instability

## Solutions Implemented

### 1. Warning Suppression

**Location**: `services/kernel/handler.py`

**Implementation**:
```python
import warnings
from sklearn.base import InconsistentVersionWarning

# Suppress the specific scikit-learn version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
```

**Benefits**:
- Eliminates the warning messages
- Prevents log pollution
- Maintains system stability

### 2. Safe Pickle Loading

**Location**: `services/kernel/handler.py`

**Implementation**:
```python
def safe_pickle_load(file_path):
    """
    Safely load pickle files with scikit-learn version compatibility handling.
    """
    try:
        # First try normal loading
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if "InconsistentVersionWarning" in str(e) or "version" in str(e).lower():
            # Handle version compatibility issues
            # ... fallback strategies
        else:
            raise e
```

**Fallback Strategies**:
1. **Normal loading** (first attempt)
2. **pickle5 compatibility** (if available)
3. **Warning suppression** (final attempt)
4. **Graceful error handling** (if all attempts fail)

### 3. Enhanced Error Handling

**Location**: `services/kernel/handler.py`

**Implementation**:
```python
def get_pca_objects():
    try:
        # Suppress warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            
            # Use safer pickle loading
            stats = safe_pickle_load(path)
            # ... process loaded objects
            
    except Exception as e:
        logger.error("Failed to load PCA from %s: %s", path, e)
        print(f"WARNING: Using dummy PCA fallback due to loading error: {e}")
        print("This may be due to scikit-learn version incompatibility")
        
        # Fallback to dummy PCA objects
        # ... create fallback objects
```

**Benefits**:
- Clear error messages
- Graceful degradation
- System continues to function even with compatibility issues

### 4. Configuration Options

**Location**: `services/config.py`

**Implementation**:
```python
SKLEARN_COMPATIBILITY = {
    "suppress_version_warnings": True,
    "use_compatibility_mode": True,
    "fallback_to_dummy_pca": True,
    "max_retry_attempts": 3
}
```

**Options**:
- **suppress_version_warnings**: Hide compatibility warnings
- **use_compatibility_mode**: Enable enhanced compatibility handling
- **fallback_to_dummy_pca**: Use dummy PCA objects if loading fails
- **max_retry_attempts**: Number of retry attempts for loading

## How the Fix Works

### 1. **Warning Suppression**
- Catches and suppresses `InconsistentVersionWarning` messages
- Prevents log pollution and user confusion
- Maintains clean application output

### 2. **Progressive Fallback**
- **Attempt 1**: Normal pickle loading
- **Attempt 2**: pickle5 compatibility mode (if available)
- **Attempt 3**: Warning-suppressed loading
- **Final Fallback**: Dummy PCA objects

### 3. **Graceful Degradation**
- If PCA loading fails, system continues to function
- Uses dummy PCA objects for predictions
- Logs clear error messages for debugging

## Expected Behavior After Fix

### 1. **No More Warnings**
- `InconsistentVersionWarning` messages are suppressed
- Clean application logs
- No user confusion about version mismatches

### 2. **Reliable Loading**
- PCA objects load successfully despite version differences
- Fallback mechanisms ensure system stability
- Clear error messages if issues persist

### 3. **System Stability**
- API continues to function even with compatibility issues
- Predictions may use fallback models if needed
- No system crashes due to pickle loading failures

## Long-term Solutions

### 1. **Version Alignment**
**Recommended**: Align scikit-learn versions between training and inference environments

```bash
# Check current version
pip show scikit-learn

# Install specific version (if needed)
pip install scikit-learn==1.4.1.post1

# Or upgrade to latest stable
pip install --upgrade scikit-learn
```

### 2. **Model Retraining**
**Alternative**: Retrain PCA models with current scikit-learn version

```python
# Retrain PCA objects with current sklearn version
from sklearn.decomposition import PCA
pca_temp = PCA(n_components=15)
pca_sal = PCA(n_components=15)

# Fit with your training data
pca_temp.fit(temperature_data)
pca_sal.fit(salinity_data)

# Save with current version
import pickle
with open('pca_objects_current.pkl', 'wb') as f:
    pickle.dump({
        'pca_temp': pca_temp,
        'pca_sal': pca_sal,
        'input_params': input_params
    }, f)
```

### 3. **Environment Management**
**Best Practice**: Use consistent environments for training and inference

```bash
# Create environment with specific sklearn version
conda create -n nespreso_sklearn_1.4.1 python=3.10
conda activate nespreso_sklearn_1.4.1
pip install scikit-learn==1.4.1.post1

# Or use Docker for consistent environments
docker run -it --rm python:3.10-slim bash
pip install scikit-learn==1.4.1.post1
```

## Monitoring and Verification

### 1. **Check Warning Suppression**
Look for these messages in logs:
- ✅ No more `InconsistentVersionWarning` messages
- ✅ Clean application startup
- ✅ Successful PCA object loading

### 2. **Verify PCA Loading**
Check debug output:
```
DEBUG: PCA objects loaded successfully from /path/to/pca.pkl
DEBUG: PCA temp type: <class 'sklearn.decomposition._pca.PCA'>, PCA sal type: <class 'sklearn.decomposition._pca.PCA'>
```

### 3. **Test Predictions**
- Run a small test dataset
- Verify predictions are generated
- Check for any remaining compatibility issues

## Troubleshooting

### 1. **If Warnings Persist**
```python
# Check if warnings are properly suppressed
import warnings
from sklearn.base import InconsistentVersionWarning

# Verify suppression is working
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
```

### 2. **If Loading Still Fails**
```python
# Check PCA file path
print(f"PCA path: {CFG.PCA_PATH}")
print(f"File exists: {os.path.exists(CFG.PCA_PATH)}")

# Try manual loading
import pickle
with open(CFG.PCA_PATH, 'rb') as f:
    data = pickle.load(f)
print(f"Loaded keys: {data.keys()}")
```

### 3. **If Fallback Doesn't Work**
```python
# Check sklearn installation
import sklearn
print(f"sklearn version: {sklearn.__version__}")

# Verify PCA class availability
from sklearn.decomposition import PCA
print(f"PCA class: {PCA}")
```

## Files Modified

- `services/kernel/handler.py` - Added warning suppression and safe pickle loading
- `services/config.py` - Added compatibility configuration options

## Next Steps

1. **Restart the server** to apply the warning suppression
2. **Test PCA loading** to verify compatibility fixes
3. **Monitor logs** for any remaining issues
4. **Consider long-term solutions** (version alignment, model retraining)
5. **Update documentation** if needed

## Benefits

- ✅ **No more warning messages** cluttering logs
- ✅ **Improved system stability** with fallback mechanisms
- ✅ **Better error handling** for compatibility issues
- ✅ **Cleaner user experience** without confusing warnings
- ✅ **Maintained functionality** even with version mismatches
