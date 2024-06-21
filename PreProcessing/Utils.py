import numpy as np

def nanmean(arr, axis=None, where=None):
    """
    Compute the mean ignoring NaN values along the specified axis.

    Parameters:
    - arr: Input array containing NaN values.
    - axis: Axis or axes along which to operate. Default is None, meaning flattened input.
    - where: Mask array of the same shape as arr, indicating where NaNs should be ignored.

    Returns:
    - mean: Mean value(s) along the specified axis, ignoring NaNs.
    """
    if where is None:
        where = ~np.isnan(arr)  # Default to ignoring NaN values

    if axis is None:
        arr = arr.flatten()
        where = where.flatten()
        total_sum = 0
        count_valid = 0
        for num, valid in zip(arr, where):
            if valid and not np.isnan(num):
                total_sum += num
                count_valid += 1
        mean = total_sum / count_valid if count_valid > 0 else np.nan
        return mean
    else:
        arr = np.moveaxis(arr, axis, 0)
        where = np.moveaxis(where, axis, 0)
        means = []
        for slice, valid_slice in zip(arr, where):
            valid_values = slice[valid_slice & ~np.isnan(slice)]
            mean = np.mean(valid_values) if len(valid_values) > 0 else np.nan
            means.append(mean)
        return np.array(means)

def nanmedian(arr, axis=None, where=None):
    """
    Compute the median ignoring NaN values along the specified axis.

    Parameters:
    - arr: Input array containing NaN values.
    - axis: Axis or axes along which to operate. Default is None, meaning flattened input.
    - where: Mask array of the same shape as arr, indicating where NaNs should be ignored.

    Returns:
    - median: Median value(s) along the specified axis, ignoring NaNs.
    """
    arr = arr.astype('float64')

    if where is None:
        where = np.ones_like(arr, dtype=bool)  # Default to including all elements
    
    if axis is None:
        arr = arr.flatten()
        where = where.flatten()
        valid_values = arr[where & ~np.isnan(arr)]
        median = np.median(valid_values) if len(valid_values) > 0 else np.nan
        return median
    else:
        arr = np.moveaxis(arr, axis, 0)
        where = np.moveaxis(where, axis, 0)
        medians = []
        for slice, valid_slice in zip(arr, where):
            valid_values = slice[valid_slice & ~np.isnan(slice)]
            median = np.median(valid_values) if len(valid_values) > 0 else np.nan
            medians.append(median)
        return np.array(medians)

def nanmode(arr, axis=None, where=None):
    """
    Compute the mode ignoring NaN values along the specified axis.

    Parameters:
    - arr: Input array containing NaN values.
    - axis: Axis or axes along which to operate. Default is None, meaning flattened input.
    - where: Mask array of the same shape as arr, indicating where NaNs should be ignored.

    Returns:
    - mode: Mode value(s) along the specified axis, ignoring NaNs.
    """
    if where is None:
        where = ~np.isnan(arr)  # Default to ignoring NaN values

    if axis is None:
        arr = arr.flatten()
        where = where.flatten()
        counts = {}
        for num, valid in zip(arr, where):
            if valid and not np.isnan(num):
                if num in counts:
                    counts[num] += 1
                else:
                    counts[num] = 1
        max_count = 0
        mode = None
        for num, count in counts.items():
            if count > max_count:
                max_count = count
                mode = num
        return mode
    else:
        arr = np.moveaxis(arr, axis, 0)
        where = np.moveaxis(where, axis, 0)
        modes = []
        for slice, valid_slice in zip(arr, where):
            counts = {}
            for num, valid in zip(slice, valid_slice):
                if valid and not np.isnan(num):
                    if num in counts:
                        counts[num] += 1
                    else:
                        counts[num] = 1
            max_count = 0
            mode = None
            for num, count in counts.items():
                if count > max_count:
                    max_count = count
                    mode = num
            modes.append(mode)
        return np.array(modes)