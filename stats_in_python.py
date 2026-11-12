import sys
import math


def _numeric_values(numbers:list)->list[float]:
    """Return the numeric values, excluding NaN entries."""
    values = []
    for number in numbers:
        value = float(number)
        if not math.isnan(value):
            values.append(value)
    if not values:
        raise ValueError("No numeric values are available for the requested statistic.")
    return values


def load_data(filepath:str, column:int)->tuple[list[float], int]:
    # Load the selected tab-delimited column as floats and count its entries.
    if not isinstance(column, int) or column < 0:
        raise ValueError("Column number must be a non-negative integer.")

    numbers = []
    total_values = 0

    with open(filepath, "r") as infile:
        for line in infile:
            if not line.strip():
                continue

            fields = line.rstrip("\r\n").split("\t")
            try:
                value_text = fields[column].strip()
            except IndexError:
                sys.exit(f"Column {column} is not present in an input row.")

            total_values += 1
            try:
                value = float(value_text)
            except ValueError:
                continue

            if not math.isnan(value):
                numbers.append(value)

    if not numbers:
        raise ValueError(f"Column {column} contains no numeric values.")

    return numbers, total_values


def calculate_average(numbers:list)->float:
    values = _numeric_values(numbers)
    return float(sum(values) / len(values))


def get_max_and_min(numbers:list)->tuple[float, float]:
    values = _numeric_values(numbers)
    return max(values), min(values)


def calculate_variance(numbers:list)->float:
    values = _numeric_values(numbers)
    if len(values) == 1:
        return 0.0

    average = sum(values) / len(values)
    squared_differences = sum((value - average) ** 2 for value in values)
    return float(squared_differences / (len(values) - 1))


def calculate_std_dev(numbers:list)->float:
    return math.sqrt(calculate_variance(numbers))


def calculate_median(number:list)->float:
    values = sorted(_numeric_values(number))
    middle = len(values) // 2

    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2)
