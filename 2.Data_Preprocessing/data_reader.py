# Pure Python for Data Science & Machine Learning
# Author: William
"""Reusable CSV reading and type-conversion utilities.

Provides helper functions that other modules import to load CSV data
and convert string columns to floats.
"""

from csv import reader


# --- Step 1: CSV Reading ---

def read_csv(filename):
    """Read a CSV file and return its contents as a list of rows.

    Args:
        filename: Path to the CSV file to read.

    Returns:
        A list of lists, where each inner list is one row of the CSV.
    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# --- Step 2: Type Conversion ---

def convert_string_to_float(dataset, column):
    """Convert string values in a given column to floats (skipping header).

    Args:
        dataset: The full dataset (including the header row at index 0).
        column:  The column index to convert.
    """
    data_rows = dataset[1:]
    for row in data_rows:
        row[column] = float(row[column].strip())


# --- Step 3: Main Execution ---

if __name__ == '__main__':
    filename = 'diabetes.csv'
    dataset = read_csv(filename)

    for i in range(len(dataset[0])):
        convert_string_to_float(dataset, i)

    print(dataset)
