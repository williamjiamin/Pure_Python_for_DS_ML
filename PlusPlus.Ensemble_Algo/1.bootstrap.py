# Pure Python for Data Science & Machine Learning
# Author: William
"""Bootstrap sampling demonstration.

Shows how the estimated mean converges to the true population mean
as the number of bootstrap samples increases.
"""

from random import seed
from random import randrange


# --- Step 1: Bootstrap Subsample ---

def subsample(dataset, ratio=1.0):
    """Create a bootstrap subsample from the dataset with replacement.

    Args:
        dataset: The original dataset (list of rows).
        ratio: Fraction of the dataset to sample (default 1.0).

    Returns:
        A list containing the sampled rows.
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# --- Step 2: Mean Calculation ---

def mean(numbers):
    """Compute the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numeric values.

    Returns:
        The mean value (float).
    """
    return sum(numbers) / float(len(numbers))


# --- Step 3: Run Bootstrap Experiment ---

if __name__ == '__main__':
    seed(1)
    dataset = [[randrange(10)] for i in range(20)]

    ratio = 0.10
    for size in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        sample_means = list()
        for i in range(size):
            sample = subsample(dataset, ratio)
            sample_mean = mean([row[0] for row in sample])
            sample_means.append(sample_mean)
        print("When sample is [%d], the estimated mean is [%.3f]" % (size, mean(sample_means)))

    print("The real mean of our dataset is [%.3f]" % mean([row[0] for row in dataset]))
