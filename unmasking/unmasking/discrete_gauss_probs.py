import numpy as np


# Define the interval
def get_probs(n: int):
    interval = np.arange(1, n + 1)

    # Compute mean and standard deviation
    mean = np.mean(interval)
    std_dev = np.std(interval)

    # Calculate Gaussian probabilities
    probabilities = np.exp(-0.5 * ((interval - mean) / std_dev) ** 2)
    # Normalize the probabilities
    probabilities /= probabilities.sum()
    # print(probabilities)
    # print(n)
    prob1 = probabilities[0 : int(n / 2)]
    prob2 = probabilities[int(n / 2) : n]
    # print("LEN of half sets of probs:", len(prob1), len(prob2))
    prob12 = [sum(prob1[:i]) for i in range(1, len(prob1) + 1)]
    fin_prob = prob12 + prob12[::-1]
    # print(fin_prob)
    return fin_prob


def get_probs_sig():
    interval = np.arange(1, 25)

    # Compute mean and standard deviation
    mean = np.mean(interval)
    std_dev = np.std(interval)

    # Calculate Gaussian probabilities
    probabilities = np.exp(-0.5 * ((interval - mean) / std_dev) ** 2)

    # Normalize the probabilities
    probabilities /= probabilities.sum()
    logits = np.log(probabilities + 1e-12)  # avoid log(0)
    temperature = 0.4

    sharp_probs = np.exp(logits / temperature)
    sharp_probs /= sharp_probs.sum()
    return sharp_probs


def get_probs_softmax():
    l1 = [float(i) for i in range(1, 13)]
    l2 = [float(i) for i in range(12, 0, -1)]
    l = l1 + l2
    l = np.array(l)
    probs = np.exp(l)
    probs /= probs.sum()
    return probs


if __name__ == "__main__":
    print(get_probs())
    print(get_probs_softmax())
