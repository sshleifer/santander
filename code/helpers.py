import pandas as pd


def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    if not actual:
        return 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def is_valid_submission(path):
    df = pd.read_csv(path)
    assert df.columns.tolist() == ['ncodpers', 'added_products']
    assert df.shape[0] == 929615
    return True

