import numpy as np
import sklearn.metrics

def compute_mean_avg_prec(y_true, y_score):
    try:
        avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
        mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
    except ValueError:
        mean_a