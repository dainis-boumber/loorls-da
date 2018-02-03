import numpy as np
import time
from rlscore.learner import LeaveOneOutRLS
from rlscore.measure import ova_accuracy
from rlscore.utilities.multiclass import to_one_vs_all
from rlscore.learner import RLS
from rlscore.measure import accuracy


def load_mars(src, tgt):
    src_mat = np.loadtxt(open(src, "rb"), delimiter=",")
    tgt_mat = np.loadtxt(open(tgt, "rb"), delimiter=",")
    X_train = src_mat[:,:-1]
    y_train = src_mat[:,-1]
    X_test = tgt_mat[:,:-1]
    y_test = tgt_mat[:,-1]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def train_rls():
    X_train, Y_train, X_test, Y_test = load_mars('supernova-src.csv', 'supernova-tgt.csv')
    #Map labels from set {1,2,3} to one-vs-all encoding
    t = time.process_time()
    if np.count_nonzero(Y_train) >= len(Y_train) and np.count_nonzero(Y_test) >= len(Y_test):
        zerolabels = False
    else:
        zerolabels = True

    Y_train = to_one_vs_all(Y_train, zerolabels)
    Y_test = to_one_vs_all(Y_test, zerolabels)
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams=regparams, measure=ova_accuracy)
    P_test = learner.predict(X_test)
    #ova_accuracy computes one-vs-all classification accuracy directly between transformed
    #class label matrix, and a matrix of predictions, where each column corresponds to a class
    print("test set accuracy %f" %ova_accuracy(Y_test, P_test))
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_accuracy = 0.
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        #Leave-one-out cross-validation predictions, this is fast due to
        #computational short-cut
        P_loo = learner.leave_one_out()
        acc = accuracy(Y_train, P_loo)
        print("regparam 2**%d, loo-accuracy %f" % (log_regparam, acc))
        if acc > best_accuracy:
            best_accuracy = acc
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("Time: ", time.process_time() - t)
    print("best regparam %f with loo-accuracy %f" %(best_regparam, best_accuracy))
    print("test set accuracy %f" %accuracy(Y_test, P_test))

if __name__=="__main__":
    train_rls()