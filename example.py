from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
import numpy as np

if __name__ == '__main__':
    # data = np.loadtxt('mydata2.txt')
    # X_train = data[:, 0:8]
    # Y_train = data[:, 8]
    #
    # X_test = X_train
    # Y_test = Y_train

    random_engine = check_random_state(0)
    X_train = random_engine.uniform(-1, 1, 10000).reshape(-1, 1)
    Y_train = np.sinh(X_train)
    X_test = random_engine.uniform(-1, 1, 10000).reshape(-1, 1)
    Y_test = np.sinh(X_test)

    est_gp = SymbolicRegressor(population_size=5000, generations=1000,
                               stopping_criteria=0.01, p_crossover=0.6,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.1, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, n_jobs=1,
                               function_set=('add', 'mul', 'max'))

    est_gp.fit(X_train, Y_train)
    print(est_gp)
	with open('result.txt', 'w') as f：
	    f.write(est_gp)

    score_train = est_gp.score(X_train, Y_train)
    score_test = est_gp.score(X_test, Y_test)
    print(score_train, score_test)

