import numpy as np

def get_means(data, p):
    
    X = np.zeros((data.shape[-1],
                 data.shape[0] * p.n_parcels))
    
    cnt = 0
    non_zero_unique = np.setdiff1d(np.unique(p.mask), np.array([0]))
    for i in non_zero_unique:
        means = np.mean(data[:,p.mask==i,:], axis=1)
        
        for j in range(data.shape[0]):
            X[:,cnt] = means[j]
            cnt += 1
        
    return X

def get_X(lh, rh, lh_p, rh_p):

    X_lh = get_means(lh, lh_p)
    X_rh = get_means(rh, rh_p)
    
    X = np.concatenate([X_lh, X_rh], axis=1)
    return X



