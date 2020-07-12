class FeatureVector():

    def __init__(self, *args):
        '''Takes any number of vectors. To be used as a vector for each dinenson of the state space.
        '''

        # Limits stores the bounds of each feature for each dimension of the state space
        self._limits = [*args]
        self._ndims = len(self._limits)
        self._nlims = [len(lim) for lim in self._limits]

    def limit_indices(self,s):
        '''Get the index the state belongs to for each of the limit dimensions. Use product to get an
        iterable of limit indices activated by the state
        '''

        lim_inds = []

        for dim,s_ in enumerate(s):
            ind = []
            for i,r in enumerate(self._limits[dim]):
                if s_ in r:
                    inds.append(i)
            lim_inds.append(inds)

        lim_inds = itertools.product(*lim_inds)
        return lim_inds

    def state_feature_vector(self, s):
        '''Get the limits indices for the state and convert this into the 1d index corresponding to the feature vector
        '''

        lim_inds = self.limit_indices(s)

        fv_indices = []

        for li in lim_inds:
            # Convert each limit index tuple, indication activated feature for each dimension, to a single index corresponding to that location in the feature vector
            fv_index = sum([li[i] * np.product(self._nlims[(i+1):]) for i in range(self._ndims-1)])
            fv_index = fv_index + lim_inds[-1]
            fv_indices.append(fv_index)

        # 1d feature vector
        fv = np.zeros(np.product(self._nlims))
        fv[fv_indices] = 1
        return fv

    @property
    def fv_size(self):
        fv_size = np.product(self._nlims)
        return fv_size
    
