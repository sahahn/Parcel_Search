from Parcels import Parcels
from ROIs import get_X
from ML import get_score
from nilearn import surface

import nibabel.freesurfer.io as io
import numpy as np
import pandas as pd


class Surf_Individual():

    def __init__(self, geo, n_parcels_l, n_parcels_u, mut_probs,
                 medial_wall_inds=None, r_state=None):

        self.mut_probs = mut_probs

        if r_state is None:
            r_state = np.random.RandomState()
        self.r_state = r_state

        # Init the parcels + generate
        self.parcels_lh = Parcels(geo, n_parcels_l, n_parcels_u,
                                  medial_wall_inds, r_state)
        self.parcels_rh = Parcels(geo, n_parcels_l, n_parcels_u,
                                  medial_wall_inds, r_state)

        self.score = None
        self.mutate_flag = False
        self.age = 0

    def Evaluate(self, data, targets, models):

        if self.score is None:

            self.scores = []
            self.raw_scores = []

            for d in range(len(data)):

                # Get ROI values
                X = get_X(data[d][0], data[d][1], self.parcels_lh, self.parcels_rh)

                # Eval w/ ML, for each target using this data
                per_target_scores = []
                for t in range(len(targets[d])):
                    per_target_scores.append(get_score(X, targets[d][t], models[d][t]))

                self.scores.append(np.mean(per_target_scores))
                self.raw_scores.append(per_target_scores)

            # Could also save as raw...
            self.score = np.mean(self.scores)

        # Regardless of if evaluated up to age by 1
        self.age += 1

    def Mutate(self):

        # Choose which parcel(s) to mutate
        to_mutate = self.r_state.choice([['lh'], ['rh'], ['lh', 'rh']], p=[1/3, 1/3, 1/3])

        if 'lh' in to_mutate:
            self.parcels_lh = self.mutate(self.parcels_lh)
        if 'rh' in to_mutate:
            self.parcels_rh = self.mutate(self.parcels_rh)

        # Reset score to None if mutated + flag
        self.score = None
        self.mutate_flag = False

    def mutate(self, parcels):

        # First, reset
        parcels.reset()

        # With different probs. change various things,
        # though if none chosen, parcels are still randomly re-generated  

        # Add, remove and/or chaneg size of a parcel
        if self.r_state.random() < self.mut_probs['add_prob']:
            parcels.add_parcel()

        if self.r_state.random() < self.mut_probs['del_prob']:
            parcels.remove_parcel()

        if self.r_state.random() < self.mut_probs['sz_prob']:
            parcels.change_prob()

        # Still need to call generate parcels
        parcels.generate_parcels()

        return parcels

    def Gen_Parcels(self):

        if self.mutate_flag:
            self.Mutate()

        if not self.parcels_lh.generated:
            self.parcels_lh.generate_parcels()

        if not self.parcels_rh.generated:
            self.parcels_rh.generate_parcels()

        return self

    def Compare(self, other, strategy):
        
        # AFPO ? 
        # or use multiple dimensions as score from SST vs MID, e.g.

        if strategy == 'afpo':

            if self.score > other.score and self.age <= other.age:
                return True
            return False

        elif strategy == 'scores':

            for i in range(len(self.scores)):
                if self.scores[i] <= other.scores[i]:
                    return False
            return True

        elif strategy == 'raw_scores':

            for i in range(len(self.raw_scores)):
                for j in range(len(self.raw_scores[i])):
                    if self.raw_scores[i][j] <= other.raw_scores[i][j]:
                        return False
            return True

        else:
            if self.score > other.score:
                return True
            return False

    def get_df(self, train_data, test_data, train_targets, test_targets):
        '''Assumes no list for targets'''


        train_X = get_X(train_data[0], train_data[1], self.parcels_lh, self.parcels_rh)
        test_X = get_X(test_data[0], test_data[1], self.parcels_lh, self.parcels_rh)

        train_df = pd.DataFrame(train_X)
        for t in range(len(train_targets)):
            train_df['target_' + str(t)] = train_targets[t]
        
        train_df['src_subject_id'] = ['train_' + str(i) for i in range(len(train_df))]
        train_df = train_df.set_index('src_subject_id')

        test_df = pd.DataFrame(test_X)
        for t in range(len(test_targets)):
            test_df['target_' + str(t)] = test_targets[t]
        
        test_df['src_subject_id'] = ['test_' + str(i) for i in range(len(test_df))]
        test_df = test_df.set_index('src_subject_id')

        full_df = pd.concat([train_df, test_df])
        full_df.columns = full_df.columns.astype(str)
        return full_df, test_df.index



class Existing_Surf_Individual(Surf_Individual):

    def __init__(self, lh_loc=None, rh_loc=None, lh=None, rh=None):

        geo = np.zeros(100)
        n_parcels_l, n_parcels_u = 5,10

        self.parcels_lh = Parcels(geo, n_parcels_l, n_parcels_u)
        self.parcels_rh = Parcels(geo, n_parcels_l, n_parcels_u)

        self.score = None
        self.age = 0

        if lh_loc is not None:
            if '.annot' in lh_loc:
                lh = io.read_annot(lh_loc)[0]
            else:
                lh = surface.load_surf_data(lh_loc)

        if rh_loc is not None:
            if '.annot' in rh_loc:
                rh = io.read_annot(rh_loc)[0]
            else:
                rh = surface.load_surf_data(rh_loc)
           

        self.parcels_lh.mask = lh
        self.parcels_lh.n_parcels = np.max(lh)

        self.parcels_rh.mask = rh
        self.parcels_rh.n_parcels = np.max(rh)
            






