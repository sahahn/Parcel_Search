import numpy as np
import random

class Parcels():
    
    def __init__(self, geo, n_parcels_l, n_parcels_u, medial_wall_inds=None, r_state=None):
        
        # Proc Geo
        self.geo = [np.array(g) for g in geo]

        # Proc medial wall inds
        if medial_wall_inds is not None:
            self.m_wall = set(list(medial_wall_inds))
        else:
            self.m_wall = set()
 
        # Proc random state
        if r_state is None:
            r_state = np.random.RandomState()
        self.r_state = r_state

        # Set up mask, done and flags
        self.sz = len(self.geo)
        self.reset()

        # Set starting spots 
        self.n_parcels_l, self.n_parcels_u  = n_parcels_l, n_parcels_u

        self.set_n_parcels()
        self.init_parcels()

    def reset(self):
        '''Just reset the mask, and set w/ done info'''

        self.mask = np.zeros(self.sz, dtype='int16')
        self.done = self.m_wall.copy()
        self.ready, self.generated = False, False

    def set_n_parcels(self):
        
        self.n_parcels = self.r_state.randint(self.n_parcels_l, self.n_parcels_u)
        
    def init_parcels(self):

        # Generate the starting locs
        valid = np.setdiff1d(np.arange(self.sz), np.array(list(self.done)))
        self.start_locs = self.r_state.choice(valid, size=self.n_parcels,
                                              replace=False)

        # Set random probs. that each loc is chosen
        self.probs = self.r_state.random(size=self.n_parcels)

    def setup(self):
        '''This should be called before generating parcel,
        so after a mutation has been made, setup needs to
        be called. It also does not hurt to call setup an
        extra time, as nothing random is set.'''

        # Generate corresponding labels w/ each loc
        self.labels = np.arange(1, self.n_parcels+1, dtype='int16')

        # Mask where if == 1, then that parcel is done
        self.finished = np.zeros(self.n_parcels, dtype='bool_')

        # Drop the first points
        self.mask[self.start_locs] = self.labels

        # Set ready flag to True
        self.ready = True

    def add_parcel(self):

        if self.n_parcels + 1 < self.n_parcels_u:

            # Gen new unseen start loc + prob for a parcel
            unused = np.setdiff1d(np.arange(self.sz), self.start_locs)
            new_loc = self.r_state.choice(unused)
            new_prob = self.r_state.random()

            # Add to existing
            self.start_locs = np.append(self.start_locs, new_loc)
            self.probs = np.append(self.probs, new_prob)

            self.n_parcels += 1

    def remove_parcel(self):

        if self.n_parcels - 1 > self.n_parcels_l:

            # Select parcel to delete
            to_del = self.r_state.choice(np.arange(len(self.start_locs)))

            # Delete parcel + prob
            self.start_locs = np.delete(self.start_locs, to_del)
            self.probs = np.delete(self.probs, to_del)

            self.n_parcels -= 1

    def change_prob(self):
        '''Change the prob / size of a parcel'''

        # Select parcel to change
        to_change = self.r_state.choice(np.arange(len(self.start_locs)))
        self.probs[to_change] = self.r_state.random()
        
    def get_probs(self):

        return self.probs / np.sum(self.probs)
        
    def choice(self):
        '''Select a valid label based on probs.'''
        
        msk = self.finished == 0
        probs = self.probs[msk] / np.sum(self.probs[msk])
        label = self.r_state.choice(self.labels[msk], p=probs)
        
        return label
    
    def get_valid_neighbors(self, loc):
        
        ns = self.geo[loc]
        valid_ns = ns[self.mask[ns] == 0]
        
        return valid_ns
        
    def generate_parcels(self):

        if self.ready is False:
            self.setup()
        
        # Keep looping until every spot is filled
        while (self.finished == 0).any():
            self.add_spot()

        # Set generated flag when done
        self.generated = True
    
    def add_spot(self):

        # Select which parcel to add to
        label = self.choice()

        # Determine valid starting locations anywhere in exisitng parcel
        current = np.where(self.mask == label)[0]
        valid = set(current) - self.done

        self.proc_spot(valid, label)

    def proc_spot(self, valid, label):

        # If no valid choices, then set this parcel to finished
        if len(valid) == 0:
            self.finished[label-1] = 1
            return

        # Select randomly from the valid starting locs
        loc = random.choice(tuple(valid))
            
        # Select a valid + open neighbor
        valid_ns = self.get_valid_neighbors(loc)

        if len(valid_ns) > 0:

            # Select a valid choice, and add it w/ the right label
            choice = random.choice(valid_ns)
            self.mask[choice] = label

            # If this was the only choice, mark start loc as done
            if len(valid_ns) == 1:
                self.done.add(loc)

        # If there are no valid choices, mark as done
        else:
            self.done.add(loc)

            valid.remove(loc)
            self.proc_spot(valid, label)

    

    