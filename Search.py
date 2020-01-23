#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 11 11:25:00 2019
@author: sage
"""
from Population import Population
import pickle, os, sys
import pandas as pd
import numpy as np
import time

class Search():

    def __init__(self, save_loc, geo, data, targets, models,
                 strategy='afpo', n_gens=100, n_indv=10,
                 new_rand=1, n_parcels_l=20, n_parcels_u=100,
                 add_prob=.2, del_prob=.2,
                 sz_prob=.2, medial_wall_inds=None,
                 n_jobs=8, save_every=5,
                 verbose=False):

        self.save_loc = save_loc
        self.data = data
        self.targets = targets
        self.models = models
        self.n_gens = n_gens
        self.save_every = save_every
        self.verbose = verbose

        mut_probs = {'add_prob': add_prob,
                     'del_prob': del_prob,
                     'sz_prob': sz_prob}

        print('Init Pop.')
        self.pop = Population(geo=geo, strategy=strategy, n_indv=n_indv, new_rand=new_rand,
                              n_parcels_l=n_parcels_l, n_parcels_u=n_parcels_u,
                              mut_probs=mut_probs, medial_wall_inds=medial_wall_inds,
                              n_jobs=n_jobs, verbose=verbose)
        self.gens = 1

    def save(self):

        save_start = time.time()

        with open(self.save_loc, 'wb') as f:
            pickle.dump(self, f)

        if self.verbose:
            print('save:', time.time() - save_start)

    def run(self):

        if self.gens == 1:
            self.eval()

        while self.gens <= self.n_gens:
            self.run_gen()

            if self.gens % self.save_every == 0:
                self.save()

        self.save()

    def run_gen(self):

        self.pop.Tournament()
        self.pop.Fill()

        self.eval()
        
    def eval(self):

        self.pop.Evaluate(self.data, self.targets, self.models)

        print('Gen:', str(self.gens),
              'Best Score:', self.pop.Get_Best_Score())

        self.gens += 1



