#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:38:12 2019
@author: sage
"""
import numpy as np
import copy, random, time
from multiprocessing import Pool

from Individual import Surf_Individual

def multi_proc_gen_parcels(indv):
    return indv.Gen_Parcels()

class Population():
    
    def __init__(self, geo, strategy, n_indv, new_rand,
                 n_parcels_l, n_parcels_u, mut_probs,
                 medial_wall_inds, n_jobs, verbose):

        self.geo = geo
        self.strategy = strategy
        self.n_indv = n_indv
        self.new_rand = new_rand
        self.n_parcels_l, self.n_parcels_u = n_parcels_l, n_parcels_u
        self.mut_probs = mut_probs
        self.medial_wall_inds = medial_wall_inds
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.individuals = []
        self.add_new(self.n_indv)
        self.gen_parcels()

        self.best_scores = []
        self.best_sizes = []

    def add_new(self, amt=1):

        for i in range(amt):
            self.individuals.append(Surf_Individual(self.geo, self.n_parcels_l,
                                                    self.n_parcels_u, self.mut_probs,
                                                    self.medial_wall_inds,
                                                    np.random.RandomState()))

    def gen_parcels(self):

        start = time.time()

        pool = Pool(self.n_jobs)
        self.individuals = pool.map(multi_proc_gen_parcels, self.individuals)
        pool.close()

        if self.verbose:
            print('gen parcel:', time.time() - start)

    def Evaluate(self, data, targets, models):

        start = time.time()

        for indv in self.individuals:
            indv.Evaluate(data, targets, models)

        if self.verbose:
            print('evaluate:', time.time() - start)

    def Tournament(self):

        start = time.time()
        
        while len(self.individuals) > self.n_indv // 2:
            self.attempt_remove()

        if self.verbose:
            print('tournament:', time.time() - start)
            
    def Fill(self):

        start = time.time()

        # New mutated
        while len(self.individuals) + self.new_rand < self.n_indv:
            self.add_mutated()

        # New random
        self.add_new(amt=self.new_rand)

        if self.verbose:
            print('fill:', time.time() - start)

        # Gen parcels for all new
        self.gen_parcels()
                      
    def attempt_remove(self):
        
        r1, r2 = random.sample(range(len(self.individuals)), 2)
        r1, r2 = int(r1), int(r2)
        indv1, indv2 = self.individuals[r1], self.individuals[r2]
        
        if indv1.Compare(indv2, self.strategy):
            del self.individuals[r2]
        elif indv2.Compare(indv1, self.strategy):
            del self.individuals[r1]
            
    def add_mutated(self):
       
        r = random.randint(0, (self.n_indv // 2)-1)
        new_copy = copy.deepcopy(self.individuals[r])
        new_copy.mutate_flag = True

        self.individuals.append(new_copy)

    def Get_Best_Score(self):

        scores = [indv.score for indv in self.individuals]
        best_ind = np.argmax(scores)
        best_indv = self.individuals[best_ind]

        best_score = best_indv.score
        best_size = best_indv.parcels_lh.n_parcels + best_indv.parcels_rh.n_parcels

        self.best_scores.append(best_score)
        self.best_sizes.append(best_size)

        return best_score

            
            
        
    
        
    
