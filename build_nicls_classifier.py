#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import cmlreaders as cml
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ptsa.data.filters import MorletWaveletFilter, ButterworthFilter
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(filename='classifier.log', level=logging.DEBUG)
logger = logging.getLogger('ClassifierBuild')



########################
# Cross Validation Functions
########################
# Based on RAM project code in pennmem/ramutils

def perform_loso_cross_validation(classifier, powers, events, **kwargs):
    """ Perform single iteration of leave-one-session-out cross validation
    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : np.recarray
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.
    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all sessions
    """
    classifier_copy = deepcopy(classifier)
    sessions = np.unique(events.session)
    recalls = events.recalled.values.astype(int)

    # Predicted probabilities should be assessed only on encoding words
    probs = np.empty_like(recalls, dtype=float)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (events.session != sess)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        classifier_copy.fit(insample_pow_mat, insample_recalls)

        # testing data 
        outsample_mask = ~insample_mask #& encoding_mask
        outsample_pow_mat = powers[outsample_mask]

        outsample_probs = classifier_copy.predict_proba(outsample_pow_mat)[:, 1]

        probs[outsample_mask] = outsample_probs

    return probs

def perform_lolo_cross_validation(classifier, powers, events, **kwargs):
    """Perform a single iteration of leave-one-list-out cross validation

    Parameters
    ----------
    classifier: sklearn model object
    powers: mean powers to use as features
    events: set of events for the session
    recalls: vector of recall outcomes
    kwargs: dict
         Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all lists

    Notes
    -----
    Be careful when passing a classifier object to this function since it's
    .fit() method will be called. If you use the classifier object after
    calling this function, the internal state may have changed. To avoid this
    problem, make a copy of the classifier object and pass the copy to this
    function.
    """
    recalls = events.recalled.astype(bool).values
    classifier_copy = deepcopy(classifier)

    probs = np.empty_like(recalls, dtype=float)
    lists = np.unique(events.trial)

    for lst in lists:
        insample_mask = (events.trial != lst)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]

        # We don't want to call fit on the passed classifier because this will
        # have side-effects for the user/program that calls this function
        classifier_copy.fit(insample_pow_mat, insample_recalls)

        # Out of sample predictions need to be on encoding only
        outsample_mask = ~insample_mask
        outsample_pow_mat = powers[outsample_mask]

        probs[outsample_mask] = classifier_copy.predict_proba(
            outsample_pow_mat)[:, 1]

    return probs

########################
# Classifier I/O
########################

class ClassifierModel:
    def __init__(self, model):
        self.model = model
        self.model_params = model.__dict__

    def save_json(self, filepath):
        for k, v in self.model_params.items():
            if isinstance(v, np.ndarray):
                self.model_params[k] = v.tolist()
        json_text = json.dumps(self.model_params)
        with open(filepath, 'w') as file:
            file.write(json_text)

    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            self.model_params = json.load(file)
        for k, v in self.model_params.items():
            if isinstance(v, list):
                self.model_params[k] = np.asarray(v)
        self.model.__dict__ = self.model_params
        return self

    def get(self):
        return self.model

#########################
# Load / Compute Features
#########################

# buffer is 1/2 the wavelet width at the lowest frequency

# NOTE: First session of LTP453 was inadvertently run with trials_per_session=8

def load_powers(subject, experiment='NiclsCourierReadOnly',
                num_readonly_sess=4, trials_per_session=10,
                rel_start=300, rel_stop=1300, buffer_time=416):
    data = cml.get_data_index(kind = 'ltp')
    data = data[(data['experiment']==experiment)&(data['subject']==subject)].sort_values('session').reset_index()
    
    full_pows = []
    full_evs = None
    for i, row in data.iterrows():
        print(f"Reading session {i} data")
        if i!=row['session']:
            logger.warn(f"Session {row['session']} and index {i} do not match")
        if i>=num_readonly_sess:
            logger.warn(f'Tried to load data for session {num_readonly_sess},\
                        but did not expect that many read-only sessions')
            break
        # intialize data reader, load words events and buffered eeg epochs
        try:
            r = cml.CMLReader(subject=subject, experiment=experiment, session=row['session'])
            evs = r.load('task_events')
            word_evs = evs[(evs.type=='WORD')&(evs.eegoffset!=-1)]
            eeg = r.load_eeg(word_evs, rel_start=rel_start - buffer_time, rel_stop=rel_stop + buffer_time).to_ptsa()
        except Exception as e:
            logger.warn(f"skipping session {row['session']} due to exception {e}")
            continue
        # if successful, append events
        full_evs = word_evs if full_evs is None else pd.concat([full_evs, word_evs], ignore_index=True)
        # average reference
        eeg = eeg[:, :128] - eeg[:, :128].mean('channel')
        # filter out line noise at 60 and 120Hz
        eeg = ButterworthFilter(eeg, filt_type='stop', freq_range=[58, 62], order=4).filter()
        eeg = ButterworthFilter(eeg, filt_type='stop', freq_range=[118, 122], order=4).filter()
        # highpass filter to account for drift 
        eeg = ButterworthFilter(eeg, filt_type='highpass', freq_range=1).filter()
        pows = MorletWaveletFilter(eeg, np.logspace(np.log10(6), np.log10(180), 8), width=5, output='power', cpus=25).filter()
        pows = pows.remove_buffer(buffer_time / 1000).data + np.finfo(float).eps/2.
        log_pows = np.log10(pows)
        # swap order of events and frequencies --> result is events x frequencies x channels x time
        # next, average over time
        avg_pows = np.nanmean(log_pows.transpose((1, 0, 2, 3)), -1)
        # reshape as events x features
        avg_pows = avg_pows.reshape((avg_pows.shape[0], -1))
        # Z-transform the features within each session
        norm_pows = StandardScaler().fit_transform(avg_pows)
        full_pows.append(norm_pows)
    full_pows = np.vstack(full_pows)
    # Need unique trial labels
    full_evs.trial = full_evs.trial + trials_per_session * full_evs.session
    return full_pows, full_evs

def NestedCV(full_pows, full_evs, c_list):
    # Selecting overall parameter using nested CV, with LOLO cross validation to get scores for every
    # parameter and session, then averaging across sessions and taking the paramter which yeilds the highest
    # average AUC
    all_scores = []
    #import pdb; pdb.set_trace()
    for sess in full_evs.session.unique():
        out_mask = full_evs.session == sess
        in_mask = ~out_mask
        score_list = []
        for c in c_list:
            model = LogisticRegression(penalty='l2', C=c, solver='liblinear')
            probs = perform_lolo_cross_validation(model, full_pows[in_mask], full_evs[in_mask])
            score_list.append(roc_auc_score(full_evs[in_mask].recalled.astype(int), probs))
        all_scores.append(score_list)
    # return scores matrix shaped sessions x hyperparameter
    scores_mat = np.stack(all_scores)
    return scores_mat

def main(subject):
    c_list = np.logspace(np.log10(2e-5), np.log10(1), 9)
    powers, events = load_powers(subject)
    scores_mat = NestedCV(powers, events, c_list)
    best_c = c_list[scores_mat.mean(0).argmax()]
    model = LogisticRegression(penalty='l2', C=best_c, class_weight='balanced', solver='liblinear')
    model.fit(powers, events.recalled.astype(int))
    save_model = ClassifierModel(model)
    save_model.save_json(f"/data/eeg/scalp/ltp/NiclsCourierReadOnly/{subject}/nicls_{subject}_classifier.json")

if __name__=="__main__":
    subject = sys.argv[1]
    print(f"Building Classifier for {subject}")
    main(subject)
