#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import emcee as mc
import time
from math import exp,sqrt
import string
import sys, getopt
import corner
from multiprocessing import cpu_count
from multiprocessing import Process, freeze_support
from asp import *

# observed data
obs_data = {}
upper_limits = {}
lower_limits = {}
chi2_min = np.inf

fixed_value = {}
#bounds = {}
mc_paras_names = []
lightcurve_function = None

def clean_results(results):
    results['time'] = []
    
    for k in l_keys.keys():
        if results.__contains__(k):
            results[k] = []
    for k in v_keys.keys():
        if results.__contains__(k):
            results[k] = []
    
def sum_upper_limits_chi2(y, upper):
    chi2 = 0.0
    
    for i in range(len(y)):
        if y[i] > upper[i]:
            chi2 += ((y[i] - upper[i])/(upper[i]/1000.0))**2
    
    return chi2

def sum_lower_limits_chi2(y, lower):
    chi2 = 0.0
    
    for i in range(len(y)):
        if y[i] < lower[i]:
            chi2 += ((y[i] - lower[i])/(lower[i]/1000.0))**2
    
    return chi2

# likelyhood function
def lnlike(paras):
    T_start = 0.0
    try:
        i = mc_paras_names.index('T_start')
        T_start = paras[i]
    except ValueError:
        if fixed_value.__contains__('T_start'):
            T_start = fixed_value['T_start']
    
    cal_results = lightcurve_function(paras)	# should be provided by user
    if (not cal_results.__contains__('time')) or len(cal_results['time']) == 0:
        return -np.inf
    
    chi2 = 0.0
    for k in cal_results.keys():
        if obs_data.__contains__(k):
            data = obs_data[k]
            try:
                t_obs = data[:,0]
                k_obs = data[:,1]
                k_err = (data[:,2]-data[:,3])/2
            except IndexError:
                t_obs = np.array([data[0]])
                k_obs = np.array([data[1]])
                k_err = np.array([(data[2]-data[3])/2])
            t_obs = t_obs-T_start
            y = np.interp(t_obs, cal_results['time'], cal_results[k])
            chi2 += sum((y-k_obs)**2 / k_err**2)
            if np.isnan(chi2):
                return -np.inf
                
        if upper_limits.__contains__(k):
            upper = upper_limits[k]
            try:
                t_obs = upper[:,0]
                u_obs = upper[:,1]
            except IndexError:
                t_obs = [upper[0]]
                u_obs = [upper[1]]
            t_obs = t_obs - T_start
            y = np.interp(t_obs, cal_results['time'], cal_results[k], left = 0.0)
            chi2 += sum_upper_limits_chi2(y, u_obs)
            if np.isnan(chi2):
                return -np.inf
                
        if lower_limits.__contains__(k):
            lower = lower_limits[k]
            try:
                t_obs = lower[:,0]
                l_obs = lower[:,1]
            except IndexError:
                t_obs = [lower[0]]
                l_obs = [lower[1]]
            t_obs = t_obs - T_start
            y = np.interp(t_obs, cal_results['time'], cal_results[k], left = 0.0)
            chi2 += sum_lower_limits_chi2(y, l_obs)
            if np.isnan(chi2):
                return -np.inf
            
    return -0.5*chi2

# prior
def lnprior(paras, bnds):
    listp = list(paras)
    for i in range(len(paras)):
        k = mc_paras_names[i]
        if bnds[k][0] > listp[i] or listp[i] > bnds[k][1]:
            return (-np.inf, k)
                
    return (0.0, "")

# total probability
count = 0

def lnprob(paras,bnds):
    global count, chi2_min
    lp,err = lnprior(paras,bnds)
    if (not np.isfinite(lp)) or np.isnan(lp):
        return -np.inf
    ln = lnlike(paras)
    if np.isnan(ln) or np.isinf(ln):
        return -np.inf
    
    chi2_min = min(chi2_min, -(lp + ln))
    count += 1
    if np.mod(count,100)==0:
        print (lp+ln)
    return lp + ln

# mcmc
def myMCMC(res, bnds, ndim, nwalkers, lightcurve):
    global lightcurve_function
    lightcurve_function = lightcurve
    t_start = time.time()
    pos = [list(res.values())+abs(1e-6*np.random.randn(ndim))for i in range(nwalkers)]
    sampler = mc.EnsembleSampler(nwalkers, ndim, lnprob, \
        args=[bnds], threads=min(cpu_count(), 1))
    pos,prob,state = sampler.run_mcmc(pos, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 800)
    t_end = time.time()
    print("time spent: %g h" %((t_end-t_start)/3600.0))
    return sampler

def convert2normal(name, value):
    for nc in need_convert:
        if nc(name):
            if value <= -200:
                return 0.0
            return 10**value
    return value
    
def convert2log(name, value):
    for nc in need_convert:
        if nc(name):
            if value < 1e-200:
                value = 1e-200
            return np.log10(value)
    return value

def set_fixed_value(v):
    global fixed_value
    fixed_value = v

def get_normal_parameters(paras):
    p = {}
    for i in range(len(paras)):
        n = mc_paras_names[i]
        p[n] = convert2normal(n, paras[i])
    for k in fixed_value.keys():
        p[k] = convert2normal(k, fixed_value[k])

    return p

def load_mc_conf_for_mc(fname):
    (bounds, values, paras) = load_mc_conf(fname)

    for k in bounds.keys():
        bounds[k][0] = convert2log(k, bounds[k][0])
        bounds[k][1] = convert2log(k, bounds[k][1])
    for k in values.keys():
        values[k] = convert2log(k, values[k])

    for k in paras.keys():
        paras[k] = convert2log(k, paras[k])
        mc_paras_names.append(k)

    lp, k = lnprior(paras.values(), bounds)
    if not np.isfinite(lp):
        print("""
Parameter value of %s is outside of bound limits in configuration file %s.
Fix it before continue.""" % (k, fname))
        sys.exit(0)
    
    return (bounds, values, paras)
        
corner_label_functions = []
def corner_labels(paras):
    labels = []
    for k in paras.keys():
        for f in corner_label_functions:
            if f(k, labels):
                break

    return labels

def write_mc_result(samples, f):
    f.writelines("""# Parameter = low, high, best-value # low and high are
# 1-sigma bounds.\n""")
    for i in range(len(mc_paras_names)):
        n = mc_paras_names[i]
        s = samples[:,i]
        s.sort()
        s_len = len(s)
        n_skip = int(s_len * (1-0.67) / 2)
        low = convert2normal(n, s[n_skip])
        high = convert2normal(n, s[s_len - n_skip])
        median = convert2normal(n, s[int(s_len/2)])
        f.writelines("%s = %g, %g, %g\n" %(n, low, high, median))
    
