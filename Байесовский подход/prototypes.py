# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 1 the following functions are required:
# In variant 1 the following functions are required:

import numpy as np
import scipy.stats as stats

def binom(n, k, p):
    """
    probability function for binomial distribution
    n - number of trials
    k - support of a distribution
    p - success probability
    """
    return stats.binom.pmf(n=n, k=k, p=p)

def poiss(k, mu):
    return stats.poisson.pmf(k=k, mu=mu)

def pa(params, model):
    """
    uniform distribution a ~ R[amin, amax]
    """
    val = np.arange(params['amin'], params['amax'] + 1, dtype=np.int32)
    p = 1 / len(val)
    prob = p * np.ones_like(val, dtype=np.float64)
    return prob, val

def pb(params, model):
    """
    uniform distribution b ~ R[bmin, bmax]
    """
    val = np.arange(params['bmin'], params['bmax'] + 1, dtype=np.int32)
    p = 1 / len(val)
    prob = p * np.ones_like(val, dtype=np.float64)
    return prob, val

def pai_a(a, params, model):
    """
    binomial distribution ai|a ~ Bin(a, p1) or Poiss(a*p1), where a = const
    """
    val = np.arange(params['amax'] + 1, dtype=np.int32)
    val_ = val.reshape(-1, 1)

    if model == 1:
        prob = binom(a, val_, params['p1'])
    
    if model == 2:
        prob = poiss(val_, params['p1'] * np.array(a))
        
    return prob, val
    
def pbi_b(b, params, model):
    """
    binomial distribution bi|b ~ Bin(b, p2) or Poiss(b*p2), where b = const
    """
    val = np.arange(params['bmax'] + 1, dtype=np.int32)
    val_ = val.reshape(-1, 1)
    
    if model == 1:
        prob = binom(b, val_, params['p2'])
    
    if model == 2:
        prob = poiss(val_, params['p2'] * np.array(b))
        
    return prob, val

def pai(params, model):
    """
    distribution ai, where ai|a ~ Bin(a, p1) or Poiss(ap1) and a ~ R[amin, amax]
    """
    a_prob, a_val = pa(params, model)
    prob, val = pai_a(a_val, params, model)
    prob = np.dot(prob, a_prob)
    return prob, val

def pbi(params, model):
    """
    distribution bi, where bi|b ~ Bin(b, p2) or Poiss(bp2) and b ~ R[bmin, bmax]
    """
    b_prob, b_val = pb(params, model)
    prob, val = pbi_b(b_val, params, model)
    prob = np.dot(prob, b_prob)
    return prob, val

def c_conds(d_ai, d_bi, params):
    ai_prob, ai_val = d_ai
    bi_prob, bi_val = d_bi
    amax, bmax = params['amax'], params['bmax']
    cmax = amax + bmax + 1
    
    val = np.arange(cmax, dtype=np.int32)
    
    idx = np.add.outer(val, -bi_val)
    idx = np.where((0 <= idx) & (idx <= amax), idx, -1)
    
    ai_is_1D = ai_prob.ndim == 1
    bi_is_1D = bi_prob.ndim == 1

    if ai_is_1D:
        ai_prob = ai_prob.reshape(-1, 1)      
    ai_prob = np.pad(ai_prob, ((0, 1), (0, 0)))
    ai_prob = ai_prob[idx]
        
    prob = np.dot(bi_prob.T, ai_prob)
    if ai_is_1D:
        prob = np.squeeze(prob, -1)
        
    if not bi_is_1D:
        prob = np.swapaxes(prob, 0, 1)
    return prob, val

def pc_ab(a, b, params, model):
    """
    sum of binomial distributions c|a,b = ai|a + bi|b, where 
    ai|a ~ Bin(a, p1) or Poiss(ap1) and a = const
    bi|b ~ Bin(b, p2) or Poiss(bp2) and b = const
    """
    prob, val = c_conds(pai_a(a, params, model),
                        pbi_b(b, params, model),
                        params)
    prob = np.swapaxes(prob, 1, 2)
    return prob, val

def pc(params, model):
    """
    sum of distributions c = ai + bi, where 
    ai|a ~ Bin(a, p1) or Poiss(ap1) and a ~ R[amin, amax]
    bi|b ~ Bin(b, p2) or Poiss(bp2) and b ~ R[bmin, bmax]
    """
    prob, val = c_conds(pai(params, model),
                        pbi(params, model),
                        params)
    return prob, val

def pc_a(a, params, model):
    """
    sum of distributions c|a = ai|a + bi, where 
    ai|a ~ Bin(a, p1) or Poiss(ap1) and a = const
    bi|b ~ Bin(b, p2) or Poiss(bp2) and b ~ R[bmin, bmax]
    """
    prob, val = c_conds(pai_a(a, params, model),
                        pbi(params, model),
                        params)
    return prob, val

def pc_b(b, params, model):
    """
    sum of distributions c|b = ai + bi|b, where 
    ai|a ~ Bin(a, p1) or Poiss(ap1) and a ~ R[amin, amax]
    bi|b ~ Bin(b, p2) or Poiss(bp2) and b = const
    """
    prob, val = c_conds(pai(params, model),
                        pbi_b(b, params, model),
                        params)
    return prob, val

def pd_c(c, params, model):
    """
    binomial distribution d|c ~ c + ci, where
    ci ~ Bin(c, p3) and c = const
    """
    amax, bmax = params['amax'], params['bmax']
    dmax = 2 * (amax + bmax) + 1
    
    val = np.arange(dmax, dtype=np.int32)
    val_ = val.reshape(-1, 1)
    prob = binom(c, val_-c, params['p3']).astype(np.float128)
    return prob, val

def pd(params, model):
    """
    distribution d, where d|c ~ c + Bin(c, p3) and c ~ p(c)
    """
    c_prob, c_val = pc(params, model)
    prob, val = pd_c(c_val, params, model)
    prob = np.dot(prob, c_prob)
    return prob, val

def pc_d(d, params, model):
    """
    distribution c|d, where d|c ~ c + Bin(c, p3) and c ~ p(c)
    """
    c_prob, val = pc(params, model)
    d_c_prob = pd_c(val, params, model)[0][d]
    
    prob = np.multiply(d_c_prob, c_prob).T
    S = prob.sum(axis=0)
    S[S==0.] = 1.
    prob /= S
    return prob, val

def pc_abd(a, b, d, params, model):
    """
    distribution c|a,b,d, where c|a,b ~ p(c|a,b) and c|d ~ p(c|d)
    """
    c_ab_prob, val = pc_ab(a, b, params, model)
    d_c_prob = pd_c(val, params, model)[0][d]
    prob = c_ab_prob[...,None] * d_c_prob.T[:,None, None,:]
    S = prob.sum(axis=0)
    S[S==0.] = 1.
    prob /= S
    return prob, val

# In variant 2 the following functions are required:
# def pa(params, model):
# def pb(params, model):
# def pc(params, model):
# def pd(params, model):
# def pc_a(a, params, model):
# def pc_b(b, params, model):
# def pb_a(a, params, model):
# def pb_d(d, params, model):
# def pb_ad(a, d, params, model):

# In variant 3 the following functions are required:
# def pa(params, model):
# def pb(params, model):
# def pc(params, model):
# def pd(params, model):
# def generate(N, a, b, params, model):
# def pb_d(d, params, model):
# def pb_ad(a, d, params, model):

