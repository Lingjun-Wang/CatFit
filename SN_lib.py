#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SN light curve module.

Shock cooling included.
Multiband-enabled.
Photospheric radius handled.
'''

import numpy as np
from math import exp,sqrt
import string
from asp import *
from mc_lib import corner_label_functions

beta = 13.8

'''
model parameters:
    M_ej,B_p,P_0,M_Ni,kappa,kappa_gamma_mag,kappa_gamma_Ni, v_sc0, 
    M_env, R_env, R0, E_th0
'''

# t in observer's frame
# shock cooling will be not included if M_env is not defined in 'parameters'
# FIXME: not do K correction
def SN_lightcurve(parameters, z, DL, timeq, results, power = None):
    R0 = 1.0e11
    E_th0 = 1.0e50 # initial thermal energy
    kappa = 0.1
    M_ej = 5.9
    kappa_gamma_Ni = 0.027
    kappa_gamma_mag = 1.5
    v_sc0 = 0.0
    B_p = 1.26e-10
    P0 = 1.1e10
    M_Ni = 0.0
    #M_env = 1.0e-10
    #R_env = 1e-10
    E_th0 = 1.0e48 # initial thermal energy
    
    for k in parameters.keys():
        if k == 'M_ej':
            M_ej = parameters[k]
        elif k == 'M_Ni':
            M_Ni = parameters[k]
        elif k == 'B_p':
            B_p = parameters[k]
        elif k == 'P0':
            P0 = parameters[k]
        elif k == 'kappa':
            kappa = parameters[k]
        elif k == 'kappa_gamma_Ni':
            kappa_gamma_Ni = parameters[k]
        elif k == 'kappa_gamma_mag':
            kappa_gamma_mag = parameters[k]
        elif k == 'v_sc0':
            v_sc0 = parameters[k]
        elif k == 'M_env':
            M_env = parameters[k]*Msun
        elif k == 'R_env':
            R_env = parameters[k]
        elif k == 'R0':
            R0 = parameters[k]
        elif k == 'E_th0':
            E_th0 = parameters[k]
    
    M_ej = M_ej*Msun
    M_Ni = M_Ni*Msun
    v_sc0 = v_sc0 * 1e5
    
    M_total = M_ej
    if 'M_env' in locals().keys():
        Mcore = M_ej # Mcore is always set as Mej, see below
        M_total = M_ej + M_env
    
    if results.__contains__('lum'):
        lum = results['lum']
    if results.__contains__('M_U'):
        M_U = results['M_U']
    if results.__contains__('M_B'):
        M_B = results['M_B']
    if results.__contains__('M_V'):
        M_V = results['M_V']
    if results.__contains__('M_R'):
        M_R = results['M_R']
    if results.__contains__('M_I'):
        M_I = results['M_I']

    if results.__contains__('vel'):
        vels = results['vel']
    if results.__contains__('Teff'):
        Teffs = results['Teff']

    E_K = 3.0/5.0 * M_ej * v_sc0**2 / 2.0
    E_SN = E_K      # E_SN is always set to E_k
    E_inp = 0.0
    phi = 1.0
    x_ph = 1.0
    x_ph_old = 1.0
    E_atm = 0.0
    Volume_atm = 0.0
    Volume_atm_old = 0.0
    E_kinetic_mag = 0.0
    x_ph0 = 0.55
    v_sc = v_sc0
    
    R = R0
    tau0 = kappa*M_total / (beta*c*R0) # timescale
    
    if 'M_env' in locals().keys():
        t_p_breakout = 0.9*24*3600 *(kappa/0.34)**0.5 * (E_SN/1e51)**(-1./4) *\
            (Mcore/Msun)**0.17 * (M_env/0.01/Msun)**0.57
        Eenv = 4e49 * E_SN/1e51 * (Mcore/Msun)**(-0.7) * (M_env/0.01/Msun)**0.7
        v_env = 2e9 * (E_SN/1e51)**0.5 * (Mcore/Msun)**(-0.35) * \
            (M_env/0.01/Msun)**(-0.15)
        if M_env < 4 * PI * R_env**2 * c/kappa/v_env:
            #print("Envelope mass is too small")
            return

    t = 0.0
    l = len(timeq)
 
    for i in range(l):
        delta = (timeq[i] - t)#/(1+z)
        t = timeq[i]#/(1+z)
        
        x_ph = 1 - 1e-6
        x_ph_old = x_ph
        tau_gamma_Ni_SN = 3*kappa_gamma_Ni*M_total*x_ph/(4*PI*R**2)
        tau_gamma_mag_SN = 3*kappa_gamma_mag*M_total*x_ph/(4*PI*R**2)
        L_Ni = M_Ni*((eps_Ni-eps_Co)*exp(-t/tau_Ni)+eps_Co*exp(-t/tau_Co))
        L_inp_SN_Ni = L_Ni*(1-exp(-tau_gamma_Ni_SN))
        Lp = 0.0
        if power is None:
            Lp = magnetar_power(t, P0, B_p)
        else:
            Lp = power(t, parameters)
        L_inp_SN_mag = Lp*(1-exp(-tau_gamma_mag_SN))
        
        #photosphere
        Volume = 4*PI*R**3/3.0
        lam = 1./(M_total/Volume*kappa)
        #x_ph_old = x_ph
        #x_ph = 1-2./3 * lam/R
        x_ph_derivative = 0.0
        L_ph = 0.0
        if x_ph >= 0.0:
            x_ph_derivative = (x_ph-x_ph_old)/delta
            if x_ph > x_ph0:
                dphi = 1.0/E_th0*R/R0*(L_inp_SN_mag+L_inp_SN_Ni)/x_ph**3 \
                    -R/R0/tau0*phi/x_ph**2 - 3*x_ph_derivative/x_ph*phi
                phi = phi+delta*dphi
                L_ph = E_th0/tau0*x_ph*phi
                ratio0 = L_ph / (L_inp_SN_mag + L_inp_SN_Ni)
                if x_ph < x_ph0 + 0.05 and 1.0 < ratio0 < 1.07:
                    x_ph0 = x_ph
            else:
                L_ph = (L_inp_SN_mag + L_inp_SN_Ni) * ratio0
        else:
            x_ph = -1e-6
        
        Volume_atm_old = Volume_atm
        Volume_atm = 4./3 * PI*R**3 * (1-x_ph**3)
        L_atm = 0.0
        
        if (L_atm<0.0 or L_ph<0.0):
            print('time=%.3e;L_atm=%.3e;L_ph=%.3e;E_atm=%.3e;x_ph=%.3e;phi=%.3e' \
                %(t/24/3600,L_atm,L_ph,E_atm,x_ph,phi))
            #return
        
        tau_gamma_Ni = 3*kappa_gamma_Ni*M_total/4/PI/R**2
        tau_gamma_mag = 3*kappa_gamma_mag*M_total/4/PI/R**2
        L_Ni_atm = L_Ni*(exp(-tau_gamma_Ni_SN) - exp(-tau_gamma_Ni))
        L_mag_atm = Lp*(exp(-tau_gamma_mag_SN) - exp(-tau_gamma_mag))

        if 'M_env' in locals().keys():
            L_p_breakout = R_env*Eenv/v_env/t_p_breakout**2* \
                exp(-(t_p_breakout+2*R_env/v_env)/2/t_p_breakout)

            L_breakout = -L_p_breakout * (t/t_p_breakout)**2 + \
                2 * L_p_breakout *t/t_p_breakout
            L_breakout = R_env*Eenv/v_env/t_p_breakout**2 * \
                exp(-t*(v_env*t+2*R_env)/2/v_env/t_p_breakout**2)
        
            L_ph = L_ph + L_breakout

        if x_ph >= 0.0:
            Teff_ph = (L_ph/4/PI/sig_st/(R*x_ph)**2)**0.25
	
        T_atm = (E_atm/arad/Volume_atm)**0.25
        p_atm = E_atm/Volume_atm/3
        E_atm = E_atm-p_atm*(Volume_atm-Volume_atm_old)\
            + delta*(L_Ni_atm+L_mag_atm-L_atm\
            - 4*PI*R**3 * x_ph**2 * arad*Teff_ph**4 * x_ph_derivative)
        
        try:
            if E_atm < 0.0:
                E_atm = 0.0
        except TypeError:
            print(Volume_atm, Volume_atm_old, L_Ni_atm, L_mag_atm,\
               L_atm, x_ph, Teff_ph, x_ph_derivative)
            return
        Teff = ((L_ph*Teff_ph**4 + L_atm*T_atm**4)/(L_ph+L_atm))**0.25
        L_kinetic = Lp*(1-exp(-tau_gamma_mag))
        E_inp = E_inp+(L_kinetic-L_ph-L_atm - p_atm*4*PI*R**2*v_sc)*delta
        E_inp = max(0.0, E_inp)
        if np.isnan(E_K+E_inp):
            print("Oops: E_K+E_inp is NaN")
            clean_results(results)
            return
        if E_K+E_inp < 0:
            # This is certainly caused by the large time step
            print("Oops: E_K+E_inp=%g\n" %(E_K+E_inp))
            clean_results(results)
            return
        v_sc = sqrt(10./3 * (E_K+E_inp)/M_total)
        R += v_sc*delta
        E_kinetic_mag += L_kinetic*delta
        
        if 'lum' in locals().keys():
            lum.append(np.log10(L_ph + L_atm)) # log
        if 'M_U' in locals().keys():
            Fv_U = bb_Fv_lambda_A(lambda_U, Teff_ph, R*x_ph, DL) #+ \
            M_U.append(U_mag(Fv_U))
        if 'M_B' in locals().keys():
            Fv_B = bb_Fv_lambda_A(lambda_B, Teff_ph, R*x_ph, DL) #+ \
            M_B.append(B_mag(Fv_B))
        if 'M_V' in locals().keys():
            Fv_V = bb_Fv_lambda_A(lambda_V, Teff_ph, R*x_ph, DL) #+ \
            M_V.append(V_mag(Fv_V))
        if 'M_R' in locals().keys():
            Fv_R = bb_Fv_lambda_A(lambda_R, Teff_ph, R*x_ph, DL) #+ \
            M_R.append(R_mag(Fv_R))
        if 'M_I' in locals().keys():
            Fv_I = bb_Fv_lambda_A(lambda_I, Teff_ph, R*x_ph, DL) #+ \
            M_I.append(I_mag(Fv_I))
	    
        x_ph = max(1-2./3 * lam/R, 1e-16)
	
        if 'vels' in locals().keys():
            vels.append(max(v_sc * (1-2./3 * lam/R)/1e5, 0.0)) # in km/s
        if 'Teffs' in locals().keys():
            Teffs.append(Teff)
        if results.__contains__('R_ph'):
            results['R_ph'].append(R*x_ph)
        
    return

# for mc programme
def need_convert_SN(name):
    if name in ("B_p", "P0", "v_sc0", "R_env", "R0", "E_th0"):
        return True
    return False
    
if not need_convert.__contains__(need_convert_SN):
    need_convert.append(need_convert_SN)

def corner_label_SN(k, labels):
    found = True
    if k == 'M_ej':
        labels.append(r'$M_{\mathrm{ej},\odot}$')
    elif k == 'M_Ni':
        labels.append(r'$M_{\mathrm{Ni},\odot}$')
    elif k == 'B_p':
        labels.append(r'$\log(B_p)$')
    elif k == 'P0':
        labels.append(r'$\log(P_0)$')
    elif k == 'kappa':
        labels.append(r'$\kappa$')
    elif k == 'kappa_gamma_Ni':
        labels.append(r'$\kappa_{\gamma}^{\mathrm{Ni}}$')
    elif k == 'kappa_gamma_mag':
        labels.append(r'$\kappa_{\gamma}^{\mathrm{mag}}$')
    elif k == 'T_start':
        labels.append(r'$T_{\mathrm{start}}$')
    elif k == 'v_sc0':
        labels.append(r'$\log(v_{\mathrm{sc}0})$')
    elif k == 'M_env':
        labels.append(r'$M_{\mathrm{env},\odot}$')
    elif k == 'R_env':
        labels.append(r'$\log(R_{\mathrm{env},\odot})$')
    elif k == 'R0':
        labels.append(r'$\log(R_0)$')
    elif k == 'E_th0':
        labels.append(r'$\log(E_{\mathrm{th}0})$')
    else:
        found = False
    
    return found

if not corner_label_functions.__contains__(corner_label_SN):
    corner_label_functions.append(corner_label_SN)

