import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
import astropy.constants as constants
from scipy.integrate import odeint
#from parameter import *
from scipy.integrate import quad
from math import *
import sys
import string
from SN_lib import *

# model constant (units:cgs)
beta = 13.8
R0 = 1.0e10
km = 1e5

# interaction parameters 
n6s0 = {
'n': 6.0,    # the slope of the outer ejecta
's': 0.0,    # the power-law index for CSM density profile, shell
'beta_FS': 1.256,   # n=6,s=0   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.906,     # n=6,s=0   # constants representing the ratio of the shock radii to the cd radius
'A': 2.4,           # n=6,s=0   # a constant
}

n6s2 = {
'n': 6.0,    # the slope of the outer ejecta
's': 2.0,    # the power-law index for CSM density profile, wind
'beta_FS': 1.377,   # n=6,s=2   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.958,     # n=6,s=2   # constants representing the ratio of the shock radii to the cd radius
'A': 0.62,           # n=6,s=2   # a constant
}

n7s0 = {
'n': 7.0,    # the slope of the outer ejecta
's': 0.0,    # the power-law index for CSM density profile, shell
'beta_FS': 1.181,   # n=7,s=0   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.935,     # n=7,s=0   # constants representing the ratio of the shock radii to the cd radius
'A': 1.2,           # n=7,s=0   # a constant
}

n7s2 = {
'n': 7.0,    # the slope of the outer ejecta
's': 2.0,    # the power-law index for CSM density profile, wind
'beta_FS': 1.299,   # n=7,s=2   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.97,     # n=7,s=2   # constants representing the ratio of the shock radii to the cd radius
'A': 0.27,           # n=7,s=2   # a constant
}

n8s0 = {
'n': 8.0,    # the slope of the outer ejecta
's': 0.0,    # the power-law index for CSM density profile, shell
'beta_FS': 1.154,   # n=8,s=0   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.95,     # n=8,s=0   # constants representing the ratio of the shock radii to the cd radius
'A': 0.71,           # n=8,s=0   # a constant
}

n8s2 = {
'n': 8.0,    # the slope of the outer ejecta
's': 2.0,    # the power-law index for CSM density profile, wind
'beta_FS': 1.267,   # n=8,s=2   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.976,     # n=8,s=2   # constants representing the ratio of the shock radii to the cd radius
'A': 0.15,           # n=8,s=2   # a constant
}

n9s2 = {
'n': 9.0,    # the slope of the outer ejecta
's': 2.0,    # the power-law index for CSM density profile, wind
'beta_FS': 1.25,   # n=9,s=2   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.981,     # n=9,s=2   # constants representing the ratio of the shock radii to the cd radius
'A': 0.096,           # n=9,s=2   # a constant
}

n12s0 = {
'n': 12.0,    # the slope of the outer ejecta
's': 0.0,    # the power-law index for CSM density profile, shell
'beta_FS': 1.121,   # n=12,s=0   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.974,     # n=12,s=0   # constants representing the ratio of the shock radii to the cd radius
'A': 0.19,           # n=12,s=0   # a constant
}

n12s2 = {
'n': 12.0,    # the slope of the outer ejecta
's': 2.0,    # the power-law index for CSM density profile, wind
'beta_FS': 1.226,   # n=12,s=2   # constants representing the ratio of the shock radii to the cd radius
'beta_RS': 0.987,     # n=12,s=2   # constants representing the ratio of the shock radii to the cd radius
'A': 0.038,           # n=12,s=2   # a constant
}

#model
delta = 0.0  # the slope of the inner ejecta
n = 12 #csms_param[0]['ns']['n']    # the slope of the outer ejecta 
x_0 = 0.7    # x_0 denotes the dimensionless radius of break in the supernova ejecta density profile from the inner component to the outer component
E_SN = 1e50
g_n = 1e10
R_ph= 1e10	# photosphere radius, in the outmost shell, set in set_layer()
M_ej = 0.0
v_SN = 0.0
R = 1e10	# the radius of the outmost shell, set in set_layer()

def set_layer(csms_param, i):
    global R_ph
    global R
    rho_CSM = csms_param[i]['rho_CSM']
    R_in = csms_param[i]['R_in']
    M_CSM = csms_param[i]['M_CSM']
    kappa = csms_param[i]['kappa']
    s = csms_param[i]['ns']['s']
    A = csms_param[i]['ns']['A']
    beta_FS = csms_param[i]['ns']['beta_FS']
    beta_RS = csms_param[i]['ns']['beta_RS']

    csms_param[i]['g_n']=g_n
    csms_param[i]['v_SN']=v_SN
    q=rho_CSM*R_in**(s)                # a scaling constant of CSM density
    csms_param[i]['q']=q
    csms_param[i]['t_in']=R_in/v_SN #inital time of CSM-ejecta interaction
    R_CSM=((3-s)/(4*PI*q)*M_CSM + R_in**(3-s))**(1/(3.0-s)) # the outer radius of CSM
    csms_param[i]['R_CSM']= R_CSM
    R_ph_i=(R_CSM**(1-s)-2*(1-s)/(3.*kappa*q))**(1/(1.0-s)) # the photosphere radius of layer i
    #if R_ph_i < R_in:
    #    return -1	# not a optically thick shell, reject this parameter

    M_CSM_TH=M_CSM
    if i == len(csms_param) - 1:
        R_ph = R_ph_i
        R = R_CSM
        M_CSM_TH=(4*PI*q)/(3.0-s)*(R_ph_i**(3-s)-R_in**(3.0-s)) # optical thick mass of CSM
    csms_param[i]['M_CSM_TH'] = M_CSM_TH
    csms_param[i]['t_FS']=(((3.0-s)*q**((3.0-n)/(n-s)))*((A*g_n)**((s-3.0)/(n-s)))/(4*PI*beta_FS**(3-s)))**((n-s)/((n-3)*(3.0-s)))*M_CSM_TH**((n-s)/((n-3.0)*(3.0-s))) #termination timescale of forward shock
    csms_param[i]['t_RS']=(v_SN/(beta_RS*(A*g_n/q)**(1/(n-s)))*(1-((3-n)*M_ej)/(4*PI*v_SN**(3-n)*g_n))**(1/(3.-n)))**((n-s)/(s-3.)) #termination timescale of revese shock
    return 0

def set_csms(parameters, csms_param, i_start = 0):
    global E_SN, g_n

    E_SN=(3-delta)*(n-3)/(2*(5.-delta)*(n-5.0))*M_ej*(x_0*v_SN)**2.0  # the kinetic energy of the SN
    g_n=1/(4*PI*(n-delta))*(2*(5.0-delta)*(n-5)*E_SN)**((n-3.)/2)/((3-delta)*(n-3)*M_ej)**((n-5.)/2) # equation(2) g_n

    n_csms = len(csms_param)
    for i in range(i_start, n_csms):
        if set_layer(csms_param, i) == -1:
            return -1

    tr0 = 0.0
    for i in range(n_csms):
        tr0 += csms_param[i]['kappa']*csms_param[i]['M_CSM_TH']/(beta*c*R_ph) #diffusion timescale of photon for reverse shock
    tr0 += parameters['kappa']*parameters['M_ej']/(beta*c*R_ph)

    for i in range(i_start, n_csms):
        if i < n_csms - 1:
            if csms_param[i]['R_in'] > csms_param[i+1]['R_in']:
                print('Inner radius (%f) of CSM %d is larger than the inner radius (%f) of next layer'\
                    %(csms_param[i]['R_in'], i, csms_param[i+1]['R_in']))
                return -1	# reject this parameter

            if csms_param[i]['R_CSM'] > csms_param[i+1]['R_in']:
                print('Outer radius (%f) of CSM %d is larger than the inner radius (%f) of next layer'\
                    %(csms_param[i]['R_CSM'], i, csms_param[i+1]['R_in']))
                #return -1

        tf0 = 0.0
        for j in range(i, n_csms):
            tf0 += csms_param[j]['kappa']*csms_param[j]['M_CSM_TH']/(beta*c*R_ph) #diffusion timescale of photon
        csms_param[i]['tf0'] = tf0
        csms_param[i]['tr0'] = tr0

    return 0

# input luminosity of FS
def L_FS(csms_param, t, i):
    beta_FS = csms_param[i]['ns']['beta_FS']
    s = csms_param[i]['ns']['s']
    A = csms_param[i]['ns']['A']
    q = csms_param[i]['q']
    t_in = csms_param[i]['t_in']
    t_FS = csms_param[i]['t_FS']
    g_n = csms_param[i]['g_n']
    v_SN = csms_param[i]['v_SN']

    if t < t_in:
        return 0.0
    LFS=2*PI/((n-s)**3.)*(g_n**((5.-s)/(n-s)))*(q**((n-5.)/(n-s)))*((n-3)**2)*\
        (n-5)*(beta_FS**(5-s))*(A**((5.-s)/(n-s)))*t**((2*n+6*s-n*s-15.)/(n-s))*Heaviside(t_FS - t + t_in)
    return LFS

# input luminosity of RS
def L_RS(csms_param, t, i):
    beta_RS = csms_param[i]['ns']['beta_RS']
    s = csms_param[i]['ns']['s']
    A = csms_param[i]['ns']['A']
    q = csms_param[i]['q']
    t_in = csms_param[i]['t_in']
    t_RS = csms_param[i]['t_RS']
    g_n = csms_param[i]['g_n']
    v_SN = csms_param[i]['v_SN']

    if t < t_in:
        return 0.0
    LRS=2*PI*(g_n)/(n-3.)/(beta_RS**(n-5))*(((3.-s)/(n-s))**3)*\
        ((q/A/g_n)**((n-5.)/(n-s)))*t**((2*n+6*s-n*s-15.)/(n-s))*\
        ((n-5)+2*(beta_RS/v_SN*(A*g_n/t**(3-s)/q)**(1./(n-s)))**(n-3))*Heaviside(t_RS - t + t_in)
    return LRS
    
def integralFun(x, csms_param, i):
    t0 = csms_param[i]['t0']
    epsilon = csms_param[i]['epsilon']
    func=(L_RS(csms_param, x, i)+L_FS(csms_param, x, i))*exp(x/t0)*epsilon 
    return func
    
#  Bolomertic luminosity of CSM model compared to observation    # fixed photosphere
def Interaction_Lum_layer_i(csms_param, t, time_delta, i):
    t0 = csms_param[i]['t0']
    t_in = csms_param[i]['t_in']

    if t < t_in:
        return 0.0
    I = quad(integralFun, t_in, t, args = (csms_param, i))[0]
    return I/t0 * exp(-t/t0)
    
def Interaction_Lum_i(csms_param, t, time_delta):
    lum = 0.0
    for i in range(len(csms_param)):
        lum += Interaction_Lum_layer(csms_param, t, time_delta, i)
    return lum

# Bolomertic luminosity of CSM model
# fixed photosphere
def Interaction_Lum_layer(csms_param, t, time_delta, i):
    tf0 = csms_param[i]['tf0']
    tr0 = csms_param[i]['tr0']
    t_in = csms_param[i]['t_in']
    epsilon_f = csms_param[i]['epsilon_f']
    epsilon_r = csms_param[i]['epsilon_r']

    L_R = 0.0
    L_F = 0.0
    if t > t_in:
        last = len(csms_param[i]['L_R']) - 1
        L_R0 = 0.0
        L_F0 = 0.0
        if last != -1:
            L_R0 = csms_param[i]['L_R'][last]
            L_F0 = csms_param[i]['L_F'][last]
        L_R = L_R0 + time_delta/tr0*(L_RS(csms_param, t, i)*epsilon_r - L_R0)
        L_F = L_F0 + time_delta/tf0*(L_FS(csms_param, t, i)*epsilon_f - L_F0)

    csms_param[i]['L_R'].append(L_R)
    csms_param[i]['L_F'].append(L_F)
    return L_R + L_F
    
def Interaction_Lum(csms_param, t, time_delta):
    lum = 0.0
    for i in range(len(csms_param)):
        lum += Interaction_Lum_layer(csms_param, t, time_delta, i)
    return lum

def handle_csm_para(k, v, csms_param):
    if k.find('csm_M') == 0:
        i = int(k[5:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['M_CSM'] = v
    elif k.find('csm_R_in') == 0:
        i = int(k[8:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['R_in'] = v
    elif k.find('csm_rho') == 0:
        i = int(k[7:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['rho_CSM'] = v
    elif k.find('csm_epsilon_f') == 0:
        i = int(k[13:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['epsilon_f'] = v
    elif k.find('csm_epsilon_r') == 0:
        i = int(k[13:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['epsilon_r'] = v
    elif k.find('csm_epsilon') == 0:
        i = int(k[11:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['epsilon_f'] = v
        csms_param[i-1]['epsilon_r'] = v
    elif k.find('csm_kappa') == 0:
        i = int(k[9:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['kappa'] = v
    elif k.find('csm_type') == 0:
        i = int(k[8:])
        while len(csms_param) < i:
            csms_param.append({})
        csms_param[i-1]['type'] = v

def SN_interact_lightcurve(parameters, z, DL, timeq, results, power = None):
    global n, M_ej, v_SN, R, E_SN, delta
    B_p = 1.09252e-15
    P0 = 1.1e10
    kappa = 0.34      # the optical opacity
    kappa_gamma_mag = 0.16
    kappa_gamma_Ni = 0.057
    M_Ni = 0.5e-6
    E_sn0 = 1.0e40 # initial kinetic energy
    E_th0 = 1.0e40 # initial thermal energy
    v_SN =  6.7e3  # the velocity of the SN
    csms_param = []

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
        elif k == 'v_SN':
            v_SN = parameters[k]
        elif k == 'R0':
            R0 = parameters[k]
        elif k == 'E_th0':
            E_th0 = parameters[k]
        elif k == 'ejecta_n':
            n = parameters[k]
        elif k == 'ejecta_delta':
            delta = parameters[k]
        elif k.find('csm_') == 0:
            handle_csm_para(k, parameters[k], csms_param)

    M_ej = ((M_ej*units.Msun).cgs).value
    M_Ni = ((M_Ni*units.Msun).cgs).value
    if not parameters.__contains__('kappa'):
        parameters['kappa'] = kappa
    v_SN = v_SN * 1e5
    M_ej_tot = M_ej
    for i in range(len(csms_param)):
        csms_param[i]['M_CSM'] *= Msun
        M_ej_tot += csms_param[i]['M_CSM']
        csms_param[i]['L_R'] = []
        csms_param[i]['L_F'] = []
        if csms_param[i]['type'] == 'wind':
            if int(n) == 6:
                csms_param[i]['ns'] = n6s2
            elif int(n) == 7:
                csms_param[i]['ns'] = n7s2
            elif int(n) == 8:
                csms_param[i]['ns'] = n8s2
            elif int(n) == 9:
                csms_param[i]['ns'] = n9s2
            elif int(n) == 12:
                csms_param[i]['ns'] = n12s2
            else:
                print('ejecta n', n, 'not supported')
                return (-1, csms_param)
        elif csms_param[i]['type'] == 'shell':
            if int(n) == 6:
                csms_param[i]['ns'] = n6s0
            elif int(n) == 7:
                csms_param[i]['ns'] = n7s0
            elif int(n) == 8:
                csms_param[i]['ns'] = n8s0
            elif int(n) == 9:
                csms_param[i]['ns'] = n9s0
            elif int(n) == 12:
                csms_param[i]['ns'] = n12s0
            else:
                print('ejecta n', n, 'not supported')
                return (-1, csms_param)
        else:
            print('CSM type', csms_param[i]['type'], 'not supported')
            return (-1, csms_param)
    print(csms_param)

    if results.__contains__('lum'):
        lum = results['lum']
    if results.__contains__('vel'):
        vels = results['vel']
    if results.__contains__('Teff'):
        Teffs = results['Teff']

    if set_csms(parameters, csms_param) == -1:
        return (-1, csms_param)

    Omega_0 = 2*PI/P0 # initial angular velocity

    t_p = 6.*I_ns*c**3 / (B_p**2 * R_ns**6 * Omega_0**2) # timescale of spin-down
    E_p = 0.5*I_ns*Omega_0**2 # initial rotation energy of NS
    E_inp = 0.0
    phi = 1.0
    x_ph = 1.0
    x_ph_old = 1.0
    E_atm = 0.0
    Volume_atm = 0.0
    Volume_atm_old = 0.0
    T_ion = 0.0
    E_mag_absorb = 0.0
    x_ph0 = 0.5
    R_ej = 1.0e10	# initial radius of ejecta
    R0 = R
    in_layer_i = -1
    E_SN_ej = E_SN	# E_SN_ej is set as a constant
    ratio0 = 1.0

    tau0 = kappa*M_ej_tot/beta/c/R0

    t = 0.0
    l = len(timeq)
 
    for i in range(l):
        time_delta = (timeq[i] - t)
        t = timeq[i]
    
        tau_gamma_Ni_SN = 3*kappa_gamma_Ni*M_ej_tot*x_ph/(4*PI*R**2)
        tau_gamma_mag_SN = 3*kappa_gamma_mag*M_ej_tot*x_ph/(4*PI*R**2)
        L_Ni = M_Ni*((eps_Ni-eps_Co)*exp(-t/tau_Ni)+eps_Co*exp(-t/tau_Co))
        L_inp_SN_Ni = L_Ni*(1-exp(-tau_gamma_Ni_SN))
        Lp = E_p/t_p/(1+t/t_p)**2
        L_inp_SN_mag = Lp*(1-exp(-tau_gamma_mag_SN))
    
        #photosphere
        Volume = 4*PI*R**3/3.0
        lam = 1./(M_ej_tot/Volume*kappa)
        x_ph_old = x_ph
        x_ph = 1-2./3 * lam/R
        x_ph_derivative = 0.0
        L_ph = 0.0
        if x_ph >= 0.0:
            x_ph_derivative = (x_ph-x_ph_old)/time_delta
            if x_ph > x_ph0:
                dphi = 1.0/E_th0*R/R0*(L_inp_SN_mag+L_inp_SN_Ni)/x_ph**3 \
                -R/R0/tau0*phi/x_ph**2 - 3*x_ph_derivative/x_ph*phi
                phi = phi+time_delta*dphi
                L_ph = E_th0/tau0*x_ph*phi
                ratio0 = L_ph / (L_inp_SN_mag + L_inp_SN_Ni)
            else:
                L_ph = (L_inp_SN_mag + L_inp_SN_Ni) * ratio0
                #phi = L_ph * tau0 / Eint0 / x_ph
        else:
            x_ph = -1e-6
    
    
        L_int = Interaction_Lum(csms_param, t, time_delta)
        L_ph += L_int
        if x_ph >= 0.0:
            Teff_ph = (L_ph/4/PI/sig_st/(R*x_ph)**2)**0.25
            if Teff_ph<T_ion:
                Teff_ph = T_ion
                x_ph = (L_ph/4/PI/sig_st/T_ion**4)**0.5 / R
    
        Volume_atm_old = Volume_atm
        Volume_atm = 4./3 * PI*R**3 * (1-x_ph**3)
        L_atm = E_atm * min(c/R/(1-x_ph),1./time_delta)
    
        if (L_atm<0.0 or L_ph<0.0):
            print('L_atm=%.3e;L_ph=%.3e;E_atm=%.3e;x_ph=%.3e\n' \
            %(L_atm,L_ph,E_atm,x_ph))
            #return
    
        tau_gamma_Ni = 3*kappa_gamma_Ni*M_ej_tot/4/PI/R**2
        tau_gamma_mag = 3*kappa_gamma_mag*M_ej_tot/4/PI/R**2
        L_Ni_atm = L_Ni*(exp(-tau_gamma_Ni_SN) - exp(-tau_gamma_Ni))
        L_mag_atm = Lp*(exp(-tau_gamma_mag_SN) - exp(-tau_gamma_mag))
        T_atm = (E_atm/arad/Volume_atm)**0.25
        p_atm = E_atm/Volume_atm/3
        E_atm = E_atm-p_atm*(Volume_atm-Volume_atm_old)\
        + time_delta*(L_Ni_atm+L_mag_atm-L_atm\
        - 4*PI*R**3 * x_ph**2 * arad*Teff_ph**4 * x_ph_derivative)
    
        if E_atm<0.0:
            E_atm = 0.0
        Teff = ((L_ph*Teff_ph**4 + L_atm*T_atm**4)/(L_ph+L_atm))**0.25
        if in_layer_i < len(csms_param) - 1:
            if R_ej > csms_param[in_layer_i + 1]['R_in']:
                E_SN = E_SN_ej+E_inp
                if in_layer_i > -1:
                    M_ej += csms_param[in_layer_i]['M_CSM']
                    v_SN = sqrt(2.*(5-delta)*(n-5)/((3-delta)*(n-3)) * (E_SN_ej+E_inp)/M_ej)/x_0
                in_layer_i += 1
                set_csms(parameters, csms_param, in_layer_i+1)
                csms_param[in_layer_i]['t_in'] = t	# overwrite R_in/v_SN

        L_absorb = Lp*(1-exp(-tau_gamma_mag))
        E_inp += (L_absorb-L_ph-L_atm-L_int)*time_delta
        R_ej += v_SN*time_delta
        if R_ej > R:
            R = R_ej
        E_mag_absorb += L_absorb*time_delta
        #v_SN = sqrt(10./3 * (E_SN_ej+E_inp)/M_ej) # n->inf., delta=0
        v_SN = sqrt(2.*(5-delta)*(n-5)/((3-delta)*(n-3)) * (E_SN_ej+E_inp)/M_ej)/x_0
    
        if 'lum' in locals().keys():
            lum.append(L_ph + L_atm) # log
        if 'vels' in locals().keys():
            vels.append(max(v_SN * x_ph/1e5, 0.0)) # in km/s
        if 'Teffs' in locals().keys():
            Teffs.append(Teff)
        if results.__contains__('R_ph'):
            results['R_ph'].append(R*x_ph)

    return (0, csms_param)

if len(sys.argv) > 1:
    (bounds, values, paras) = load_mc_conf(sys.argv[1])
    mergep = dict(paras, **values)
    print(mergep)

z = 0.0
DL = 1e18

output = open('output.txt','w')
temp1 = '#time\tlum\tTeff\tv_ph'
temp2 = '\tR_ph'
output.write(temp1+temp2)

results = {}
results['lum'] = []
results['Teff'] = []
results['vel'] = []	# photospheric velocity
results['R_ph'] = []

t = 1.0e-6
delta_t = 1.0e-3
timeq = []
while t<2.0e8:
    if t<1e-5:
        t = delta_t
        time_delta = delta_t
    elif t<1:
        t += delta_t
        time_delta = delta_t
    else:
        time_delta = t*delta_t
        t += time_delta
    
    timeq.append(t)

v, csmp = SN_interact_lightcurve(mergep, z, DL, timeq, results, power = None)
clen = len(csmp)
for j in range(clen):
    output.write('\t%dth-%s\t%dth-%s' % (j+1, 'fshock', j+1, 'rshock'))
output.write('\n')

l = len(timeq)
for i in range(l):
    o1,o2,o3 = timeq[i]/24./3600., results['lum'][i], results['Teff'][i]
    o4,o5 = results['vel'][i], results['R_ph'][i]
    output.write('%.6e\t%.6e\t%.6e\t%.6e\t%.6e' \
%(o1,o2,o3,o4,o5))
    for j in range(clen):
        output.write('\t%.6e\t%.6e' % (csmp[j]['L_F'][i], csmp[j]['L_R'][i]))
    
    output.write('\n')
output.close()
