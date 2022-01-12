#!/usr/bin/env python
# -*- coding: utf-8 -*-

' astrophysical module '

import numpy as np
import astropy.units as units
import astropy.constants as constants
from math import exp,sqrt

# constants (units:cgs)
c = constants.c.cgs.value
h = constants.h.cgs.value
k_B = constants.k_B.cgs.value
sigma_Stefan = constants.sigma_sb.cgs.value
sigma_T = constants.sigma_T.cgs.value
m_p = constants.m_p.cgs.value
m_e = constants.m_e.cgs.value
q_e = constants.e.esu.value
eV = constants.e.si.value*1e7
pc = constants.pc.cgs.value
Mpc = 1e6*pc
PI = np.pi
arad = 7.56e-15
sig_st = 5.67e-5
Msun = ((1.0*units.Msun).cgs).value

# model constant (units:cgs)
tau_Co = ((111.3*units.day).cgs).value
tau_Ni = ((8.8*units.day).cgs).value
eps_Co = 6.8e9
eps_Ni = 3.9e10
lambda_U=3660
lambda_B=4380
lambda_V=5450
lambda_R=6410
lambda_I=7980

# SDSS, on the AB magnitude system
lambda_u = 3550
lambda_g = 4770
lambda_r = 6230
lambda_i = 7620
lambda_z = 9130

DL = 10*constants.pc.cgs.value

# neutron star's mass, radius, moment of inertia
M_ns = ((1.41*units.Msun).cgs).value
R_ns = ((10*units.km).cgs).value
I_ns = 0.4*M_ns*R_ns**2

def U_in_Hz():
    return c / (lambda_U * 1e-8)

def B_in_Hz():
    return c / (lambda_B * 1e-8)

def V_in_Hz():
    return c / (lambda_V * 1e-8)

def R_in_Hz():
    return c / (lambda_R * 1e-8)

def I_in_Hz():
    return c / (lambda_I * 1e-8)

def u_in_Hz():
    return c / (lambda_u * 1e-8)

def g_in_Hz():
    return c / (lambda_g * 1e-8)

def r_in_Hz():
    return c / (lambda_r * 1e-8)

def i_in_Hz():
    return c / (lambda_i * 1e-8)

def z_in_Hz():
    return c / (lambda_z * 1e-8)

def AB_mag(Fv):
    return -2.5*np.log10(Fv)-48.6
    
def U_mag(Fv):
    return AB_mag(Fv)-0.770+0.002

def U_flux(Umag):
    return 10**(-0.4*(Umag + 48.6 + 0.770 - 0.002))

def B_mag(Fv):
    return AB_mag(Fv) +0.120+0.002

def B_flux(Bmag):
    return 10**(-0.4*(Bmag + 48.6 - 0.120 - 0.002))

def V_mag(Fv):
    return AB_mag(Fv) -0.0+0.002
    
def V_flux(Vmag):
    return 10**(-0.4*(Vmag + 48.6 + 0.0 - 0.002))

def R_mag(Fv):
    return AB_mag(Fv) -0.186+0.002
    
def R_flux(Rmag):
    return 10**(-0.4*(Rmag + 48.6 + 0.186 - 0.002))

def I_mag(Fv):
    return AB_mag(Fv) -0.444+0.002

def I_flux(Imag):
    return 10**(-0.4*(Imag + 48.6 + 0.444 - 0.002))

def bb_Fv_nu(nu, Teff, R, DL):
    if h*nu/k_B/Teff > 700:
        print("underflow in bb_Fv_nu")
        return 1e-300
    return PI*(R/DL)**2*2*h*nu**3/c**2/(exp(h*nu/k_B/Teff)-1)

def bb_Fv_lambda_A(lambda_A, Teff, R, DL):
    nu = c / (lambda_A * 1e-8)
    return bb_Fv_nu(nu, Teff, R, DL)

def syn_F_nu(F_nu_max, nu_a, nu_m, nu_c, p, nu):
    # slow cooling
    # Case II: nu_m < nu_a < nu_c
    if (nu_m <= nu_a and nu_a <= nu_c):
        if (nu < nu_m):
            F_nu = F_nu_max*(nu/nu_m)**2*(nu_m/nu_a)**(p/2.0+2)
        elif (nu < nu_a):
            F_nu = F_nu_max*(nu/nu_a)**2.5*(nu_a/nu_m)**(-(p-1)/2)
        elif (nu < nu_c):
            F_nu = F_nu_max * (nu/nu_m)**(-(p-1)/2.0)
        else:
            F_nu = F_nu_max * (nu_c / nu_m) ** (-(p-1)/2.0)\
                * (nu / nu_c) ** (-p / 2.0)

        return F_nu

    # Case I: nu_a < nu_m < nu_c
    if (nu_a <= nu_m and nu_m <= nu_c):
        if (nu < nu_a):
            F_nu = F_nu_max*(nu/nu_a)**2*(nu_a/nu_m)**(1./3)
        elif (nu < nu_m):
            F_nu = F_nu_max*(nu/nu_m)**(1./3)
        elif (nu < nu_c):
            F_nu = F_nu_max*(nu/nu_m)**(-(p-1)/2.)
        else:
            F_nu = F_nu_max * (nu_c / nu_m) ** (-(p-1)/2.)\
                * (nu / nu_c) ** (-p / 2.0)

        return F_nu

    # Case III: nu_m < nu_c < nu_a
    if (nu_m <= nu_c and nu_c <= nu_a):
        if (nu < nu_m):
            F_nu = F_nu_max*(nu/nu_m)**2 * (nu_m/nu_a)**((p+4)/2.) \
              * (nu_a/nu_c)**(-1./2)
        elif (nu < nu_a):
            F_nu = F_nu_max*(nu/nu_a)**2.5 * (nu_a/nu_c)**(-p/2.) \
              * (nu_c/nu_m)**((p-1)/2.)
        else:
            F_nu = F_nu_max*(nu/nu_c)**(-p/2.) * (nu_c/nu_m)**(-(p-1)/2.)

        return F_nu

    # fast cooling
    # Case II: nu_c < nu_a < nu_m
    if (nu_c <= nu_a and nu_a <= nu_m):
        if (nu < nu_c):
            F_nu = F_nu_max*(nu_c/nu_a)**3*(nu/nu_c)**2
        elif (nu < nu_a):
            F_nu = F_nu_max*(nu/nu_a)**(5./2)*(nu_a/nu_c)**(-1./2)
        elif (nu < nu_m):
            F_nu = F_nu_max*(nu/nu_c)**(-1./2)
        else:
            F_nu = F_nu_max*(nu_m/nu_c)**(-1./2)*(nu/nu_m)**(-p/2.0)

        return F_nu

    # Case I: nu_a < nu_c < nu_m
    if (nu_a <= nu_c and nu_c <= nu_m):
        if (nu < nu_a):
            F_nu = F_nu_max*(nu_a/nu_c)**(1./3)*(nu/nu_a)**2
        elif (nu < nu_c):
            F_nu = F_nu_max*(nu/nu_c)**(1./3)
        elif (nu < nu_m):
            F_nu = F_nu_max*(nu/nu_c)**(-1./2)
        else:
            F_nu = F_nu_max*(nu_m/nu_c)**(-1./2)*(nu/nu_m)**(-p/2.)

        return F_nu

    # Case III: nu_c < nu_m < nu_a
    if (nu_c <= nu_m and nu_m <= nu_a):
        if (nu < nu_c):
            F_nu = F_nu_max*(nu/nu_c)**2 * (nu_c/nu_a)**3 \
              * (nu_a/nu_m)**(-(p-1)/2.)
        elif (nu < nu_a):
            F_nu = F_nu_max*(nu/nu_a)**2.5 * (nu_a/nu_m)**(-p/2) \
              * (nu_m/nu_c)**(-1./2)
        else:
            F_nu = F_nu_max*(nu/nu_m)**(-p/2.) *(nu_m/nu_c)**(-1./2)

    return F_nu

def Heaviside(t):
    if t > 0.0:
        return 1.0
    return 0.0

def magnetar_power(t, P0, B_p):
    Omega_0 = 2*PI/P0
    E_p = 0.5*I_ns*Omega_0**2
    tau_ns = 6.*I_ns*c**3 / (B_p**2 * R_ns**6 * Omega_0**2)
    return E_p/tau_ns/(1+t/tau_ns)**2

need_convert = []

def load_mc_conf(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    bounds = {}
    values = {}
    paras = {}
    for l in lines:
        l = l.split("#")[0].strip()
        if not l:
            continue
        (name, value) = l.split("=")
        name = name.strip()
        
        tup = value.split(",")
        if len(tup) == 3:
            (low, high, para) = tup
            low = float(low.strip())
            high = float(high.strip())
            bounds[name] = [low, high]
            paras[name] = float(para.strip())
        elif len(tup) == 1:
            v = value.strip()
            if v.isalpha():
                values[name] = v
            else:
                values[name] = float(v)
        else:
            raise ValueError('Invalid config item', l)
            
    f.close()
    return (bounds, values, paras)
 
def test():
    print("astrophysical module")

if __name__=='__main__':
    test()

