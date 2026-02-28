import sympy as sy
import numpy as np
from sympy.physics.wigner import wigner_3j,wigner_6j,wigner_9j

c_from_inversecm_to_MHz = 29979.2458
au_to_debye = 2.54174636

params_general = {
'mu_B': 1.399624494, #MHz/Gauss
'g_S': 2.0023,
'g_L': 1,
'2/e0c': 2*3.76730314*10**(10),# (V/cm)^2/(W/um^2)
}


### YbOH Parameters ###

params_174X000 = {
'Be': 7348.40053,
'Gamma_SR': -81.150,
'bF': 4.80,
'c': 2.46,
'b': (4.80-2.46/3),
'D': 0.006084,
'Gamma_D': 0.00476,
'muE': 1.9*0.503412, #Debye in MHz/(V/cm)
'g_S_eff': 2.0023
}

# { #YbF
# 'Be': 7233.8271,
# 'Gamma_SR': -13.41679,
# 'bF': 170.26374,
# 'c': 85.4028,
# 'muE': 3.91*0.503412 #Debye in MHz/(V/cm)
# }


'''
#This is the real YbOH 174X010 parameter
params_174X010 = {
'Be': 0.244457*c_from_inversecm_to_MHz,
'Gamma_SR': -0.00296*c_from_inversecm_to_MHz,
'Gamma_Prime': 5.2e-04*c_from_inversecm_to_MHz,
#'bF': 1.601e-04*c_from_inversecm_to_MHz,
#'c': 8.206e-05*c_from_inversecm_to_MHz,
'bF': 4.07,
'c': 3.49,
'q_lD': -4.01e-04*c_from_inversecm_to_MHz, #Should be minus if neg parity lower and parity is (-1)^(J-l-S), but for modeling I treat this as positive for now....
'p_lD': -3.6e-04*c_from_inversecm_to_MHz,
'muE': 2.16*0.503412,
'g_S_eff': 2.07
}

'''


#This is the YbOH 174X010 parameter to try simulating magic better
params_174X010 = {
'Be': 0.24445*c_from_inversecm_to_MHz,
'Gamma_SR': -0.0029255*c_from_inversecm_to_MHz,
'Gamma_Prime': 5.2e-04*c_from_inversecm_to_MHz,
#'bF': 1.601e-04*c_from_inversecm_to_MHz,
#'c': 8.206e-05*c_from_inversecm_to_MHz,
'bF': 4.07,
'c': 3.49,
'q_lD': -4.01e-04*c_from_inversecm_to_MHz -0.2*0.8, #Should be minus if neg parity lower and parity is (-1)^(J-l-S), but for modeling I treat this as positive for now....
'p_lD': -3.6e-04*c_from_inversecm_to_MHz +1*0.452,
'muE': (2.16+1*0.01)*0.503412,
'g_S_eff': 2.07+0
}



'''
params_174X010 = {
'Be': 0.24445*c_from_inversecm_to_MHz + 0.2,
'Gamma_SR': -0.0029255*c_from_inversecm_to_MHz + 0.9,
'Gamma_Prime': 5.2e-04*c_from_inversecm_to_MHz,
#'bF': 1.601e-04*c_from_inversecm_to_MHz,
#'c': 8.206e-05*c_from_inversecm_to_MHz,
'bF': 4.07,
'c': 3.49,
'q_lD': -4.01e-04*c_from_inversecm_to_MHz +0.2*1, #Should be minus if neg parity lower and parity is (-1)^(J-l-S), but for modeling I treat this as positive for now....
'p_lD': -3.6e-04*c_from_inversecm_to_MHz +1*1,
'muE': (2.16-1*0.01)*0.503412,
'g_S_eff': 2.07+0.02
}
'''


#This is the SrOH_88X010 parameter
params_SrOH_88X010 = {
'Be': 7452.2430,
'Gamma_SR': 72.24,
'Gamma_Prime': 0,
'bF': 0.00016*c_from_inversecm_to_MHz,
'c': 0.000082*c_from_inversecm_to_MHz,
'q_lD': -11.8546,
'p_lD': -1.03,
'muE': 1.900*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.0023
}


params_174A000 = {
'Be': 0.253052*c_from_inversecm_to_MHz,
'ASO': 1350*c_from_inversecm_to_MHz, #Fixed from 1350 cm^-1
'bF': 2.33e-06*c_from_inversecm_to_MHz,     # extrapolated from YbF
'c': 6.0e-06*c_from_inversecm_to_MHz,     # extrapolated from YbF
'p+2q': -0.43807*c_from_inversecm_to_MHz,
'q':0*c_from_inversecm_to_MHz,
'D': 2.319*(10**(-7))*c_from_inversecm_to_MHz,
'p2q_D':3.8*(10**(-6))*c_from_inversecm_to_MHz,
'g_lp': -0.865,
'muE': 0.43*0.503412
}


# Yuiki is using '174A00' in YbF for 3Delta1 state


#ThF+ parameter
params_YbF_174A000 = {
'Be': 0.24261*c_from_inversecm_to_MHz, #value from Kia Boon's email
'A_parallel': -20.1,     
'Omega': 5.29,
'g_eff': 0.0149,
'muE': 3.37*0.503412
}

'''

#HfF+ parameter
params_YbF_174A000 = {
'Be': 8983,
'A_parallel': -62.0,     
'Omega': 0.74,
'g_eff': 0.0031,
'muE': 1.79
}
'''

params_173X000 = { # all units MHz except for muE
'Be': 7351.24,
'Gamma_SR': -81.06,
'bFYb': -1883.21,
'cYb': -81.84,
'bFH': 4.80,
'cH': 2.46,
'e2Qq0': -3318.70,
'muE': 1.9*0.503412 #Debye in MHz/(V/cm)
}

params_171X000 = {
'Be': 7359.81,
'Gamma_SR': -80.85,
'bFYb': 6823.58,
'cYb': 233.84,
'bFH': 4.80,
'cH': 2.46,
'e2Qq0': 0,
'muE': 1.9*0.503412 #Debye in MHz/(V/cm)
}


#'''
#This is the real YbOH 171X010 parameter
params_171X010 = {
'Be': 0.24486571*c_from_inversecm_to_MHz,
'Gamma_SR': -0.00300962*c_from_inversecm_to_MHz,
'Gamma_Prime': 3.8539e-04*c_from_inversecm_to_MHz,
'bFYb': 0.22666508*c_from_inversecm_to_MHz,
'cYb': 0.00938256*c_from_inversecm_to_MHz,
#'bFH': 0.00016*c_from_inversecm_to_MHz,
#'cH': 0.000082*c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 3.49,
'e2Qq0': 0*c_from_inversecm_to_MHz,
'q_lD': 4.2158e-04*c_from_inversecm_to_MHz,
'p_lD': 3.5302e-04*c_from_inversecm_to_MHz,
'muE': 2.16*0.503412 , #Debye in MHz/(V/cm)
'g_S_eff': 2.07
}
'''


#This is the BaOH_137X010 parameter
params_BaOH_137X010 = {

params_171X010 = {
'Be': 6498.926,
'Gamma_SR': 72.01,
'Gamma_Prime': 0,
'bFYb': 2200.2,
'cYb': 0,
#'bFH': 0.00016*c_from_inversecm_to_MHz,
#'cH': 0.000082*c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 3.49,
'e2Qq0': -394.2,
'q_lD': 9.4932,
'p_lD': 0,
'muE': 1.43*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.0023
}

'''


#This is the LuOH+_175X010 parameter
params_LuOH_175X010 = {
'Be': 0.288*c_from_inversecm_to_MHz,
'Gamma_SR': 0.0046*c_from_inversecm_to_MHz,
'Gamma_Prime': 0,
'bFYb': 7956.6666666,
'cYb': 278,
'bFH': 0.2,
'cH': 3.3,
'e2Qq0': -5012,
'q_lD': 23.5,
'p_lD': 0,
'muE': 0.55*au_to_debye*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.0023
}




'''
#This is a test for NSDPV
params_171X010 = {
'Be': 0.24480635*c_from_inversecm_to_MHz,
'Gamma_SR': 20,
'Gamma_Prime': 0,
'bFYb': 30,
'cYb': 0,
'bFH': 0,
'cH': 0,
'e2Qq0': 0,
'q_lD': 100,
'p_lD': 0,
'muE': 2.10*0.503412 #Debye in MHz/(V/cm)
}

'''



#This is the new YbOH 173X010 parameter
params_173X010 = { # all units MHz except for muE
'Be': 0.2446402 *c_from_inversecm_to_MHz,
'Gamma_SR': -0.00337366  *c_from_inversecm_to_MHz,
'Gamma_Prime': 0.0007639237 *c_from_inversecm_to_MHz,
'bFYb': -0.06277542 *c_from_inversecm_to_MHz,
'cYb': -0.002510204 *c_from_inversecm_to_MHz,
'bFH': 4.07 -1.6,
'cH': 5.49 - (0.3 * 1.6),
'e2Qq0': -0.108204 *c_from_inversecm_to_MHz + 2,  # Temoporary test
'q_lD': 0.0004055556*c_from_inversecm_to_MHz + 12.158251*0.15 ,
'p_lD': 0,
'muE': 0.97 * 2.16*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.07
}

'''
#This is not correct because this includes p term which deos not agree with two-photon Stark spectrum at all. This was only temporalily used for the YbOH 173X010 parameters.
params_173X010 = { # all units MHz except for muE
'Be': 0.2446402 *c_from_inversecm_to_MHz,
'Gamma_SR': -0.002799349 *c_from_inversecm_to_MHz,
'Gamma_Prime': 0.0006994695 *c_from_inversecm_to_MHz,
'bFYb': -0.06278075 *c_from_inversecm_to_MHz,
'cYb': -0.002546895 *c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 5.49,
'e2Qq0': -0.108347 *c_from_inversecm_to_MHz,  # Temoporary test
'q_lD': 4.1815e-04*c_from_inversecm_to_MHz,
'p_lD': 0.0004364426*c_from_inversecm_to_MHz,
'muE': 2.16*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.07
}
'''

'''
#This is the YbOH 173X010 parameter from the PRL paper with some updates I think
params_173X010 = { # all units MHz except for muE
'Be': 0.24464027*c_from_inversecm_to_MHz,
'Gamma_SR': -0.00290825*c_from_inversecm_to_MHz,
'Gamma_Prime': 4.7479e-04*c_from_inversecm_to_MHz,
'bFYb': -0.06274229*c_from_inversecm_to_MHz,
'cYb': -0.00307411*c_from_inversecm_to_MHz,
#'bFH': 0.00016*c_from_inversecm_to_MHz,
#'cH': 0.000082*c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 5.49,
#'e2Qq0': -0.11079907*c_from_inversecm_to_MHz,  # True value
'e2Qq0': -3154.2,  # Temoporary test
'q_lD': 4.1676e-04*c_from_inversecm_to_MHz,
'p_lD': 4.2584e-04*c_from_inversecm_to_MHz,
'muE': 2.16*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.07
}
'''


'''
#This is a test for varying YbOH 173X010 parameters by 1 sigma error
params_173X010 = { # all units MHz except for muE
'Be': 0.24464027*c_from_inversecm_to_MHz ,
'Gamma_SR': -0.00290825*c_from_inversecm_to_MHz,
'Gamma_Prime': 4.7479e-04*c_from_inversecm_to_MHz,
'bFYb': -0.06274229*c_from_inversecm_to_MHz,
'cYb': -0.00307411*c_from_inversecm_to_MHz,
#'bFH': 0.00016*c_from_inversecm_to_MHz,
#'cH': 0.000082*c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 3.49,
'e2Qq0': -0.11079907*c_from_inversecm_to_MHz,
'q_lD': 4.1676e-04*c_from_inversecm_to_MHz + 0.25,
'p_lD': 4.2584e-04*c_from_inversecm_to_MHz,
'muE': 2.16*0.503412,  #Debye in MHz/(V/cm)
'g_S_eff': 2.07
}
'''





params_173A000 = {
'Be': 0.253185*c_from_inversecm_to_MHz,
'ASO': 1350*c_from_inversecm_to_MHz, #Actually 4.047*10**7,
'h1/2Yb': -0.00422*c_from_inversecm_to_MHz,
'dYb': -0.00873*c_from_inversecm_to_MHz,
'bFH': 0.07, #0*c_from_inversecm_to_MHz, #0.07,     # extrapolated from YbF
'cH': -0.18, #0*c_from_inversecm_to_MHz, #-0.18,     # extrapolated from YbF
'e2Qq0': -0.0642*c_from_inversecm_to_MHz,
'p+2q': -0.438457*c_from_inversecm_to_MHz,
'D': 2.405*(10**(-7))*c_from_inversecm_to_MHz,
'p2q_D':0*3.8*(10**(-6))*c_from_inversecm_to_MHz,
'g_lp': -0.865,
'muE': 0.43*0.503412
}

params_171A000 = {
'Be': 0.253435*c_from_inversecm_to_MHz,
'ASO': 1350*c_from_inversecm_to_MHz, #Actually 4.047*10**7,
'h1/2Yb': 0.0148*c_from_inversecm_to_MHz,
'dYb': 0.03199*c_from_inversecm_to_MHz,
'bFH': 0*c_from_inversecm_to_MHz, #0.07,     # extrapolated from YbF
'cH': 0*c_from_inversecm_to_MHz, #-0.18,     # extrapolated from YbF
'e2Qq0': 0*c_from_inversecm_to_MHz,
'p+2q': -0.438667*c_from_inversecm_to_MHz,
'D': 2.608*(10**(-7))*c_from_inversecm_to_MHz,
'p2q_D':0*3.8*(10**(-6))*c_from_inversecm_to_MHz,
'g_lp': -0.865,
'muE': 0.43*0.503412
}


### CaOH Parameters ###
#X(000) Taken from Louis Baum thesis and Steimle papers from 90s
#Vibrational states from Fletcher et al

params_40X000 = {
'Be': 10023.0841,
'D': 1.154*10**-2,
'Gamma_SR': 34.7593,
'bF': 2.602,
'c': 2.053,
'b': (2.602-2.053/3),
'muE': 1.465*0.503412 #Debye in MHz/(V/cm)
}

#Whenever possible, constants are taken from Fletcher et all, Milimeter Wave Hydroxide paper
# params_40X010 = {
# 'Be': 9996.7518,
# 'D': 0.0117696,
# 'Gamma_SR': 35.051,
# 'Gamma_Prime': 0,
# 'bF': 2.244, #2.602, #2.29 fit?
# 'c': 2.607, #2.053, #2.52 fit?
# # 'b': (2.29-2.52/3),
# 'p_lD': -0.05,
# 'q_lD': -21.6492,
# 'q_lD_D': 6.4*10**-5,
# 'muE': 1.465*0.503412,
# 'azz': 3.33441*10**(-8)*5.525**2/25.875, #Debye^2/MHz in units of MHz/(V/cm)^2. Using Lan values
# 'axxyy': 9.46049*10**(-8)*6.165**2/29.34
# }

#Coxon parameters for comparison:
params_40X010 = {
'Be': 9996.82,
'D': 0.008823,
'Gamma_SR': 35.5,
'Gamma_Prime': 0,
'bF': 2.294,#2.247, #2.293,#2.2445, #2.602
'c': 2.524,#2.601,#2.522,##2.6074, #2.053
# 'b': (2.29-2.52/3),
'p_lD': -0.00,
'q_lD': -21.53,
# 'q_lD_D': 6.4*10**-5,
'muE': 1.465*0.503412,
'azz': 3.5555*10**(-8), #From Lan for X(000)
'axxyy': 1.1718*10**(-7) #From Lan for X(000)
# 'azz': 3.33441*10**(-8)*5.525**2/25.875, #Calc from Lan's static #Debye^2/MHz in units of MHz/(V/cm)^2. Using Lan values
# 'axxyy': 9.46049*10**(-8)*6.165**2/29.34 #Calc from Lan's static
}


params_40A000 = {
'Be': 10229.52,
'ASO': 2.00316*10**6,
'a': 0,         # extrapolated from YbF
'bF': 0.07,     # extrapolated from YbF
'c': -0.18,     # extrapolated from YbF
'p+2q': -1305 ,
'q': -9.764,
'g_lp': -0.865, #Unknown
'muE': 0.836*0.503412
}

params_40B000 = {
'Be': 10175.2,
'Gamma_SR': -1307.54,
'bF': 2.602*0.04, #Hyperfine extrapolated from CaF A state to X state ratio
'c': 2.053*0.04,
'b': (2.602-2.053/3)*0.04,
'muE': 0.744*0.503412 #Debye in MHz/(V/cm)
}




params_YbF_173X000 = { # all units MHz except for muE
'Be': 0.2414348*c_from_inversecm_to_MHz,
'Gamma_SR': -0.0004464*c_from_inversecm_to_MHz ,
'bFYb': -0.06704*c_from_inversecm_to_MHz ,
'cYb': -0.002510*c_from_inversecm_to_MHz ,
'bFH': 0.005679*c_from_inversecm_to_MHz ,
'cH': 0.002849*c_from_inversecm_to_MHz,
'e2Qq0': -0.10996*c_from_inversecm_to_MHz ,
'muE': 3.91*0.503412, #Debye in MHz/(V/cm)
'g_S_eff': 2.0023,
}


'''
#This is the real YbF 171X000 parameter
params_YbF_171X000 = {
'Be': 0.24171098*c_from_inversecm_to_MHz,
'Gamma_SR': -0.000448*c_from_inversecm_to_MHz,
'bFYb': 0.24260*c_from_inversecm_to_MHz,
'cYb': 0.009117*c_from_inversecm_to_MHz,
'bFH': 0.005679*c_from_inversecm_to_MHz,
'cH': 0.002849*c_from_inversecm_to_MHz,
'e2Qq0': 0*c_from_inversecm_to_MHz,
'muE': 3.91*0.503412, #Debye in MHz/(V/cm)
'g_S_eff': 2.0023,
}


#This is the RaF 225X000 parameter
params_RaF_225X000 = {
'''
params_YbF_171X000 = {
'Be': 0.19207*c_from_inversecm_to_MHz,
'Gamma_SR': 0.00585*c_from_inversecm_to_MHz,
'bFYb': -0.5445*c_from_inversecm_to_MHz,
'cYb': -0.025*c_from_inversecm_to_MHz,
'bFH': 96.3333,
'cH': 19,
'e2Qq0': 0*c_from_inversecm_to_MHz,
'muE': 1.54*au_to_debye*0.503412, #Debye in MHz/(V/cm)
'g_S_eff': 2.0023,
}



#This is the RaOH 225X000 parameter
#params_RaOH_225X000 = {
'''
params_171X010 = {
'Be': 0.19207*c_from_inversecm_to_MHz,
'Gamma_SR': 0.00585*c_from_inversecm_to_MHz,
'Gamma_Prime': 0,
'bFYb': -0.5445*c_from_inversecm_to_MHz,
'cYb': -0.025*c_from_inversecm_to_MHz,
'bFH': 4.07,
'cH': 3.49,
'q_lD': 14.467/2,
'p_lD': 0,
'e2Qq0': 0*c_from_inversecm_to_MHz,
'muE': 1.54*au_to_debye*0.503412, #Debye in MHz/(V/cm)
'g_S_eff': 2.0023,
}
'''


YbOH_params = {
'174X000':{**params_174X000,**params_general},
'174X010':{**params_174X010,**params_general},
'173X000':{**params_173X000,**params_general},
'173X010':{**params_173X010,**params_general},
'174A000':{**params_174A000,**params_general},
'173A000':{**params_173A000,**params_general},
'171A000':{**params_171A000,**params_general},
'171X000':{**params_171X000,**params_general},
'171X010':{**params_171X010,**params_general},
}

CaOH_params = {
'40X000':{**params_40X000,**params_general},
'40X010':{**params_40X010,**params_general},
'40A000':{**params_40A000,**params_general},
'40B000':{**params_40B000,**params_general},
}


YbF_params = {
'173X000':{**params_YbF_173X000,**params_general},
'171X000':{**params_YbF_171X000,**params_general},

'174X000':{**params_174X000,**params_general},
'174A000':{**params_YbF_174A000,**params_general},
'173A000':{**params_173A000,**params_general},
'171A000':{**params_171A000,**params_general},
}

all_params = {
'YbOH': YbOH_params,
'CaOH': CaOH_params,
'YbF': YbF_params}
