import numpy as np
import sympy as sy
from functools import partial
from molecule_parameters import params_general
from matrix_elements import MQM_bBS, NSM_bBS, EDM_bBS, NSDPV_bBS, EDM_even_aBJ_3Delta1, Sz_bBJ,T2QM_bBS,b2a_matrix,decouple_b_even,bBS_2_bBJ_matrix,recouple_J_even,decouple_b_I_even


def _safe_element_call(element, q_args):
    try:
        return element(**q_args)
    except Exception as e:
        import traceback
        name = getattr(element, "__name__", str(element))
        print(f"Error evaluating matrix element '{name}' with q_args:")
        # print keys sorted for readability
        for k in sorted(q_args.keys()):
            print(f"  {k}: {q_args[k]!r}")
        traceback.print_exc()
        raise

def H_even_X(q_numbers,params,matrix_elements,symbolic=True,E=0,B=0,M_values='all',precision=5,trap=False,theta_num=None):
    q_str = list(q_numbers)     # Get keys for quantum number dict
    if symbolic:
        Ez,Bz = sy.symbols('E_z B_z')
        if trap:
            I0 = sy.symbols('I_0')
            if theta_num is None:
                theta = sy.symbols('theta')
            else:
                theta = theta_num
            T00_E = -1/np.sqrt(3)*params['2/e0c']
            T20_E = (3*sy.cos(theta)**2 - 1)/np.sqrt(6)*params['2/e0c']
            T2p1_E = -(sy.sin(2*theta)/2)*params['2/e0c']
            T2m1_E = -T2p1_E
            T22_E = (sy.sin(theta)**2)/2*params['2/e0c']
        size = len(q_numbers[q_str[0]])
        # Need to construct empty matrices to fill with matrix elements
        # Sympy does not like numpy arrays, so convert to list
        H0 = np.zeros((size,size)).tolist()
        V_B = np.zeros((size,size)).tolist()
        V_E = np.zeros((size,size)).tolist()
        V_trap = np.zeros((size,size)).tolist()
        if params.get('D') is not None:
            N0 = np.zeros((size,size))
        if params.get('q_lD_D') is not None:
            lD0 = np.zeros((size,size))
        if params.get('Gamma_D') is not None:
            SR0 = np.zeros((size,size))
        #Iz = np.zeros((size,size)).tolist()
        #Sz = np.zeros((size,size)).tolist()
        for i in range(size):
            for j in range(size):
                # State out is LHS of bra ket, state in is RHS
                state_out = {q+'0':q_numbers[q][i] for q in q_str}
                state_in = {q+'1':q_numbers[q][j] for q in q_str}
                q_args = {**state_out,**state_in}
                elements = {term: sy.N(_safe_element_call(element, q_args)) for term, element in matrix_elements.items()}
                # The Hamiltonian
                H0[i][j] = params['Be']*elements['N^2'] + params['Gamma_SR']*elements['N.S'] + \
                    params['bF']*elements['I.S'] + params['c']/3*np.sqrt(6)*elements['T2_0(I,S)']
                if params.get('q_lD') is not None:
                    H0[i][j] += params['p_lD']*elements['l doubling p'] - params['q_lD']*elements['l doubling q'] -params['Gamma_SR']*elements['NzSz']+params['Gamma_Prime']*elements['NzSz']
                    
                    #old: params['q_lD']/2*elements['l-doubling']
                    # H0[i][j] += -params['q_lD']/2*elements['l-doubling'] - params['Gamma_SR']*elements['NzSz']+params['Gamma_Prime']*elements['NzSz']
                    
                if M_values!='none':
                    V_B[i][j]+=params['g_S_eff']*params['mu_B']*elements['ZeemanZ']
                    V_E[i][j]+=-params['muE']*elements['StarkZ']
                if M_values!='none' and trap:
                    V_trap[i][j]+= -1/4*(T00_E*(-elements['T0_azz']*params['azz'] + elements['T0_axxyy']*params['axxyy']))
                    V_trap[i][j]+= -1/4*(T20_E*(2*params['azz']*elements['T20_azz']+params['axxyy']*elements['T20_axxyy']))
                    V_trap[i][j]+= -1/4*(-1)*(T2p1_E*(2*params['azz']*elements['T2-1_azz']+params['axxyy']*elements['T2-1_axxyy'])+T2m1_E*(2*params['azz']*elements['T21_azz']+params['axxyy']*elements['T21_axxyy']))
                    V_trap[i][j]+= -1/4*(T22_E*(2*params['azz']*elements['T22_azz']+params['axxyy']*elements['T22_axxyy'])+T22_E*(2*params['azz']*elements['T2-2_azz']+params['axxyy']*elements['T2-2_axxyy']))
                if params.get('D') is not None:
                    N0[i,j] += elements['N^2']
                if params.get('q_lD_D') is not None:
                    lD0[i,j] += elements['l doubling q']
                if params.get('Gamma_D') is not None:
                    SR0[i,j] += elements['N.S']
                    if params.get('q_lD') is not None:
                        SR0[i,j] -= elements['NzSz']
                # H[i][j] = round(H[i][j],precision)
                #Iz[i][j] = params['c']*elements['Iz']
                #Sz[i][j] = elements['Sz']
        # Need to add centrifugal terms
        if params.get('D') is not None:
            H0 = matadd(H0,(-params['D']*N0@N0).tolist())
        if params.get('q_lD_D') is not None:
            H0 = matadd(H0,(-params['q_lD_D']/2*(lD0@N0+N0@lD0)).tolist())
        if params.get('Gamma_D') is not None:
            H0 = matadd(H0,(params['Gamma_D']/2*(SR0@N0+N0@SR0)).tolist())
        #H=matadd(H,matmult(Iz,Sz))
        # Create symbolic object
        H_symbolic = sy.Matrix(H0)+Ez*sy.Matrix(V_E)+Bz*sy.Matrix(V_B)
        if trap:
            H_symbolic+=sy.Matrix(V_trap)*I0
        H0_num = np.array(H0).astype(np.float64)
        V_E_num = np.array(V_E).astype(np.float64)
        V_B_num = np.array(V_B).astype(np.float64)
        # Use symbolic object to create function that given E and B values, returns a numpy array
        if trap and theta_num is None:
            V_trap_func = sy.lambdify((theta),V_trap, modules='numpy')
            H_func = lambda E,B,I,th: H0_num + V_E_num*E + V_B_num*B + I*np.array(V_trap_func(th))
            # H_func = sy.lambdify((Ez,Bz,I0,theta), H_symbolic, modules='numpy')
        elif trap and theta_num is not None:
            V_t_num = np.array(V_trap).astype(np.float64)
            H_func = lambda E,B,I,th: H0_num + V_E_num*E + V_B_num*B + V_t_num*I
        else:
            H_func = lambda E,B: H0_num + V_E_num*E + V_B_num*B
        # H_func = sy.lambdify((Ez,Bz), H_symbolic, modules='numpy')
        return H_func,H_symbolic
    # Same as above, but fully numeric
    # else:
    #     Ez,Bz = [E,B]
    #     size = len(q_numbers[q_str[0]])
    #     H = np.zeros((size,size))
    #     Iz = np.zeros((size,size))
    #     Sz = np.zeros((size,size))
    #     for i in range(size):
    #         for j in range(size):
    #             state_out = {q+'0':q_numbers[q][i] for q in q_str}
    #             state_in = {q+'1':q_numbers[q][j] for q in q_str}
    #             q_args = {**state_out,**state_in}
    #             elements = {term: element(**q_args) for term, element in matrix_elements.items()}
    #             H[i,j] = params['Be']*elements['N^2'] + params['Gamma_SR']*elements['N.S'] + \
    #                 params['b']*elements['I.S'] + \
    #                 bend*params['q_lD']/2*lD_bBJ(*state_out,*state_in)+\
    #                 params['g_S']*params['mu_B']*Bz*elements['ZeemanZ'] - params['muE_X']*Ez*elements['StarkZ']
    #             if params.get('q_lD') is not None:
    #                 H[i,j] += params['q_lD']/2*elements['l-doubling']
    #             Iz[i,j] = elements['Iz']
    #             Sz[i,j] = elements['Sz']
    #     H = H + params['c']*(Iz@Sz)
        return H

# See documentation for H_174X
def H_even_A(q_numbers,params,matrix_elements,symbolic=True,E=0,B=0,M_values='all',precision=5):
    q_str = list(q_numbers)
    if symbolic:
        Ez,Bz = sy.symbols('E_z B_z')
        size = len(q_numbers[q_str[0]])
        H0 = np.zeros((size,size)).tolist()
        V_B = np.zeros((size,size)).tolist()
        V_E = np.zeros((size,size)).tolist()
        if params.get('D') is not None:
            N0 = np.zeros((size,size))
        if params.get('p2q_D') is not None:
            p2q0 = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                state_out = {q+'0':q_numbers[q][i] for q in q_str}
                state_in = {q+'1':q_numbers[q][j] for q in q_str}
                q_args = {**state_out,**state_in}
                elements = {term: _safe_element_call(element, q_args) for term, element in matrix_elements.items()}
                H0[i][j] = params['Be']*elements['N^2'] + params['ASO']*elements['SO'] + \
                    params['bF']*elements['I.S'] + params['c']*np.sqrt(6)/3*elements['T2_0(IS)']+\
                    params['p+2q']*elements['Lambda Doubling p+2q'] - params['q']*elements['Lambda Doubling q']
                if M_values!='none':
                    V_B[i][j]+= params['g_L']*params['mu_B']*elements['ZeemanLZ']+params['g_S']*params['mu_B']*elements['ZeemanSZ'] +\
                    params['g_lp']*params['mu_B']*elements['ZeemanParityZ']
                    V_E[i][j]+= -params['muE']*elements['StarkZ']
                # H[i][j] = round(H[i][j],precision)
                if params.get('D') is not None:
                    N0[i,j] += elements['N^2']
                if params.get('p2q_D') is not None:
                    p2q0[i,j] += elements['Lambda Doubling p+2q']
        # Need to add centrifugal terms
        if params.get('D') is not None:
            H0 = matadd(H0,(-params['D']*N0@N0))
        if params.get('p2q_D') is not None:
            H0 = matadd(H0,(params['p2q_D']/2*(p2q0@N0+N0@p2q0)))
        H_symbolic = sy.Matrix(H0)+Ez*sy.Matrix(V_E)+Bz*sy.Matrix(V_B)
        H0_num = np.array(H0).astype(np.float64)
        V_E_num = np.array(V_E).astype(np.float64)
        V_B_num = np.array(V_B).astype(np.float64)
        H_func = lambda E,B: H0_num + V_E_num*E + V_B_num*B
        return H_func,H_symbolic
    # else:
    #     Ez,Bz = [E,B]
    #     size = len(q_numbers[q_str[0]])
    #     H = np.zeros((size,size))
    #     Iz = np.zeros((size,size))
    #     Sz = np.zeros((size,size))
    #     for i in range(size):
    #         for j in range(size):
    #             state_out = {q+'0':q_numbers[q][i] for q in q_str}
    #             state_in = {q+'1':q_numbers[q][j] for q in q_str}
    #             q_args = {**state_out,**state_in}
    #             elements = {term: element(**q_args) for term, element in matrix_elements.items()}
    #             H[i,j] = params['Be']*elements['N^2'] + SO*params['ASO']*elements['SO'] + \
    #                 (params['bF']-params['c']/3)*elements['I.S'] + params['c']*elements['IzSz']+\
    #                 params['g_L']*params['mu_B']*Bz*elements['ZeemanLZ']+params['g_S']*params['mu_B']*Bz*elements['ZeemanSZ'] +\
    #                 Bz*params['g_lp']*params['mu_B']*elements['ZeemanParityZ'] - params['muE_A']*Ez*elements['StarkZ']+\
    #                 params['p+2q']*elements['Lambda-Doubling']
    #    return H

    
# This is for 3Delta1 state
def H_even_3Delta1(q_numbers,params,matrix_elements,symbolic=True,E=0,B=0,M_values='all',precision=5):
    q_str = list(q_numbers)
    if symbolic:
        Ez,Bz = sy.symbols('E_z B_z')
        size = len(q_numbers[q_str[0]])
        H0 = np.zeros((size,size)).tolist()
        V_B = np.zeros((size,size)).tolist()
        V_E = np.zeros((size,size)).tolist()

        for i in range(size):
            for j in range(size):
                state_out = {q+'0':q_numbers[q][i] for q in q_str}
                state_in = {q+'1':q_numbers[q][j] for q in q_str}
                q_args = {**state_out,**state_in}
                elements = {term: _safe_element_call(element, q_args) for term, element in matrix_elements.items()}
                H0[i][j] = params['Be']*elements['J^2'] + \
                    params['A_parallel']*elements['A_parallel HF'] + params['Omega']*elements['Omega Doubling']
                if M_values!='none':
                    V_B[i][j]+= params['g_eff']*params['mu_B']*elements['Zeeman eff Z']
                    V_E[i][j]+= -params['muE']*elements['StarkZ']


        H_symbolic = sy.Matrix(H0)+Ez*sy.Matrix(V_E)+Bz*sy.Matrix(V_B)
        H0_num = np.array(H0).astype(np.float64)
        V_E_num = np.array(V_E).astype(np.float64)
        V_B_num = np.array(V_B).astype(np.float64)
        H_func = lambda E,B: H0_num + V_E_num*E + V_B_num*B
        return H_func,H_symbolic

        #return H    
    
    
    
# See documentation for H_174X
def H_odd_X(q_numbers,params,matrix_elements,symbolic=True,E=0,B=0,M_values='all',precision=5):
    q_str = list(q_numbers)
    Ez, Bz = sy.symbols('E_z B_z')
    size = len(q_numbers[q_str[0]])
    H0 = np.zeros((size, size)).tolist()
    V_B = np.zeros((size, size)).tolist()
    V_E = np.zeros((size, size)).tolist()

    for i in range(size):
        for j in range(size):
            state_out = {q + '0': q_numbers[q][i] for q in q_str}
            state_in = {q + '1': q_numbers[q][j] for q in q_str}
            q_args = {**state_out, **state_in}
            elements = {term: _safe_element_call(element, q_args) for term, element in matrix_elements.items()}

            H0[i][j] = (
                params['Be'] * elements['N^2']
                + params['Gamma_SR'] * elements['N.S']
                + params['bFYb'] * elements['IM.S']
                + np.sqrt(6) / 3 * params['cYb'] * elements['T2_0(IM,S)']
                + params['bFH'] * elements['IH.S']
                + np.sqrt(6) / 3 * params['cH'] * elements['T2_0(IH,S)']
            )

            if M_values != 'none':
                V_B[i][j] += params['g_S_eff'] * params['mu_B'] * elements['ZeemanZ']
                V_E[i][j] += -params['muE'] * elements['StarkZ']

            if params.get('q_lD') is not None:
                H0[i][j] += (
                    params['p_lD'] * elements['l doubling p']
                    - params['q_lD'] * elements['l doubling q']
                    - params['Gamma_SR'] * elements['NzSz']
                    + params['Gamma_Prime'] * elements['NzSz']
                )

            if params.get('e2Qq0', 0) != 0:
                H0[i][j] += (
                    np.sqrt(6) / (4 * 5 / 2 * (2 * 5 / 2 - 1)) * params['e2Qq0'] * elements['T2_0(IM^2)']
                )

    H_symbolic = sy.Matrix(H0) + Ez * sy.Matrix(V_E) + Bz * sy.Matrix(V_B)
    H0_num = np.array(H0).astype(np.float64)
    V_E_num = np.array(V_E).astype(np.float64)
    V_B_num = np.array(V_B).astype(np.float64)
    H_func = lambda E, B: H0_num + V_E_num * E + V_B_num * B
    return H_func, H_symbolic

def H_odd_A(q_numbers,params,matrix_elements,symbolic=True,E=0,B=0,M_values='all',precision=5):
    q_str = list(q_numbers)
    if symbolic:
        Ez,Bz = sy.symbols('E_z B_z')
        size = len(q_numbers[q_str[0]])
        H0 = np.zeros((size,size)).tolist()
        V_B = np.zeros((size,size)).tolist()
        V_E = np.zeros((size,size)).tolist()
        for i in range(size):
            for j in range(size):
                state_out = {q+'0':q_numbers[q][i] for q in q_str}
                state_in = {q+'1':q_numbers[q][j] for q in q_str}
                q_args = {**state_out,**state_in}
                elements = {term: _safe_element_call(element, q_args) for term, element in matrix_elements.items()}
                H0[i][j] = params['Be']*elements['N^2'] + params['ASO']*elements['SO']+\
                    params['h1/2Yb']*elements['IzLz_M'] - params['dYb']*elements['T2_2(IS)_M']+\
                    params['p+2q']*elements['Lambda-Doubling']
                if params['e2Qq0'] !=0:
                    H0[i][j]+= params['e2Qq0']*elements['T2_0(II)_M']
                # if M_values!='none':
                #     H[i][j]+=params['g_L']*params['mu_B']*Bz*elements['ZeemanLZ']+params['g_S']*params['mu_B']*Bz*elements['ZeemanSZ'] +\
                #     Bz*params['g_lp']*params['mu_B']*elements['ZeemanParityZ'] - params['muE_A']*Ez*elements['StarkZ']
                # H[i][j] = round(H[i][j],precision)


                    # params['bFH']*elements['I.S'] + params['cH']*np.sqrt(6)/3*elements['T2_0(IS)']
        H_symbolic = sy.Matrix(H0)
        H_func = sy.lambdify((Ez,Bz), H_symbolic, modules='numpy')
        return H_func,H_symbolic
    else:
        Ez,Bz = [E,B]
        size = len(q_numbers[q_str[0]])
        H0 = np.zeros((size,size)).tolist()
        V_B = np.zeros((size,size)).tolist()
        V_E = np.zeros((size,size)).tolist()
        Iz = np.zeros((size,size))
        Sz = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                state_out = {q+'0':q_numbers[q][i] for q in q_str}
                state_in = {q+'1':q_numbers[q][j] for q in q_str}
                q_args = {**state_out,**state_in}
                elements = {term: _safe_element_call(element, q_args) for term, element in matrix_elements.items()}
                H[i,j] = params['Be']*elements['N^2'] + SO*params['ASO']*elements['SO'] #+ \
                    # (params['bF']-params['c']/3)*elements['I.S'] + params['c']*elements['IzSz']+\
                    # params['g_L']*params['mu_B']*Bz*elements['ZeemanLZ']+params['g_S']*params['mu_B']*Bz*elements['ZeemanSZ'] +\
                    # Bz*params['g_lp']*params['mu_B']*elements['ZeemanParityZ'] - params['muE_A']*Ez*elements['StarkZ']+\
                    # params['p+2q']*elements['Lambda-Doubling']
        return H

def build_PTV_bBS(q_numbers,EDM_or_MQM,IM=5/2,iH=1/2):
    q_str = list(q_numbers)
    size = len(q_numbers[q_str[0]])
    H_PTV = np.zeros((size,size))
    EDM = {False: 0, True: 1}[EDM_or_MQM=='EDM']
    Mzz = {False: 0, True: 1}[EDM_or_MQM=='MQM']
    NSM = {False: 0, True: 1}[EDM_or_MQM=='NSM']
    NSDPV = {False: 0, True: 1}[EDM_or_MQM=='NSDPV']
    for i in range(size):
        for j in range(size):
            state_out = {q+'0':q_numbers[q][i] for q in q_str}
            state_in = {q+'1':q_numbers[q][j] for q in q_str}
            q_args = {**state_out,**state_in}
            if EDM_or_MQM == 'EDM':
                H_PTV[i,j] = EDM*EDM_bBS(**q_args,I=IM,iH=iH)
            elif EDM_or_MQM == 'MQM':
                H_PTV[i,j] = -Mzz/(2*IM*(2*IM-1))*(-np.sqrt(5/3)/2)*MQM_bBS(**q_args,I=IM,iH=iH)
                #H_PTV[i,j] = -Mzz/(2*5/2*(2*5/2-1))*(-2/3*np.sqrt(15))*MQM_bBS(**q_args,I=IM,iH=iH) #MQM_Tzzformat
            elif EDM_or_MQM == 'NSM':
                H_PTV[i,j] = NSM/IM*NSM_bBS(**q_args,I=IM,iH=iH)
            elif EDM_or_MQM == 'NSDPV':
                H_PTV[i,j] = -NSDPV/IM*NSDPV_bBS(**q_args,I=IM,iH=iH)                
            elif EDM_or_MQM == 'all':
                H_PTV[i,j] = -Mzz/(2*IM*(2*IM-1))*(-np.sqrt(5/3)/2)*MQM_bBS(**q_args,I=IM,iH=iH)
                
    #            H_PTV[i,j] = -Mzz/(2*5/2*(2*5/2-1))*(-np.sqrt(5/3)/2)*MQM_bBS(**q_args,I=IM,iH=iH) -EDM*EDM_bBS(**q_args,I=IM,iH=iH)
    #         elif H_MQM==2:
    #             H1[i,j] = -Mzz/(2*5/2*(2*5/2-1))*2/3*np.sqrt(6)*T2QM_bBS(**q_args)
    #             H2[i,j] = EDM_bBS(**q_args)
    #             H_PTV[i,j] = -EDM*EDM_bBS(**q_args)
    # if H_MQM==2:
    #     H_PTV = H_PTV + H1@H2
    
    return H_PTV

def build_PTV_bBJ(q_numbers):
    q_str= list(q_numbers)
    size = len(q_numbers[q_str[0]])
    H_PTV = np.zeros((size,size))
    EDM = 1
    for i in range(size):
        for j in range(size):
            state_out = {q+'0':q_numbers[q][i] for q in q_str}
            state_in = {q+'1':q_numbers[q][j] for q in q_str}
            q_args = {**state_out,**state_in}
            H_PTV[i,j] = -EDM*Sz_bBJ(**q_args)
    return H_PTV



def build_PTV_aBJ(q_numbers):
    q_str= list(q_numbers)
    size = len(q_numbers[q_str[0]])
    H_PTV = np.zeros((size,size))
    EDM = 1
    for i in range(size):
        for j in range(size):
            state_out = {q+'0':q_numbers[q][i] for q in q_str}
            state_in = {q+'1':q_numbers[q][j] for q in q_str}
            q_args = {**state_out,**state_in}
            H_PTV[i,j] = -EDM*EDM_even_aBJ_3Delta1(**q_args)
    return H_PTV

def build_p_TDM_aBJ(p, qmol, q_in, q_out,TDM_matrix_element):
    q_str_in = list(q_in)
    q_str_out = list(q_out)
    size_in = len(q_in[q_str_in[0]])
    size_out = len(q_out[q_str_out[0]])
    H_TDM = np.zeros((size_out,size_in))
    for i in range(size_out):
        for j in range(size_in):
            state_out = {q+'0':q_out[q][i] for q in q_out}
            state_in = {q+'1':q_in[q][j] for q in q_in}
            q_args = {**state_out,**state_in}
            H_TDM[i,j] = TDM_matrix_element(p,qmol,**q_args)
    return H_TDM

def build_TDM_aBJ(q_in,q_out,TDM_matrix_element):
    q_str_in = list(q_in)
    q_str_out = list(q_out)
    size_in = len(q_in[q_str_in[0]])
    size_out = len(q_out[q_str_out[0]])
    H_TDM = np.zeros((size_out,size_in))
    for i in range(size_out):
        for j in range(size_in):
            state_out = {q+'0':q_out[q][i] for q in q_out}
            state_in = {q+'1':q_in[q][j] for q in q_in}
            q_args = {**state_out,**state_in}
            H_TDM[i,j] = TDM_matrix_element(**q_args)
    return H_TDM

def build_TDM_aBJ_forbidden(p,q_in,q_out,TDM_matrix_element):
    q_str_in = list(q_in)
    q_str_out = list(q_out)
    size_in = len(q_in[q_str_in[0]])
    size_out = len(q_out[q_str_out[0]])
    H_TDM = np.zeros((size_out,size_in))
    for i in range(size_out):
        for j in range(size_in):
            state_out = {q+'0':q_out[q][i] for q in q_out}
            state_in = {q+'1':q_in[q][j] for q in q_in}
            q_args = {**state_out,**state_in}
            H_TDM[i,j] = TDM_matrix_element(p,**q_args)
    return H_TDM

def matmult(a,b):
    zip_b = list(zip(*b))
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zip_b] for row_a in a]

def scalarmult(s,mat):
    return [[s*ele for ele in inner] for inner in mat]

def matadd(a,b):
    return [[ele_a+ele_b for ele_a,ele_b in zip(row_a,row_b)] for row_a,row_b in zip(a,b)]

def convert_abBJ(input_qnumbers,output_qnumbers,S=1/2):
    input_keys = list(input_qnumbers)
    output_keys = list(output_qnumbers)
    input_size = len(input_qnumbers[input_keys[0]])
    output_size = len(output_qnumbers[output_keys[0]])
    basis_matrix = np.zeros((output_size,input_size))
    for i in range(output_size):
        for j in range(input_size):
            if 'N' in input_keys: #Convert case (b) to (a)
                a_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
                b_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            else:
                b_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
                a_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            basis_matrix[i,j] = b2a_matrix(a_qnumbers,b_qnumbers,S=S)
    return basis_matrix

def tensor_matrix(theta,q_in,q_out,params,elements,rank=2):
    T00_E = -1/np.sqrt(3)*params['2/e0c']
    T20_E = (3*np.cos(theta)**2 - 1)/np.sqrt(6)*params['2/e0c']
    T2p1_E = -(np.sin(2*theta)/2)*params['2/e0c']
    T2m1_E = -T2p1_E
    T22_E = (np.sin(theta)**2)/2*params['2/e0c']
    q_str_in = list(q_in)
    q_str_out = list(q_out)
    size_in = len(q_in[q_str_in[0]])
    size_out = len(q_out[q_str_out[0]])
    operator = np.zeros((size_out,size_in))
    for i in range(size_out):
        for j in range(size_in):
            state_out = {q+'0':q_out[q][i] for q in q_out}
            state_in = {q+'1':q_in[q][j] for q in q_in}
            q_args = {**state_out,**state_in}
            if rank==0:
                operator[i,j]= -1/4*(T00_E*(-elements['T0_azz'](**q_args)*params['azz'] + elements['T0_axxyy'](**q_args)*params['axxyy']))
            if rank==2:
                T20_me = -1/4*(T20_E*(2*params['azz']*elements['T20_azz'](**q_args)+params['axxyy']*elements['T20_axxyy'](**q_args)))
                T21_me = -1/4*(-1)*(T2p1_E*(2*params['azz']*elements['T2-1_azz'](**q_args)+params['axxyy']*elements['T2-1_axxyy'](**q_args))+T2m1_E*(2*params['azz']*elements['T21_azz'](**q_args)+params['axxyy']*elements['T21_axxyy'](**q_args)))
                T22_me = -1/4*(T22_E*(2*params['azz']*elements['T22_azz'](**q_args)+params['axxyy']*elements['T22_axxyy'](**q_args))+T22_E*(2*params['azz']*elements['T2-2_azz'](**q_args)+params['axxyy']*elements['T2-2_axxyy'](**q_args)))
                operator[i,j] = T20_me + T21_me + T22_me
    return operator


def build_operator(q_in,q_out,matrix_element,symbolic=False):
    q_str_in = list(q_in)
    q_str_out = list(q_out)
    size_in = len(q_in[q_str_in[0]])
    size_out = len(q_out[q_str_out[0]])
    operator = np.zeros((size_out,size_in))
    for i in range(size_out):
        for j in range(size_in):
            state_out = {q+'0':q_out[q][i] for q in q_out}
            state_in = {q+'1':q_in[q][j] for q in q_in}
            q_args = {**state_out,**state_in}
            operator[i,j] = matrix_element(**q_args)
            if symbolic:
                operator[i,j] = sy.nsimplify(operator[i,j])
    if symbolic:
        operator = sy.Matrix(operator)
    return operator

def convert_bbBS(input_qnumbers,output_qnumbers,S=1/2,I=5/2):
    input_keys = list(input_qnumbers)
    output_keys = list(output_qnumbers)
    input_size = len(input_qnumbers[input_keys[0]])
    output_size = len(output_qnumbers[output_keys[0]])
    basis_matrix = np.zeros((output_size,input_size))
    for i in range(output_size):
        for j in range(input_size):
            if 'G' in input_keys: #Convert case (bBS) to (bBJ)
                bBJ_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
                bBS_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            else:
                bBS_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
                bBJ_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            basis_matrix[i,j] = bBS_2_bBJ_matrix(bBS_qnumbers,bBJ_qnumbers,S=S,I=I)
    return basis_matrix

def decouple_b(input_qnumbers,output_qnumbers,S=1/2,I=1/2):
    input_keys = list(input_qnumbers)
    output_keys = list(output_qnumbers)
    input_size = len(input_qnumbers[input_keys[0]])
    output_size = len(output_qnumbers[output_keys[0]])
    basis_matrix = np.zeros((output_size,input_size))
    for i in range(output_size):
        for j in range(input_size):
            decoupled_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
            b_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            basis_matrix[i,j] = decouple_b_even(decoupled_qnumbers,b_qnumbers,S=S,I=I)
    return basis_matrix

def recouple_J(input_qnumbers,output_qnumbers,S=1/2,I=1/2):
    input_keys = list(input_qnumbers)
    output_keys = list(output_qnumbers)
    input_size = len(input_qnumbers[input_keys[0]])
    output_size = len(output_qnumbers[output_keys[0]])
    basis_matrix = np.zeros((output_size,input_size))
    for i in range(output_size):
        for j in range(input_size):
            recoupled_J_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
            decoupled_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            basis_matrix[i,j] = recouple_J_even(recoupled_J_qnumbers,decoupled_qnumbers,S=S,I=I)
    return basis_matrix

def decouple_b_I(input_qnumbers,output_qnumbers,S=1/2,I=1/2):
    input_keys = list(input_qnumbers)
    output_keys = list(output_qnumbers)
    input_size = len(input_qnumbers[input_keys[0]])
    output_size = len(output_qnumbers[output_keys[0]])
    basis_matrix = np.zeros((output_size,input_size))
    for i in range(output_size):
        for j in range(input_size):
            decoupled_I_qnumbers = {q:output_qnumbers[q][i] for q in output_keys}
            b_qnumbers = {q:input_qnumbers[q][j] for q in input_keys}
            basis_matrix[i,j] = decouple_b_I_even(decoupled_I_qnumbers,b_qnumbers,S=S,I=I)
    return basis_matrix
