from molecule_library_class import Molecule_Library
import numpy as np
import sympy as sy
import itertools as it
from numpy import linalg as npLA
from scipy import linalg as sciLA
from scipy import interpolate
import matplotlib.pyplot as plt
from copy import deepcopy
# import multiprocessing as mp
from functools import partial
from time import perf_counter

# Try importing torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from hamiltonian_builders import tensor_matrix
from sympy.physics.wigner import wigner_3j, wigner_6j

dipole_line_broadening_1mVcm_in_Hz = 1087
gfactor_line_broadening_1uG_in_Hz = 1.39962449

class MoleculeLevels(object):

    '''This class is used to determine energy levels for a given vibronic state
    in YbOH. As an input, the user must specify the isotope (string), the state
    (string), and the range of N values (tuple: (Nmin, Nmax)).

    Example isotope formatting: '174'
    Example state formatting: 'X000'

    All calculations are done in MHz, with E in V/cm, and B in Gauss
    '''

    @classmethod
    def initialize_state(cls,molecule,isotope,state,N_range,M_values='all',I=[0,1/2],S=1/2,P_values=[],M_range=[],round=6,trap=False,theta_num=None):
        if molecule=='YbOH':
            if isotope not in ['170','171','172','173','174','176']:
                print(isotope, 'is not a valid isotope of Yb')
                return None
            if isotope not in ['173','174','171']:
                print(isotope, 'is an isotope not yet supported by this code')
                return None
            if state not in ['X000','X010','A000']:
                print('Your input state, ', state, ' is not currently supported by this code. \nAn example state string: X000')
                return None
        elif molecule=='CaOH':
            if isotope not in ['40','42','43','44','46']:
                print(isotope, 'is not a stable isotope of Ca')
                return None
            if isotope not in ['40']:
                print(isotope, 'is an isotope not yet supported by this code')
                return None
            if state not in ['X010','X000','A000','B000']:
                print('Your input state, ', state, ' is not currently supported by this code. \nAn example state string: X000')
                return None
            
        elif molecule=='YbF':
            if isotope not in ['170','171','172','173','174','176']:
                print(isotope, 'is not a valid isotope of Yb')
                return None
            if isotope not in ['173','171','174']:
                print(isotope, 'is an isotope not yet supported by this code')
                return None
            if state not in ['X000','A000']:
                print('Your input state, ', state, ' is not currently supported by this code. \nAn example state string: X000')
                return None            
            if (state in ['A000']) and (isotope in ['174']):
                print('CAUTION! Currently, 174A000 YbF is used for 3Delta1 state!')
            
        iso_state = isotope + state
        if P_values == []:
            print('No P values provided, using P=1/2 as default')
            P_values=[1/2]

            # Properties contain information relevant to the isotope and state of interest
        properties = {
            'molecule': molecule,
            'iso_state': iso_state,
            'isotope': isotope,
            'state': state,
            'N_range': N_range,
            'M_values': M_values,
            'round': round,     #how much to round eigenvalues and eigenvectors
            'e_spin': S,    #electronic spin number
            'I_spins': I,    #spin of nuclei, [I_Yb, I_H]. I=0 means ignore
            'M_range': M_range,
            'P_values': P_values,
            'trap': trap,
            'theta_num':theta_num
        }
        return cls(**properties)


    def __init__(self, **properties):
        # Create attributes using properties dict
        self.__dict__.update(properties)

        # Initialize a library with relevant functions and parameters for all states
        self.library = Molecule_Library(self.molecule,self.I_spins,self.M_values,self.P_values,self.trap)
        self.parameters = self.library.parameters[self.iso_state] # Hamiltonian parameters relevant to state and isotope
        self.matrix_elements = self.library.matrix_elements[self.iso_state]
        self.hunds_case = self.library.cases[self.iso_state]
        self.K= self.library.K[self.iso_state]

        # Create quantum number dictionary, contains arrays of angular momenta eigenvalues indexed by basis vector
        # Example: {'J':[0,1,1,1], 'M':[0,-1,0,1]}
        self.q_numbers = self.library.q_number_builders[self.iso_state](self.N_range, I_list=self.I_spins,M_values = self.M_values,M_range=self.M_range)
        self.q_str = list(self.q_numbers)

        # Normalize quantum numbers to exact half-integers (avoid tiny float noise)
        for _k, _v in list(self.q_numbers.items()):
            arr = np.asarray(_v, dtype=float)
            arr = np.round(arr * 2.0) / 2.0
            self.q_numbers[_k] = arr

        # Create quantum numbers for alternate bases, like decoupled basis
        self.alt_q_numbers = {basis: q_builder(self.N_range,M_values = self.M_values,M_range=self.M_range)
            for basis,q_builder in self.library.alt_q_number_builders[self.iso_state].items()}

        # Also normalize alternate-basis quantum numbers to half-integers
        for basis, qdict in list(self.alt_q_numbers.items()):
            for _k, _v in list(qdict.items()):
                arr = np.asarray(_v, dtype=float)
                arr = np.round(arr * 2.0) / 2.0
                qdict[_k] = arr

        # Create Hamiltonian.
        # H_function is a function that takes (E,B) values as an argument, and returns a numpy matrix
        # H_symbolic is a symbolic sympy matrix of the Hamiltonian
        if self.trap:
            self.H_function, self.H_symbolic = self.library.H_builders[self.iso_state](self.q_numbers,M_values=self.M_values,precision=self.round,theta_num=self.theta_num)
        else:
            self.H_function, self.H_symbolic = self.library.H_builders[self.iso_state](self.q_numbers,M_values=self.M_values,precision=self.round)
        self.I_trap = 0
        if self.theta_num is None:
            self.theta_trap=0
        else:
            self.theta_trap = self.theta_num
        # Find free field eigenvalues and eigenvectors
        # Use numpy method during initialization for stability (torch can be used later)
        # Check matrix size first to avoid memory issues
        H_matrix = self.H_function(0, 1e-6 if (self.M_values == 'all' or self.M_values == 'pos') else 0)
        matrix_size = H_matrix.shape[0]
        
        if matrix_size > 10000:
            print(f"Warning: Large matrix size ({matrix_size}x{matrix_size}). This may take a while or cause memory issues.")
        
        try:
            if self.M_values == 'all' or self.M_values == 'pos':
                self.eigensystem(0,1e-6, method='numpy', order=False)
            else:
                self.eigensystem(0,0, method='numpy', order=False)
        except MemoryError:
            print(f"Error: Not enough memory for {matrix_size}x{matrix_size} matrix diagonalization.")
            print("Try reducing N_range or M_values, or use a machine with more memory.")
            raise
        except Exception as e:
            # If numpy fails, try scipy as fallback
            print(f"Warning: numpy diagonalization failed: {e}")
            print("Attempting with scipy...")
            try:
                if self.M_values == 'all' or self.M_values == 'pos':
                    self.eigensystem(0,1e-6, method='scipy', order=False)
                else:
                    self.eigensystem(0,0, method='scipy', order=False)
            except Exception as e2:
                # Last resort: try torch if available
                print(f"Warning: scipy diagonalization failed: {e2}")
                print("Attempting with torch (if available)...")
                try:
                    if self.M_values == 'all' or self.M_values == 'pos':
                        self.eigensystem(0,1e-6, method='torch', order=False)
                    else:
                        self.eigensystem(0,0, method='torch', order=False)
                except Exception as e3:
                    print(f"Error: All diagonalization methods failed. Last error: {e3}")
                    raise RuntimeError(f"Could not diagonalize {matrix_size}x{matrix_size} matrix with any method.") from e3
        self.size = len(self.evecs0)
        self.Parity_mat = self.library.all_parity[self.iso_state](self.q_numbers,self.q_numbers)
        self.generate_parities(self.evecs0)

        # These attrbiutes will be used to store Zeeman and Stark information
        # Each index corresponds to a value of E or B.
        # Evals contains an array of eigenvalues for each field value
        # Evecs contains an array of eigenvectors for each field value
        self.Ez = None
        self._Bz = None
        self.evals_E = None
        self.evecs_E = None

        self.Bz = None
        self._Ez = None
        self.evals_B = None
        self.evecs_B = None

        # Attributes used for evaluating PT violating shifts
        self.H_PTV = None
        self.PTV_E = None
        self.PTV_B = None
        self.PTV0 = None
        self.PTV_type = None


        self.state_str =  r'$^{{{iso}}}${mol} $\tilde{{{state}}}({vib})$'.format(iso=self.isotope,mol=self.molecule,state = self.state[:1],vib=self.state[1:])


    def eigensystem(self,Ez_val,Bz_val,method='torch',order=True, set_attr=True, Normalize=False,disable_trap=False,angle=None, qnumber_show = False):
        if angle is not None:
            self.theta_trap = angle
        if self.trap and not disable_trap:
            evals,evecs = diagonalize(self.H_function(Ez_val,Bz_val,self.I_trap,self.theta_trap),method=method,order=order, Normalize=Normalize,round=self.round)
        elif self.trap and disable_trap:
            evals,evecs = diagonalize(self.H_function(Ez_val,Bz_val,self.I_trap,self.theta_trap),method=method,order=order, Normalize=Normalize,round=self.round)
        else:
            evals,evecs = diagonalize(self.H_function(Ez_val,Bz_val),method=method,order=order, Normalize=Normalize,round=self.round)
        if set_attr:
            self.evals0,self.evecs0 = [evals,evecs]
            self.E0,self.B0 = [Ez_val, Bz_val]
        
        '''
        state_dsignation = [None]
        
        i = 0
        
        if qnumber_show:
            for evec in evecs:
                state_dsignation.append({'energy': evals[i], 'N': self.q_numbers['N'][np.argmax(evec**2)] , 'G':self.q_numbers['G'][np.argmax(evec**2)], 'F1':self.q_numbers['F1'][np.argmax(evec**2)]  })
                i += 1
        
#        return evals,evecs, state_dsignation
        '''
        
        return evals,evecs

    
    def AngleMap(self,angle_array, Ez_val, Bz_val, I_trap = None,output=False,write_attribute=True,method='torch',initial_evecs=None,**kwargs):
        if self.trap == False:
            return none
        if I_trap is not None:
            self.I_trap = I_trap
        self._Bz = Bz_val
        self._Ez = Ez_val
        # t0 = perf_counter()
        angle_matrices = np.array([self.H_function(Ez_val,Bz_val,self.I_trap,angle_val) for angle_val in angle_array])
        #results = [self.eigensystem(Ez_val,Bz_val,set_attr=False) for Bz_val in Bz_array]
        # pool = mp.Pool(mp.cpu_count())
        # worker = partial(diagonalize,order=True, Normalize=False,round=self.round)
        # results = pool.starmap(worker,[(self.H_function(Ez_val,Bz_val),) for Bz_val in Bz_array.tolist()])
        # pool.close()
        # pool.join()
        # t1 = perf_counter()
        #evals_B, evecs_B = list(zip(*results))
        # evals_B = np.array(evals_B)
        # evecs_B = np.array(evecs_B)
        evals_t,evecs_t = diagonalize_batch(angle_matrices,method=method,round=self.round)
        # t2 = perf_counter()
        if initial_evecs is None:
            evecs_old = evecs_t[0]
        else:
            evecs_old = initial_evecs
        for i in range(len(angle_array)):
            if i == 0 and initial_evecs is None:
                continue
            evals_new,evecs_new = evals_t[i], evecs_t[i]
            order = state_ordering(evecs_old,evecs_new,round=self.round)
            evecs_ordered = evecs_new[order,:]
            evals_ordered = evals_new[order]
            #fix phase?
            sgn = np.sign(np.diag(evecs_ordered@evecs_old.T))
            evecs_ordered = (evecs_ordered.T*sgn).T
            evecs_t[i] = evecs_ordered
            evals_t[i] = evals_ordered
            evecs_old = evecs_t[i]
        # t3 = perf_counter()
        # print('1:',(t1-t0))
        # print('2:',(t2-t1))
        # print('3:',(t3-t2))
        # evals_B,evecs_B = [np.array(evals_B),np.array(evecs_B)]
        if write_attribute:
            self.evals_t = evals_t
            self.evecs_t = evecs_t
        if output:
            return evals_t,evecs_t
        else:
            return


    # Bz must be in Gauss
    def ZeemanMap(self,Bz_array,Ez_val=0,plot=False,output=False,write_attribute=True,method='torch',initial_evecs=None,order=True,**kwargs):
        self.Bz = Bz_array
        self._Ez = Ez_val
        # t0 = perf_counter()
        if self.trap:
            B_matrices = np.array([self.H_function(Ez_val,Bz_val,self.I_trap,self.theta_trap) for Bz_val in Bz_array])
        else:
            B_matrices = np.array([self.H_function(Ez_val,Bz_val) for Bz_val in Bz_array])
        #results = [self.eigensystem(Ez_val,Bz_val,set_attr=False) for Bz_val in Bz_array]
        # pool = mp.Pool(mp.cpu_count())
        # worker = partial(diagonalize,order=True, Normalize=False,round=self.round)
        # results = pool.starmap(worker,[(self.H_function(Ez_val,Bz_val),) for Bz_val in Bz_array.tolist()])
        # pool.close()
        # pool.join()
        # t1 = perf_counter()
        #evals_B, evecs_B = list(zip(*results))
        # evals_B = np.array(evals_B)
        # evecs_B = np.array(evecs_B)
        evals_B,evecs_B = diagonalize_batch(B_matrices,method=method,round=self.round)
        # t2 = perf_counter()
        if order:
            if initial_evecs is None:
                evecs_old = evecs_B[0]
            else:
                evecs_old = initial_evecs
            for i in range(len(Bz_array)):
                if i == 0 and initial_evecs is None:
                    continue
                evals_new,evecs_new = evals_B[i], evecs_B[i]
                order = state_ordering(evecs_old,evecs_new,round=self.round)
                evecs_ordered = evecs_new[order,:]
                evals_ordered = evals_new[order]
                #fix phase?
                sgn = np.sign(np.diag(evecs_ordered@evecs_old.T))
                evecs_ordered = (evecs_ordered.T*sgn).T
                evecs_B[i] = evecs_ordered
                evals_B[i] = evals_ordered
                evecs_old = evecs_B[i]
        # t3 = perf_counter()
        # print('1:',(t1-t0))
        # print('2:',(t2-t1))
        # print('3:',(t3-t2))
        # evals_B,evecs_B = [np.array(evals_B),np.array(evecs_B)]
        if write_attribute:
            self.evals_B = evals_B
            self.evecs_B = evecs_B
        if plot:
            self.plot_evals_EB('B',**kwargs)
        if output:
            return evals_B,evecs_B
        else:
            return

    # Ez must be in V/cm
    def StarkMap(self,Ez_array,Bz_val=1e-9,plot=False,output=False,write_attribute=True,method='torch',initial_evecs=None,order=True,**kwargs):
        self.Ez = Ez_array
        self._Bz = Bz_val
        if self.trap:
            E_matrices = np.array([self.H_function(Ez_val,Bz_val,self.I_trap,self.theta_trap) for Ez_val in Ez_array])
        else:
            E_matrices = np.array([self.H_function(Ez_val,Bz_val) for Ez_val in Ez_array])
        # results = [self.eigensystem(Ez_val,Bz_val,set_attr=False) for Ez_val in Ez_array.tolist()]
        # pool = mp.Pool(mp.cpu_count())
        # worker = partial(diagonalize,order=True, Normalize=False,round=self.round)
        # results = pool.starmap(worker,[(self.H_function(Ez_val,Bz_val),) for Ez_val in Ez_array.tolist()])
        # pool.close()
        # pool.join()
        # evals_E, evecs_E = list(zip(*results))
        # evals_E = np.array(evals_E)
        # evecs_E = np.array(evecs_E)
        evals_E,evecs_E = diagonalize_batch(E_matrices,method=method,round=self.round)
        if initial_evecs is None:
            evecs_old = evecs_E[0]
        else:
            evecs_old = initial_evecs
        if order:
            for i in range(len(Ez_array)):
                if i == 0 and initial_evecs is None:
                    continue
                evals_new,evecs_new = evals_E[i], evecs_E[i]
                order = state_ordering(evecs_old,evecs_new,round=self.round)
                evecs_ordered = evecs_new[order,:]
                evals_ordered = evals_new[order]
                #fix phase
                sgn = np.sign(np.diag(evecs_ordered@evecs_old.T))
                evecs_ordered = (evecs_ordered.T*sgn).T
                evecs_E[i] = evecs_ordered
                evals_E[i] = evals_ordered
                evecs_old = evecs_E[i]
        evals_E,evecs_E = [np.array(evals_E),np.array(evecs_E)]
        if write_attribute:
            self.evals_E = evals_E
            self.evecs_E = evecs_E
        if plot:
            self.plot_evals_EB('E',**kwargs)
        if output:
            return evals_E,evecs_E
        else:
            return
        
        
    def g_eff_evecs(self,evals,evecs,Ez,Bz,step=1e-7):
        if np.sign(Bz) != 0:
            step_size = np.sign(Bz)*step
        if Bz == 0:
            step_size = step
        Bz_array = np.linspace(Bz, Bz+step_size*2,3)
        evals_B,evecs_B = self.ZeemanMap(Bz_array,Ez_val=Ez,plot=False,output=True,write_attribute=False,initial_evecs=None,order=False)
       
        g_eff = []
        
        for i in range(len(evals)):
            g_eff.append(np.gradient(evals_B.T[i])[1]/(step*self.parameters['mu_B']))

        g_eff = np.array(g_eff)
        
        return g_eff
        
        
    def g_eff_EB(self,Ez,Bz,step=1e-7):
        if Ez == self.E0 and Bz == self.B0:
            return self.g_eff_evecs(self.evals0,self.evecs0,Ez,Bz,step=step)
        else:
            return self.g_eff_evecs(*self.eigensystem(Ez,Bz),Ez,Bz,step=step)
        
        
        
    def dipole_evecs(self,evals,evecs,Ez,Bz,step=1e-7):
        if np.sign(Ez) != 0:
            step_size = np.sign(Ez)*step
        if Ez == 0:
            step_size = step        
        Ez_array = np.linspace(Ez-step_size, Ez+step_size,3)
        evals_E,evecs_E = self.StarkMap(Ez_array,Bz_val=Bz,plot=False,output=True,write_attribute=False,initial_evecs=None,order=False)
       
        dipole = []
        
        for i in range(len(evals)):
            dipole.append(np.gradient(evals_E.T[i])[1]/(step*self.parameters['muE']))

        dipole = np.array(dipole)
        
        return dipole
        
        
    def dipole_EB(self,Ez,Bz,step=1e-7):
        if Ez == self.E0 and Bz == self.B0:
            return self.dipole_evecs(self.evals0,self.evecs0,Ez,Bz,step=step)
        else:
            return self.dipole_evecs(*self.eigensystem(Ez,Bz),Ez,Bz,step=step)
        
        
    def show_biggest_mixture(self,Ez, Bz, state_index_array, round = 5):
        
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
        for i in range(len(state_index_array)): 
            evec0 = evecs[state_index_array[i]]
            Parity = evec0@self.Parity_mat@evec0
            indices = np.argsort(evec0**2)
            N0 = self.q_numbers['N'][np.argmax(evec0**2)]
            M0 = self.q_numbers['M'][np.argmax(evec0**2)]
            F0 = self.q_numbers['F'][np.argmax(evec0**2)]
            N0_second_biggest = self.q_numbers['N'][indices[-2]]
            M0_second_biggest = self.q_numbers['M'][indices[-2]]
            F0_second_biggest = self.q_numbers['F'][indices[-2]]            
            if '174' in self.iso_state:
                J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                J0_second_biggest = self.q_numbers['J'][indices[-2]]
            if '174' not in self.iso_state:
                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                F10_second_biggest = self.q_numbers['F1'][indices[-2]]
                G0_second_biggest = self.q_numbers['G'][indices[-2]]

            print('state index: ', state_index_array[i])
            print('energy: ', np.round(evals[state_index_array[i]], round))
            print('N: ', N0 )
            if '174' in self.iso_state:
                print('J: ', J0)
                print('J_second_biggest: ', J0_second_biggest)                
            if '174' not in self.iso_state:
                print('G: ', G0 )
                print('F1: ', F10)
                print('G_second_biggest: ', G0_second_biggest )
                print('F1_second_biggest: ', F10_second_biggest)                
            print('F: ', F0 )
            print('M_F: ', M0 )
            
            print('F_second_biggest: ', F0_second_biggest )
            print('M_F_second_biggest: ', M0_second_biggest )
            
            print('mix ratio of the biggest mixture',evec0[np.argmax(evec0**2)])
            print('mix ratio of the biggest mixture',evec0[indices[-1]])       
            print('mix ratio of the second biggest mixture',evec0[indices[-2]])  
            Parity_biggest = self.Parity_mat[indices[-1], indices[-1]]
            Parity_second_biggest = self.Parity_mat[indices[-2], indices[-2]]
            #print('parity of biggest mixture', Parity_biggest)
            #print('parity of second biggest mixture', Parity_second_biggest)
            print('Parity: ', Parity)
            print(' ')
            
        return
    
    
    def calculate_two_photon_spectrum(self,Ez, Bz, state_index_array,laser_polarization = 'both', parity_sign = -1, round = 5):
        
        #print('this function only calculates transitions connecting same parity states')
        
        #print('specify laser_polarization from both, orth, para ')
        
        if laser_polarization in ['both']:
            laser_polarization_M = [0,1,2]
        if laser_polarization in ['orth','orthogonal','perpendicular']:
            laser_polarization_M = [1]            
        if laser_polarization in ['para','parallel']:
            laser_polarization_M = [0,2]             
        
        freqs = []
        state_info = []

        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

        # Work on a numpy array of unique indices
        state_index_array = np.unique(np.asarray(state_index_array, dtype=int))
        if state_index_array.size < 2:
            return [], []

        # Select eigenvectors/energies for the requested indices
        evecs_sub = evecs[state_index_array]
        energies_sub = np.asarray(evals)[state_index_array]

        # Parities for each state (vectorized)
        Parities = np.einsum('ni,ij,nj->n', evecs_sub, self.Parity_mat, evecs_sub)

        # Dominant quantum numbers per eigenvector (vectorized argmax)
        maxidx = np.argmax(evecs_sub**2, axis=1)
        N_array = np.asarray(self.q_numbers['N'])[maxidx]
        M_array = np.asarray(self.q_numbers['M'])[maxidx]
        F_array = np.asarray(self.q_numbers['F'])[maxidx]
        if '174' in self.iso_state:
            J_array = np.asarray(self.q_numbers.get('J', np.zeros_like(F_array)))[maxidx]
        else:
            F1_array = np.asarray(self.q_numbers.get('F1', np.zeros_like(F_array)))[maxidx]
            G_array = np.asarray(self.q_numbers.get('G', np.zeros_like(F_array)))[maxidx]

        # Build pairwise masks using broadcasting, keep only i<j (upper triangle)
        idx_i, idx_j = np.triu_indices(state_index_array.size, k=1)

        Parity_mask = (np.sign(Parities[idx_i] * Parities[idx_j]) == 1) & (np.sign(Parities[idx_i]) == parity_sign)
        deltaM = np.abs(M_array[idx_i] - M_array[idx_j])
        pol_mask = np.isin(deltaM, laser_polarization_M)
        deltaF = np.abs(F_array[idx_i] - F_array[idx_j]) < 2

        valid = Parity_mask & pol_mask & deltaF

        if not np.any(valid):
            return [], []

        valid_i = idx_i[valid]
        valid_j = idx_j[valid]

        # Build outputs
        for ii, jj in zip(valid_i, valid_j):
            state_index_0 = int(state_index_array[ii])
            state_index_1 = int(state_index_array[jj])

            energy0 = float(energies_sub[ii])
            energy1 = float(energies_sub[jj])
            freqs.append(abs(energy0 - energy1))

            N0 = int(N_array[ii])
            N1 = int(N_array[jj])
            M0 = int(M_array[ii])
            M1 = int(M_array[jj])
            F0 = int(F_array[ii])
            F1 = int(F_array[jj])
            Parity0 = float(Parities[ii])
            Parity1 = float(Parities[jj])

            if '174' in self.iso_state:
                J0 = int(J_array[ii])
                J1 = int(J_array[jj])
                state_info.append({"freq": abs(energy0-energy1), "state index 0": state_index_0,"energy 0": energy0,"N0": N0, "J0": J0, "F0": F0, "M0": M0, "Parity0": Parity0,  "state index 1": state_index_1,"energy 1": energy1,"N1": N1, "J1": J1, "F1": F1, "M1": M1, "Parity1": Parity1})
            else:
                F10 = int(F1_array[ii])
                G0 = int(G_array[ii])
                F11 = int(F1_array[jj])
                G1 = int(G_array[jj])
                state_info.append({"freq": abs(energy0-energy1), "state index 0": state_index_0,"energy 0": energy0,"N0": N0, "G0": G0, "F10": F10, "F0": F0, "M0": M0, "Parity0": Parity0, "state index 1": state_index_1,"energy 1": energy1,"N1": N1, "G1": G1,  "F11": F11, "F1": F1, "M1": M1, "Parity1": Parity1})

        return freqs, state_info
    

    def calculate_174_two_photon_Rabi_freq(self,Ez, Bz, state_index_array, J_e, laser_polarization_array, N_e = 1, spin_rotation_coupling_consider = True, round = 5, print_result = True):
        
        I_h = 1/2 # hydrogen hyperfine
        S = 1/2  # electron spin
        J_m = J_e
        N_m = N_e
        
        if print_result:
        
            if '174' not in self.iso_state:
                print('This is only for 174 YbOH')

            print('J_e and N_e is the J and N quantum number of the excited state')
            print('laser_polarization_array specifies the polarization of the laser, e.g, For X polarization between state_index_array[0] and excited state, and Z polarization for state_index_array[1] and excited state, you specify laser_polarization_array as [1,0]')
            print('state_index_array should contain two states that are connected with two-photon laser (e.g, [1, 10])')
        
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
            
        if print_result:    
            print('first two-photon state index: ', state_index_array[0])
            print('first two-photon state energy: ', np.round(evals[state_index_array[0]], round))
            print('second two-photon state index: ', state_index_array[1])
            print('second two-photon state energy: ', np.round(evals[state_index_array[1]], round))
            print('two-photon freq (corrected)', abs(np.round(evals[state_index_array[1]], round) - np.round(evals[state_index_array[0]], round))-1)
            print('Note that the two-photon freq above takes into account of the fact that the prediction in my code is 1 MHz higher than observed line position in June 2024, so the above freq is already subtracted from the prediction by 1 MHz')
            print('')
            print('AOM0 freq for this corrected two-photon freq (w/ AOM1 freq of 70 MHz, double path)', (abs(np.round(evals[state_index_array[1]], round) - np.round(evals[state_index_array[0]], round))-1)/2 + 70)
            print('')
        
        evec0 = evecs[state_index_array[0]]
        evec1 = evecs[state_index_array[1]]
        
        dict = {(0,0):0, (1,1):0, (1,0):0, (0,1):0}
        final_amplitudes = []
        
        for j in range(len(evec0)):

            N_i = self.q_numbers['N'][j]
            m_F_i = self.q_numbers['M'][j]
            F_i = self.q_numbers['F'][j]          
            if '174' in self.iso_state:
                J_i = self.q_numbers['J'][j]
            
            weight_i = evec0[j]
           
            if weight_i != 0:
                for jj in range(len(evec1)):

                    N_f = self.q_numbers['N'][jj]
                    m_F_f = self.q_numbers['M'][jj]
                    F_f = self.q_numbers['F'][jj]          
                    if '174' in self.iso_state:
                        J_f = self.q_numbers['J'][jj]   

                    weight_f = evec1[jj]
                    if weight_f != 0:
                        
                        
                        possibilities = []
                        amplitudes = []
                        second_possibilities = []
                        second_amplitudes = []
                        transitions = []
                        transition_amplitudes = []
                        transition_polarizations = []

                        for m_p in range(-1, 2, 1): # photon polarizations
                            for sign in range(-1, 1+1, 2): # try both possibilities for unresolved excited state hydrogen hyperfine
                                F_m = J_m + sign*I_h
                                amplitude = ((-1)**(F_i-m_F_i)) * wigner_3j(F_i, 1, F_m, m_F_i, m_p, -(m_F_i+m_p)) # second transition: mediating to final state
                                amplitude *= ((-1)**(F_m+J_i+I_h+1)) * np.sqrt((2*F_m+1)*(2*F_i+1)) * wigner_6j(J_m, F_m, I_h, F_i, J_i, 1) # recoupling
                                if spin_rotation_coupling_consider:
                                    amplitude *= ((-1)**(J_m+N_i+S+1)) * np.sqrt((2*J_m+1)*(2*J_i+1)) * wigner_6j(N_m, J_m, S, J_i, N_i, 1) # recoupling
                                amplitude *= weight_i


                                # first transition: initial to mediating state

                                if amplitude != 0: # ignore forbidden transitions
                                    possibilities += [(m_p, F_m, m_F_i+m_p)]
                                    amplitudes += [amplitude]

                        for i, possibility in enumerate(possibilities): # explore second photon possibilities given allowed first photon transition
                            for m_p in range(-1, 2, 1): # photon polarizations
                                F_m = possibility[1] 
                                m_F_m = possibility[2]
                                amplitude = ((-1)**(F_m-m_F_m)) * wigner_3j(F_m, 1, F_f, m_F_m, m_p, -(m_F_m+m_p)) # second transition: mediating to final state
                                amplitude *= ((-1)**(F_f+J_m+I_h+1)) * np.sqrt((2*F_m+1)*(2*F_f+1)) * wigner_6j(J_f, F_f, I_h, F_m, J_m, 1) # recoupling
                                if spin_rotation_coupling_consider:
                                    amplitude *= ((-1)**(J_f+N_m+S+1)) * np.sqrt((2*J_m+1)*(2*J_f+1)) * wigner_6j(N_f, J_f, S, J_m, N_m, 1) # recoupling
                                amplitude *= weight_f

                                if amplitude != 0:# ignore forbidden transitions
                                    

                                    if m_F_m+m_p == m_F_f:
                                        second_possibilities += [(m_p, F_f, m_F_f)]
                                        second_amplitudes += [amplitude]
                                        transitions += [f'{possibility[0]} to F_m={F_m}, m_F_m={m_F_m} then {m_p}']
                                        transition_amplitudes += [amplitudes[i] * amplitude]
                                        transition_polarizations += [(possibility[0], m_p)]


                        transition_amplitudes = np.array(transition_amplitudes)
                        for polarization in [(0,0), (1,1), (1,0), (0,1)]: # summing over polarizations
                            for i, transition_polarization in enumerate(transition_polarizations):
                                if np.abs(transition_polarization[0]) == polarization[0] and np.abs(transition_polarization[1]) == polarization[1]:
                                    dict[polarization] += float(transition_amplitudes[i])
                                    
        if print_result:
            print('')
            print('final two-photon Rabi freq', str(dict[(laser_polarization_array[0],laser_polarization_array[1])]) )
            print('')
            print('both z polarization: '+str(dict[(0,0)]))
            print('both x polarization: '+str(dict[(1,1)]))
            print('x, then z polarization: '+str(dict[(1,0)]))
            print('z, then x polarization: '+str(dict[(0,1)]))
            print('')
        
        return   dict[(laser_polarization_array[0],laser_polarization_array[1])]
    
    
    '''
    def Calculate_two_photon_spectrum(self, Ez, Bz, state_index_array, J_e, laser_polarization_array, state_index_to_consider_array, N_e = 1, spin_rotation_coupling_consider = True,  print_result = False):
        
        print('state_index_to_consider_array is the array ')
    '''
        
        
    
    def Calculate_TDM(self, p, q_array, Ez, Bz, state_index_array, round = 5):
        
        print('This function will calculate the E1 TDM between a pair of two states at specified E and B fields')
        
        if (len(state_index_array) > 0) and (len(state_index_array[0]) != 2):
            print('The state_index_array should be something like [ [1,101], [2,102], [3,103], ...] ')
        
        if (self.iso_state != '173X010' or self.molecule !='YbOH' ):
            print('This function is only compatible with 173YbOH X010')
        
        self.TDM_p_mat = self.library.TDM_p_builders[self.iso_state](p, q_array, self.q_numbers,self.q_numbers)
        
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
        for i in range(len(state_index_array)): 
            evec0 = evecs[state_index_array[i][0]]
            evec1 = evecs[state_index_array[i][1]]
            TDM = evec1@self.TDM_p_mat@evec0
            
            Parity0 = evec0@self.Parity_mat@evec0
            N0 = self.q_numbers['N'][np.argmax(evec0**2)]
            M0 = self.q_numbers['M'][np.argmax(evec0**2)]
            F0 = self.q_numbers['F'][np.argmax(evec0**2)]
            if '174' in self.iso_state:
                J0 = self.q_numbers['J'][np.argmax(evec0**2)]
            if '174' not in self.iso_state:
                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]

            Parity1 = evec1@self.Parity_mat@evec1
            N1 = self.q_numbers['N'][np.argmax(evec1**2)]
            M1 = self.q_numbers['M'][np.argmax(evec1**2)]
            F1 = self.q_numbers['F'][np.argmax(evec1**2)]
            if '174' in self.iso_state:
                J1 = self.q_numbers['J'][np.argmax(evec1**2)]
            if '174' not in self.iso_state:
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]
                
            print('state index: ', state_index_array[i])
            print('energy: ', np.round(evals[state_index_array[i][0]], round), np.round(evals[state_index_array[i][1]], round))
            print('N0 N1: ', N0, N1 )
            if '174' in self.iso_state:
                print('J0 J1: ', J0, J1)
            if '174' not in self.iso_state:
                print('G0 G1: ', G0, G1)
                print('F10 F11: ', F10, F11)
            print('F0 F1: ', F0, F1)
            print('M_F0 M_F1: ', M0, M1 )
            print('Parity0 Parity1: ', Parity0, Parity1)
            print('TDM: ', TDM)
            print(' ')
        
        
        return    
        
        

    def return_freq(self, Ez, Bz, state_index_array, print_suru = True, round = 10, energy_print = False):
        
        if print_suru:
            print('This function will just return transition frequency between a pair of two states at specified E and B fields')
        
            print('The state_index_array should be something like [1,101] ')
        
      
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

        if energy_print:
            return np.round(evals[state_index_array[0]], round), np.round(evals[state_index_array[1]], round)
        else:
            return np.round(abs(evals[state_index_array[0]] - evals[state_index_array[1]]), round)
  
    
    
    
    
    def Calculate_coherence_time_at_pol_lim(self, Ez_at_magic, Bz_at_magic, Ez_at_pol, Bz_at_pol, state_index_array_at_magic, round = 5, idx = None, step_B = 1e-4, step_E = 1e-2):
        
        if (len(state_index_array_at_magic) > 0) and (len(state_index_array_at_magic[0]) != 2):
            print('The state_index_array_at_magic should be something like [ [1,101], [2,102], [3,103], ...] ')
        

        evals_at_magic,evecs_at_magic = self.eigensystem(Ez_at_magic,Bz_at_magic, set_attr=True)
            
        evals_at_pol,evecs_at_pol = self.eigensystem(Ez_at_pol,Bz_at_pol, set_attr=True)

        if idx is not None:
            evals_at_magic = evals_at_magic[idx]
            evecs_at_magic = evecs_at_magic[idx]             
            evals_at_pol = evals_at_pol[idx]
            evecs_at_pol = evecs_at_pol[idx]     
            
        g_effs = []
        Orientations = []

        if idx is None:
            for i in range(len(evals_at_pol)):
                Orientations.append(self.dipole_EB(Ez_at_pol,Bz_at_pol,step=step_E)[i])
                g_effs.append(self.g_eff_EB(Ez_at_pol,Bz_at_pol,step=step_B)[i])
        else:
            for i in idx:
                Orientations.append(self.dipole_EB(Ez_at_pol,Bz_at_pol,step=step_E)[i])
                g_effs.append(self.g_eff_EB(Ez_at_pol,Bz_at_pol,step=step_B)[i])          

        Orientations = np.array(Orientations)
        g_effs = np.array(g_effs)            
            
        for i in range(len(state_index_array_at_magic)): 
            evec0_at_magic = evecs_at_magic[state_index_array_at_magic[i][0]]
            evec1_at_magic = evecs_at_magic[state_index_array_at_magic[i][1]]
            
            N0_at_magic = self.q_numbers['N'][np.argmax(evec0_at_magic**2)]
            M0_at_magic = self.q_numbers['M'][np.argmax(evec0_at_magic**2)]
            F0_at_magic = self.q_numbers['F'][np.argmax(evec0_at_magic**2)]
            if '174' in self.iso_state:
                J0_at_magic = self.q_numbers['J'][np.argmax(evec0_at_magic**2)]
            if '174' not in self.iso_state:
                F10_at_magic = self.q_numbers['F1'][np.argmax(evec0_at_magic**2)]
                G0_at_magic = self.q_numbers['G'][np.argmax(evec0_at_magic**2)]

            N1_at_magic = self.q_numbers['N'][np.argmax(evec1_at_magic**2)]
            M1_at_magic = self.q_numbers['M'][np.argmax(evec1_at_magic**2)]
            F1_at_magic = self.q_numbers['F'][np.argmax(evec1_at_magic**2)]
            if '174' in self.iso_state:
                J1_at_magic = self.q_numbers['J'][np.argmax(evec1_at_magic**2)]
            if '174' not in self.iso_state:
                F11_at_magic = self.q_numbers['F1'][np.argmax(evec1_at_magic**2)]
                G1_at_magic = self.q_numbers['G'][np.argmax(evec1_at_magic**2)]


            overlaps = []
            for e in range(len(evecs_at_pol)):
                if self.q_numbers['M'][np.argmax(evecs_at_pol[e]**2)] == M0_at_magic:
                    overlaps.append(abs(evec0_at_magic@evecs_at_pol[e].T))
                else:
                    overlaps.append(0)

            overlaps = np.array(overlaps)
            state_track_index0 = np.argmax(overlaps)

            #print('new state index for 0: ', state_track_index0)

            overlaps = []
            for e in range(len(evecs_at_pol)):
                if self.q_numbers['M'][np.argmax(evecs_at_pol[e]**2)] == M1_at_magic:
                    overlaps.append(abs(evec1_at_magic@evecs_at_pol[e].T))
                else:
                    overlaps.append(0)

            overlaps = np.array(overlaps)
            state_track_index1 = np.argmax(overlaps)

            #print('new state index for 1: ', state_track_index1)


            evec0_at_pol = evecs_at_pol[state_track_index0]
            evec1_at_pol = evecs_at_pol[state_track_index1]                            

            N0_at_pol = self.q_numbers['N'][np.argmax(evec0_at_pol**2)]
            M0_at_pol = self.q_numbers['M'][np.argmax(evec0_at_pol**2)]
            F0_at_pol = self.q_numbers['F'][np.argmax(evec0_at_pol**2)]
            if '174' in self.iso_state:
                J0_at_pol = self.q_numbers['J'][np.argmax(evec0_at_pol**2)]
            if '174' not in self.iso_state:
                F10_at_pol = self.q_numbers['F1'][np.argmax(evec0_at_pol**2)]
                G0_at_pol = self.q_numbers['G'][np.argmax(evec0_at_pol**2)]

            N1_at_pol = self.q_numbers['N'][np.argmax(evec1_at_pol**2)]
            M1_at_pol = self.q_numbers['M'][np.argmax(evec1_at_pol**2)]
            F1_at_pol = self.q_numbers['F'][np.argmax(evec1_at_pol**2)]
            if '174' in self.iso_state:
                J1_at_pol = self.q_numbers['J'][np.argmax(evec1_at_pol**2)]
            if '174' not in self.iso_state:
                F11_at_pol = self.q_numbers['F1'][np.argmax(evec1_at_pol**2)]
                G1_at_pol = self.q_numbers['G'][np.argmax(evec1_at_pol**2)]                            

            delta_d = Orientations[state_track_index0] - Orientations[state_track_index1]
            delta_g = g_effs[state_track_index0] - g_effs[state_track_index1]

            coherence_time_E = 1/(abs(delta_d*dipole_line_broadening_1mVcm_in_Hz) + abs(0*1e-3*dipole_line_broadening_1mVcm_in_Hz))
            coherence_time_B = 1/(abs(delta_g*gfactor_line_broadening_1uG_in_Hz) + abs(0*1e-3*gfactor_line_broadening_1uG_in_Hz))
            coherence_time_total = 1/((1/coherence_time_E) + (1/coherence_time_B))
                
            print('E field (V/cm) at magic: ', Ez_at_magic)
            print('state index at magic: ', state_index_array_at_magic[i])
            print('energy at magic: ', np.round(evals_at_magic[state_index_array_at_magic[i][0]], round), np.round(evals_at_magic[state_index_array_at_magic[i][1]], round))
            print('N0 N1 at magic: ', N0_at_magic, N1_at_magic )
            if '174' in self.iso_state:
                print('J0 J1 at magic: ', J0_at_magic, J1_at_magic)
            if '174' not in self.iso_state:
                print('G0 G1 at magic: ', G0_at_magic, G1_at_magic)
                print('F10 F11 at magic: ', F10_at_magic, F11_at_magic)
            print('F0 F1 at magic: ', F0_at_magic, F1_at_magic)
            print('M_F0 M_F1 at magic: ', M0_at_magic, M1_at_magic )
            print(' ')

            
            print('E field (V/cm) at polarized limit: ', Ez_at_pol)
            print('state index at pol: ', [state_track_index0, state_track_index1])
            print('energy at pol: ', np.round(evals_at_pol[state_track_index0], round), np.round(evals_at_pol[state_track_index1], round))
            print('N0 N1 at pol: ', N0_at_pol, N1_at_pol )
            if '174' in self.iso_state:
                print('J0 J1 at pol: ', J0_at_pol, J1_at_pol)
            if '174' not in self.iso_state:
                print('G0 G1 at pol: ', G0_at_magic, G1_at_pol)
                print('F10 F11 at pol: ', F10_at_pol, F11_at_pol)
            print('F0 F1 at pol: ', F0_at_pol, F1_at_pol)
            print('M_F0 M_F1 at pol: ', M0_at_pol, M1_at_pol )
            print(' ')
            
            print('delta d: ', delta_d)
            print('delta g: ', delta_g)
            print('Coherence time E: ', coherence_time_E)
            print('Coherence time B: ', coherence_time_B)
            print('Coherence time total: ', coherence_time_total)
            print(' ')
            print(' ')
            
        return     
    
    
    
    
    
    def find_magic(self,Efield_array,Bz, EDM_or_MQM, B_criteria = 10, E_criteria = 10, CPV_criteria = -1, M_criteria = [0,1,2], step_B = 1e-4, step_E = 1e-2, idx = None, round = None, neighbor_state_rejection = False):
        
        print('M_criteria needs to be array; e.g, [0,1,2]')
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        BE_magic_index_all = []
        PTV_shifts_magic_all = []    
        
        found_magic_index = []
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            BE_magic_index = []
            PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            for i in range(len(B_combis)):
                if (abs(B_combis[i][0] - B_combis[i][1]) < B_criteria) and (abs(E_combis[i][0] - E_combis[i][1]) < E_criteria):                    

                    evec0 = evecs[index_combis[i][0]]
                    evec1 = evecs[index_combis[i][1]]

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1
                    
                    M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                    M1 = self.q_numbers['M'][np.argmax(evec1**2)]

                    F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                    F1 = self.q_numbers['F'][np.argmax(evec1**2)] 
                    
                    if '174' in self.iso_state:
                        J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                        J1 = self.q_numbers['J'][np.argmax(evec1**2)]
                    
                    if '174' not in self.iso_state:
                        F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                        F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                        G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                        G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                        
                        
                    flag = True 
                    if neighbor_state_rejection:
                        flag = False
                        if abs(index_combis[i][0] - index_combis[i][1]) != 1:
                            flag = True                        
                        
                        
                    if (flag is True) and (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) in M_criteria):
                        
                        
                        if i not in found_magic_index:
                            found_magic_index.append(i)
                        
                        BE_magic_index.append([index_combis[i],Ez])
                        

                    
                        if round is not None:
                            E_PTV_0 = np.round( E_PTV_0, round)
                            E_PTV_1 = np.round( E_PTV_1, round)

                        PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                        print('state index: ',index_combis[i], i)
                        print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                        if '174' in self.iso_state:
                            print('J: ', J0, J1 )
                        if '174' not in self.iso_state:
                            print('G: ', G0, G1 )
                            print('F1: ', F10, F11 )
                        print('F: ', F0, F1 )
                        print('M_F: ', M0, M1 )
                        print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ',np.round( B_combis[i][0] - B_combis[i][1], round))
                        print('polarization: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                        print('PTV shifts: ', E_PTV_0, E_PTV_1, ' difference: ', np.round(E_PTV_0 - E_PTV_1,round))  
                        print(' ')


            BE_magic_index = np.array(BE_magic_index)
            PTV_shifts_magic = np.array(PTV_shifts_magic)
            
            BE_magic_index_all.append(BE_magic_index)
            PTV_shifts_magic_all.append(PTV_shifts_magic)
            
            
        BE_magic_index_all = np.array(BE_magic_index_all)
        PTV_shifts_magic_all = np.array(PTV_shifts_magic_all)
        
        found_magic_index = np.array(found_magic_index)
        
        print('# of identified magic transitions',len(found_magic_index))
        
        return found_magic_index, BE_magic_index_all, PTV_shifts_magic_all


    
    def find_long_coherence_time(self,Efield_array,Bz, EDM_or_MQM, coherence_time_criteria = 1e-3, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, neighbor_state_rejection = False, interpolation_number = 200, show_max_B_and_E_coherence_time = True):
        
        print('coherence time is calcurated assuming 1 mV/cm E field and 1 uG B field fluctuations')
            
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  '171' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
            
        elif  '173' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
                
                
        found_magic_index = []
        BE_magic_index = []

        Is_this_first_Efield = True

        previous_Ez = 0
        
        
        for Ez in Efield_array:            

            print('E field (V/cm): ', Ez)
            print(' ')

            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]

            g_effs = []
            Orientations = []

            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)





            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            if Is_this_first_Efield:
                previous_ds = np.zeros(len(B_combis))
                previous_gs = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                
                
                if 'A000' not in self.state:
                    N0 = self.q_numbers['N'][np.argmax(evec0**2)]
                    N1 = self.q_numbers['N'][np.argmax(evec1**2)]
                
                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                    
                if '174' in self.iso_state:
                    
                    J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                    J1 = self.q_numbers['J'][np.argmax(evec1**2)]                    

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]   
                
                dnow = E_combis[i][0] - E_combis[i][1]
                gnow = B_combis[i][0] - B_combis[i][1]
                

                if Is_this_first_Efield is False:   
                    
                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1



                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       



                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]) and (previous_gs[i]*previous_ds[i]*dnow*gnow != 0):
                        
                        flag = True
                        if neighbor_state_rejection:
                            flag = False
                            if abs(index_combis[i][0] - index_combis[i][1]) != 1:
                                flag = True
                        
                        if flag:
                            
                            fg = interpolate.interp1d([previous_Ez, Ez], [previous_gs[i], gnow])
                            fd = interpolate.interp1d([previous_Ez, Ez], [previous_ds[i], dnow])

                            g_arrays = fg(np.linspace(previous_Ez, Ez, interpolation_number))
                            d_arrays = fd(np.linspace(previous_Ez, Ez, interpolation_number))
                            
                            coherence_time_E = 1/(abs(d_arrays*dipole_line_broadening_1mVcm_in_Hz) + abs(np.gradient(d_arrays)*1e-3*dipole_line_broadening_1mVcm_in_Hz))
                            coherence_time_B = 1/(abs(g_arrays*gfactor_line_broadening_1uG_in_Hz) + abs(np.gradient(g_arrays)*1e-3*gfactor_line_broadening_1uG_in_Hz))
                            coherence_time_total = 1/((1/coherence_time_E) + (1/coherence_time_B))

                            max_coherence_time = max(coherence_time_total)
                            max_E_coherence_time = max(coherence_time_E)
                            max_B_coherence_time = max(coherence_time_B)

                            
                            if max_coherence_time > coherence_time_criteria:
                                
                                max_coherence_time_index = np.argmax(coherence_time_total)
                                max_E_coherence_time_index = np.argmax(coherence_time_E)
                                max_B_coherence_time_index = np.argmax(coherence_time_B)
                                
                                
                                max_coherence_time_at_max_E_coherence_time = coherence_time_total[max_E_coherence_time_index]
                                max_coherence_time_at_max_B_coherence_time = coherence_time_total[max_B_coherence_time_index]

                                E_at_max_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_coherence_time_index]
                                E_at_max_E_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_E_coherence_time_index]
                                E_at_max_B_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_B_coherence_time_index]
                                
                                g_at_max_coherence_time = fg(E_at_max_coherence_time)
                                d_at_max_coherence_time = fd(E_at_max_coherence_time)

                                g_at_max_E_coherence_time = fg(E_at_max_E_coherence_time)
                                d_at_max_E_coherence_time = fd(E_at_max_E_coherence_time)
                                g_at_max_B_coherence_time = fg(E_at_max_B_coherence_time)
                                d_at_max_B_coherence_time = fd(E_at_max_B_coherence_time)                                
                                
                                dgdE_at_max_coherence_time = interpolation_number/(Ez - previous_Ez)*np.gradient(g_arrays)[max_coherence_time_index]
                                
                                d_zero_crossed = False
                                g_zero_crossed = False
                                if np.sign(dnow) != np.sign(previous_ds[i]):
                                    d_zero_crossed = True
                                if np.sign(gnow) != np.sign(previous_gs[i]):
                                    g_zero_crossed = True

                                
                                if i not in found_magic_index:
                                    found_magic_index.append(i)

                                BE_magic_index.append([index_combis[i],Ez])                            

                                print('state index: ',index_combis[i], i)
                                print('energy (MHz): ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                                print('Transition frequency (MHz): ', evals[index_combis[i][0]] - evals[index_combis[i][1]])                                
                                if 'A000' not in self.state:
                                    print('N: ', N0, N1 )
                                if '174' in self.iso_state:
                                    print('J: ', J0, J1 )
                                if '174' not in self.iso_state:
                                    print('G: ', G0, G1 )
                                    print('F1: ', F10, F11 )
                                print('F: ', F0, F1 )
                                print('M_F: ', M0, M1 )
                                print('max coherence time between',previous_Ez, 'and', Ez, 'V/cm:', max_coherence_time,'sec')
                                print('max coherence time at', E_at_max_coherence_time, 'V/cm')
                                
                                if g_zero_crossed:
                                    print('g zero crossing!')
                                if d_zero_crossed:
                                    print('d zero crossing!')
                                    print('dg/dE at max coherence time:', dgdE_at_max_coherence_time)
                                    print('(dg/dE)/(CPV) at max coherence time:', dgdE_at_max_coherence_time/(E_PTV_0 - E_PTV_1),'(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                    #print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                    
                                print('g at max coherence time:', g_at_max_coherence_time)
                                print('d at max coherence time:', d_at_max_coherence_time)
                                
                                if show_max_B_and_E_coherence_time:
                                    print('max E coherence time at', E_at_max_E_coherence_time, 'V/cm')
                                    print('max coherence time at max E coherence time point', max_coherence_time_at_max_E_coherence_time, 'sec')
                                    print('g at max E coherence time:', g_at_max_E_coherence_time)
                                    print('d at max E coherence time:', d_at_max_E_coherence_time)
                                    print('max B coherence time at', E_at_max_B_coherence_time, 'V/cm')
                                    print('max coherence time at max B coherence time point', max_coherence_time_at_max_B_coherence_time, 'sec')
                                    print('g at max B coherence time:', g_at_max_B_coherence_time)
                                    print('d at max B coherence time:', d_at_max_B_coherence_time) 
                                    
                                print('At',Ez,'V/cm:')
                                print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ',np.round( B_combis[i][0] - B_combis[i][1], round))
                                print('polarization: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                                print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                print('PTV shifts: ', E_PTV_0, E_PTV_1, ' difference: ', np.round(E_PTV_0 - E_PTV_1,round))
                                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                    print('NSM shifts: ', E_PTV_NSM_0, E_PTV_NSM_1, ' difference: ', np.round(E_PTV_NSM_0 - E_PTV_NSM_1,round))
                                    print('MQM/EDM shifts: ', np.round( (E_PTV_0 - E_PTV_1)/(E_PTV_EDM_0 - E_PTV_EDM_1),round))

                                print(' ')
                                print(' ')



                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                
                if '174' not in self.iso_state:
                    previous_F1s[i][0] = F10
                    previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous_ds[i] = E_combis[i][0] - E_combis[i][1]
                previous_gs[i] = B_combis[i][0] - B_combis[i][1]
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False
                
            previous_Ez = Ez

        found_magic_index = np.array(found_magic_index)
        
        print('# of identified magic transitions',len(found_magic_index))                                            
             
        return  BE_magic_index
    
        
    
    
    
    def find_magic_uptodate(self,Efield_array,Bz, EDM_or_MQM, g_criteria = 10,  d_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, neighbor_state_rejection = False, interpolation_number = 200, show_max_B_and_E_coherence_time = True):
        
        print('coherence time is calcurated assuming 1 mV/cm E field and 1 uG B field fluctuations')
            
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  '171' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
            
        elif  '173' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
                
                
        found_magic_index = []
        BE_magic_index = []

        Is_this_first_Efield = True

        previous_Ez = 0
        
        
        for Ez in Efield_array:            

            print('E field (V/cm): ', Ez)
            print(' ')

            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]

            g_effs = []
            Orientations = []

            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)





            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            if Is_this_first_Efield:
                previous_ds = np.zeros(len(B_combis))
                previous_gs = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]

                N0 = self.q_numbers['N'][np.argmax(evec0**2)]
                N1 = self.q_numbers['N'][np.argmax(evec1**2)]

                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                    
                if '174' in self.iso_state:
                    
                    J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                    J1 = self.q_numbers['J'][np.argmax(evec1**2)]                    

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]   
                
                dnow = E_combis[i][0] - E_combis[i][1]
                gnow = B_combis[i][0] - B_combis[i][1]
                

                if Is_this_first_Efield is False:   
                    
                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1



                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       



                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]) and (previous_gs[i]*previous_ds[i]*dnow*gnow != 0):
                        
                        flag = True
                        if neighbor_state_rejection:
                            flag = False
                            if abs(index_combis[i][0] - index_combis[i][1]) != 1:
                                flag = True
                        
                        if flag:
                            
                            fg = interpolate.interp1d([previous_Ez, Ez], [previous_gs[i], gnow])
                            fd = interpolate.interp1d([previous_Ez, Ez], [previous_ds[i], dnow])

                            g_arrays = fg(np.linspace(previous_Ez, Ez, interpolation_number))
                            d_arrays = fd(np.linspace(previous_Ez, Ez, interpolation_number))
                            
                            minimum_g = min(abs(g_arrays))
                            minimum_d = min(abs(d_arrays))
                            
                            coherence_time_E = 1/(abs(d_arrays*dipole_line_broadening_1mVcm_in_Hz) + abs(np.gradient(d_arrays)*1e-3*dipole_line_broadening_1mVcm_in_Hz))
                            coherence_time_B = 1/(abs(g_arrays*gfactor_line_broadening_1uG_in_Hz) + abs(np.gradient(g_arrays)*1e-3*gfactor_line_broadening_1uG_in_Hz))
                            coherence_time_total = 1/((1/coherence_time_E) + (1/coherence_time_B))

                            max_coherence_time = max(coherence_time_total)
                            max_E_coherence_time = max(coherence_time_E)
                            max_B_coherence_time = max(coherence_time_B)

                            
                            if minimum_g <= g_criteria and minimum_d <= d_criteria:
                                
                                max_coherence_time_index = np.argmax(coherence_time_total)
                                max_E_coherence_time_index = np.argmax(coherence_time_E)
                                max_B_coherence_time_index = np.argmax(coherence_time_B)
                                

                                E_at_max_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_coherence_time_index]
                                E_at_max_E_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_E_coherence_time_index]
                                E_at_max_B_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_B_coherence_time_index]
                                
                                g_at_max_coherence_time = fg(E_at_max_coherence_time)
                                d_at_max_coherence_time = fd(E_at_max_coherence_time)

                                g_at_max_E_coherence_time = fg(E_at_max_E_coherence_time)
                                d_at_max_E_coherence_time = fd(E_at_max_E_coherence_time)
                                g_at_max_B_coherence_time = fg(E_at_max_B_coherence_time)
                                d_at_max_B_coherence_time = fd(E_at_max_B_coherence_time)                                
                                
                                dgdE_at_max_coherence_time = interpolation_number/(Ez - previous_Ez)*np.gradient(g_arrays)[max_coherence_time_index]
                                
                                d_zero_crossed = False
                                g_zero_crossed = False
                                if np.sign(dnow) != np.sign(previous_ds[i]):
                                    d_zero_crossed = True
                                if np.sign(gnow) != np.sign(previous_gs[i]):
                                    g_zero_crossed = True

                                
                                if i not in found_magic_index:
                                    found_magic_index.append(i)

                                BE_magic_index.append([index_combis[i],Ez])                            

                                print('state index: ',index_combis[i], i)
                                print('energy (MHz): ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                                print('Transition frequency (MHz): ', evals[index_combis[i][0]] - evals[index_combis[i][1]])                                
                                print('N: ', N0, N1 )
                                if '174' in self.iso_state:
                                    print('J: ', J0, J1 )
                                if '174' not in self.iso_state:
                                    print('G: ', G0, G1 )
                                    print('F1: ', F10, F11 )
                                print('F: ', F0, F1 )
                                print('M_F: ', M0, M1 )
                                print('minimum g', minimum_g)
                                print('minimum d', minimum_d)
                                print('max coherence time between',previous_Ez, 'and', Ez, 'V/cm:', max_coherence_time,'sec')
                                print('max coherence time at', E_at_max_coherence_time, 'V/cm')
                                
                                if g_zero_crossed:
                                    print('g zero crossing!')
                                if d_zero_crossed:
                                    print('d zero crossing!')
                                    print('dg/dE at max coherence time:', dgdE_at_max_coherence_time)
                                    print('(dg/dE)/(CPV) at max coherence time:', dgdE_at_max_coherence_time/(E_PTV_0 - E_PTV_1),'(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                    #print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                    
                                print('g at max coherence time:', g_at_max_coherence_time)
                                print('d at max coherence time:', d_at_max_coherence_time)
                                
                                if show_max_B_and_E_coherence_time:
                                    print('max E coherence time at', E_at_max_E_coherence_time, 'V/cm')
                                    print('max E coherence time', max_E_coherence_time, 'sec')
                                    print('g at max E coherence time:', g_at_max_E_coherence_time)
                                    print('d at max E coherence time:', d_at_max_E_coherence_time)
                                    print('max B coherence time at', E_at_max_B_coherence_time, 'V/cm')
                                    print('max B coherence time', max_B_coherence_time, 'sec')
                                    print('g at max B coherence time:', g_at_max_B_coherence_time)
                                    print('d at max B coherence time:', d_at_max_B_coherence_time) 
                                    
                                print('At',Ez,'V/cm:')
                                print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ',np.round( B_combis[i][0] - B_combis[i][1], round))
                                print('polarization: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                                print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                print('PTV shifts: ', E_PTV_0, E_PTV_1, ' difference: ', np.round(E_PTV_0 - E_PTV_1,round))
                                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                    print('NSM shifts: ', E_PTV_NSM_0, E_PTV_NSM_1, ' difference: ', np.round(E_PTV_NSM_0 - E_PTV_NSM_1,round))
                                    print('MQM/EDM shifts: ', np.round( (E_PTV_0 - E_PTV_1)/(E_PTV_EDM_0 - E_PTV_EDM_1),round))

                                print(' ')
                                print(' ')



                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                
                if '174' not in self.iso_state:
                    previous_F1s[i][0] = F10
                    previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous_ds[i] = E_combis[i][0] - E_combis[i][1]
                previous_gs[i] = B_combis[i][0] - B_combis[i][1]
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False
                
            previous_Ez = Ez

        found_magic_index = np.array(found_magic_index)
        
        print('# of identified magic transitions',len(found_magic_index))                                            
             
        return  BE_magic_index
    
    
    
    
    
    
    def find_EB_zero_slope(self,Efield_array,Bz, EDM_or_MQM,  B_slope_criteria = 10, E_slope_criteria = 10,  B_value_criteria = 10, E_value_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, F1_same_as_previous_check = False, F_same_as_previous_check = False):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        #BE_magic_index_all = []
        #PTV_shifts_magic_all = []
        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        F1_is_same = True
        F_is_same = True
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            
            if Is_this_first_Efield is False:
                print('previous E field (V/cm): ', previous_Efield)
                
            print(' ')
            
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            
            if Is_this_first_Efield:
                previous_B = np.zeros(len(B_combis))
                previous_E = np.zeros(len(E_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):
                
                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 
                
                if (Is_this_first_Efield is False) and (abs((B_combis[i][0] - B_combis[i][1] - previous_B[i])/(Ez-previous_Efield)) < B_slope_criteria) and (abs((E_combis[i][0] - E_combis[i][1] - previous_E[i])/(Ez-previous_Efield)) < E_slope_criteria) and (abs(B_combis[i][0] - B_combis[i][1]) < B_value_criteria) and (abs(E_combis[i][0] - E_combis[i][1]) < E_value_criteria):                    

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1


                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]):
                                                
                        if (F10 != previous_F1s[i][0]) or (F11 != previous_F1s[i][1]):
                            print('Caution! F1 value is not the same as the previous F1 value! F1 now: ', F10, F11)
                            print('F1 previous: ', previous_F1s[i][0], previous_F1s[i][1])
                            print('')
                            
                            if F1_same_as_previous_check:
                                F1_is_same = False
                            
                        if (F0 != previous_Fs[i][0]) or (F1 != previous_Fs[i][1]):
                            print('Caution! F value is not the same as the previous F value! F now: ', F0, F1)
                            print('F previous: ', previous_Fs[i][0], previous_Fs[i][1])
                            print('')

                            if F_same_as_previous_check:
                                F_is_same = False
                                
                        if (F1_is_same is False) or (F_is_same is False):
                            F1_is_same = True
                            F_is_same = True
                        
                        else:
                            B_zero_crossing.append([Ez, index_combis[i], np.round(previous_B[i], round), np.round( B_combis[i][0] - B_combis[i][1], round), np.round(previous_E[i], round), np.round( E_combis[i][0] - E_combis[i][1], round)])



                            if round is not None:
                                E_PTV_0 = np.round( E_PTV_0, round)
                                E_PTV_1 = np.round( E_PTV_1, round)


                            #PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                            print('state index: ',index_combis[i], i)
                            print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                            print('G: ', G0, G1 )
                            print('F1: ', F10, F11 )
                            print('F: ', F0, F1 )
                            print('M_F: ', M0, M1 )
                            print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                            print('g factor difference previous: ', np.round( previous_B[i], round))
                            print('g factor slope: ',  np.round( abs((B_combis[i][0] - B_combis[i][1] - previous_B[i])/(Ez-previous_Efield)), round))
                            print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                            print('dipole difference previous: ', np.round( previous_E[i], round))
                            print('dipole slope: ',  np.round( abs((E_combis[i][0] - E_combis[i][1] - previous_E[i])/(Ez-previous_Efield)), round))
                            print('PTV shifts (a factor of two): ', 2*E_PTV_0, 2*E_PTV_1, ' difference: ', np.round(2*(E_PTV_0 - E_PTV_1),round))  
                            print(' ')
                        
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                previous_F1s[i][0] = F10
                previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous_B[i] = B_combis[i][0] - B_combis[i][1]
                previous_E[i] = E_combis[i][0] - E_combis[i][1]
            previous_Efield = Ez
               
           
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            #BE_magic_index = np.array(BE_magic_index)
            B_zero_crossing = np.array(B_zero_crossing)
            
            #BE_magic_index_all.append(BE_magic_index)
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all
            
    def plot_EB_zero_crossing_E_scan(self,Efield_array,Bz, EDM_or_MQM, state_plot_index_array, figsize_x = 15, figsize_y = 7.5, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, B_plot = True, E_plot = True, CPV_plot = True, E_energy_plot = True, title_defalut = True, ylabel_defalut = True, legend_defalut = True,  fontsize=20, state_track = False, save_fig = None, CPV_hlines = None, offset = 'middle', combined_plot = False, legend_combined = ['d', 'g left', 'g right'], save_fig_combined = None, width_plot_E = None, width_plot_B = None, coherence_time_plot = False, legend_coherence_time = ['E', 'B', 'total'], save_fig_coherence_time = None, gradient_plot = False, interpolation = None, coherence_time_y_lim = None):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            
        elif  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            
        else:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)


        previous_Ms = np.zeros((len(state_plot_index_array),2))
        previous_F1s = np.zeros((len(state_plot_index_array),2))
        previous_Fs = np.zeros((len(state_plot_index_array),2))  
        
        
        B_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))
        E_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))
        CPV_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))
        if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
            EDM_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))
        if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
            EDM_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))                
            NSM_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))  
        energy_plot_array = np.zeros((len(state_plot_index_array), len(Efield_array)))

        
        j = 0
        
        for Ez in Efield_array:            

            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            #B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            jj = 0
            for i in state_plot_index_array:
                
                plot_index = i

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                if CPV_plot:
                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1

                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1                    

                    
                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                
                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]
       

              

                if j > 0:
                    if (M0 != previous_Ms[jj][0]) or (M1 != previous_Ms[jj][1]):
                        print('Caution! M value is not the same as the previous M value! E field (V/cm) & cobination & state index: ', Ez, i, index_combis[i][0], index_combis[i][1])
                        print('M now: ', M0, M1)
                        print('M previous: ', previous_Ms[jj][0], previous_Ms[jj][1])
                        if '174' not in self.iso_state:
                            print('F1 previous: ', previous_F1s[jj][0], previous_F1s[jj][1])
                        print('F previous: ', previous_Fs[jj][0], previous_Fs[jj][1])
                        print('')
                        
                        if state_track:
                            
                            print('state track is on. We will fix the state index.')
                            
                            overlaps = []
                            for e in range(len(evecs)):
                                if self.q_numbers['M'][np.argmax(evecs[e]**2)] == previous_Ms[jj][0]:
                                    overlaps.append(abs(evecs_old[index_combis[i][0]]@evecs[e].T))
                                else:
                                    overlaps.append(0)
                                
                            overlaps = np.array(overlaps)
                            state_track_index0 = np.argmax(overlaps)
                            
                            print('new state index for 0', state_track_index0)
                            
                        
                            overlaps = []
                            for e in range(len(evecs)):
                                if self.q_numbers['M'][np.argmax(evecs[e]**2)] == previous_Ms[jj][1]:
                                    overlaps.append(abs(evecs_old[index_combis[i][1]]@evecs[e].T))
                                else:
                                    overlaps.append(0)
                                        
                            overlaps = np.array(overlaps)
                            state_track_index1 = np.argmax(overlaps)  
                            
                            print('new state index for 1', state_track_index1)
                            
                            M0 = self.q_numbers['M'][np.argmax(evecs[state_track_index0]**2)]
                            M1 = self.q_numbers['M'][np.argmax(evecs[state_track_index1]**2)]
                            
                            print('M new', M0, M1)

                            if '174' not in self.iso_state:

                                G0 = self.q_numbers['G'][np.argmax(evecs[state_track_index0]**2)]
                                G1 = self.q_numbers['G'][np.argmax(evecs[state_track_index1]**2)]

                                F10 = self.q_numbers['F1'][np.argmax(evecs[state_track_index0]**2)]
                                F11 = self.q_numbers['F1'][np.argmax(evecs[state_track_index1]**2)]
                                
                                print('G new', G0, G1)
                                print('F1 new', F10, F11)

                            F0 = self.q_numbers['F'][np.argmax(evecs[state_track_index0]**2)]
                            F1 = self.q_numbers['F'][np.argmax(evecs[state_track_index1]**2)]
                            
                            print('F new', F0, F1)
                           
                            
                            
                            
                            for ee in range(len(index_combis)):
                                if (index_combis[ee][0] == state_track_index0) and (index_combis[ee][1] == state_track_index1):
                                    state_plot_index_array[jj] = ee
                                    plot_index = ee
                                    print('new state index for combination array', ee, state_plot_index_array[jj])
                                    print('')       
                                    
                            if CPV_plot:
                                
                                evec0 = evecs[state_track_index0]
                                evec1 = evecs[state_track_index1]  
                                
                                E_PTV_0 = evec0@H_PTV@evec0
                                E_PTV_1 = evec1@H_PTV@evec1

                                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                                    E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                                    E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                                    E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                                    E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       
                        
                        
                    if '174' not in self.iso_state:
                        if (F10 != previous_F1s[jj][0]) or (F11 != previous_F1s[jj][1]):
                            print('Caution! F1 value is not the same as the previous F1 value! E field (V/cm) & state index: ', Ez, i)
                            print('F1 now: ', F10, F11)
                            print('F1 previous: ', previous_F1s[jj][0], previous_F1s[jj][1])
                            print('')

                    if (F0 != previous_Fs[jj][0]) or (F1 != previous_Fs[jj][1]):
                        print('Caution! F value is not the same as the previous F value! E field (V/cm) & state index: ', Ez, i)
                        print('F now: ', F0, F1)
                        print('F previous: ', previous_Fs[jj][0], previous_Fs[jj][1])
                        print('')

                    
                plt.figure(i)
                
                
                
                if B_plot:
                    B_plot_array[jj][j] = B_combis[plot_index][0] - B_combis[plot_index][1]
                if E_plot:
                    E_plot_array[jj][j] = E_combis[plot_index][0] - E_combis[plot_index][1]
                if CPV_plot:
                    CPV_plot_array[jj][j] = E_PTV_0 - E_PTV_1
                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        EDM_plot_array[jj][j] = E_PTV_EDM_0 - E_PTV_EDM_1
                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        EDM_plot_array[jj][j] = E_PTV_EDM_0 - E_PTV_EDM_1
                        NSM_plot_array[jj][j] = E_PTV_NSM_0 - E_PTV_NSM_1
                        
                if E_energy_plot:
                    energy_plot_array[jj][j] = abs(evals[index_combis[plot_index][0]] - evals[index_combis[plot_index][1]])
          
                
    
                previous_Ms[jj][0] = M0
                previous_Ms[jj][1] = M1
                if '174' not in self.iso_state:
                    previous_F1s[jj][0] = F10
                    previous_F1s[jj][1] = F11
                previous_Fs[jj][0] = F0
                previous_Fs[jj][1] = F1 
                jj += 1
            
            
            evecs_old = evecs
            j += 1
        
        
        

        
        if interpolation is not None:
            B_plot_array_2 = np.zeros((len(state_plot_index_array), len(interpolation)))
            E_plot_array_2 = np.zeros((len(state_plot_index_array), len(interpolation)))
            for iii  in range(len(state_plot_index_array)): 
                f_E = interpolate.interp1d(Efield_array, E_plot_array[iii])
                f_B = interpolate.interp1d(Efield_array, B_plot_array[iii])                
                E_plot_array_2[iii] = f_E(interpolation)
                B_plot_array_2[iii] = f_B(interpolation)
                
            B_plot_array = np.zeros((len(state_plot_index_array), len(interpolation)))
            E_plot_array = np.zeros((len(state_plot_index_array), len(interpolation)))
            
            for iii  in range(len(state_plot_index_array)):              
                E_plot_array[iii] = E_plot_array_2[iii]
                B_plot_array[iii] = B_plot_array_2[iii]
                
            Efield_array = interpolation

                    
        for iii in range(len(state_plot_index_array)): 
            
            plt.figure(figsize=(figsize_x,figsize_y))
            
            if title_defalut:
                if '174' not in self.iso_state:
                    plt.title('Differential g factor, dipole moment, and CPV sensitivity\nstate G = '+ str(G0) + ' and ' + str(G1) + ', F1 = ' + str(F10) + ' and ' + str(F11) + ', F = ' + str(F0) + ' and ' + str(F1) + ', MF = ' + str(M0) + ' and ' + str(M1) + ' (state combination index: ' + str(state_plot_index_array[iii]) + ', ' + str(index_combis[state_plot_index_array[iii]][0]) + ', ' + str(index_combis[state_plot_index_array[iii]][1]) + ')', fontsize=fontsize)
                elif '174' in self.iso_state:
                    plt.title('Differential g factor, dipole moment, and CPV sensitivity\nstate F = ' + str(F0) + ' and ' + str(F1) + ', MF = ' + str(M0) + ' and ' + str(M1) + ' (state combination index: ' + str(state_plot_index_array[iii]) + ', ' + str(index_combis[state_plot_index_array[iii]][0]) + ', ' + str(index_combis[state_plot_index_array[iii]][1]) + ')', fontsize=fontsize)
                    
            plt.xlabel('Electric field (V/cm)', fontsize=fontsize)  
            
                            
            if ylabel_defalut is not True:
                if ylabel_defalut is not None:
                    plt.ylabel(ylabel_defalut, fontsize=fontsize)
            else:
                plt.ylabel('g factor, normalized dipole moment, and <STn>', fontsize=fontsize)
            
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
            #plt.hlines( 0, min(Bfield_array), max(Bfield_array), linestyles='dashed' , colors= 'orange')

            color_array = ['green','blue','red']
            
            if B_plot:
                plt.plot(Efield_array, B_plot_array[iii] , 'r-')
            if E_plot:
                plt.plot(Efield_array,  E_plot_array[iii], 'b-.')
            if CPV_plot:
                plt.plot(Efield_array,  CPV_plot_array[iii] , 'g-')
                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                    plt.plot(Efield_array,  EDM_plot_array[iii] , 'b-.')
                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                    plt.plot(Efield_array,  NSM_plot_array[iii] , 'b-.')
                    plt.plot(Efield_array,  EDM_plot_array[iii] , 'r:')
                if CPV_hlines is not None:
                    for k in range(len(CPV_hlines[iii])):
                        plt.hlines( CPV_hlines[iii][k], min(Bfield_array), max(Bfield_array) , linestyles='dashed', colors = color_array[k])
                    
            if legend_defalut is not True:
                plt.legend(legend_defalut, fontsize=fontsize)
            else:
                plt.legend(['g','d','CPV'], fontsize=fontsize)  
            
            if save_fig is not None:
                plt.savefig(save_fig + str(iii) + '_gdCPV_vs_E.pdf')
                
            plt.show()
            plt.clf()    
            
            
            if E_energy_plot:
                plt.figure(figsize=(figsize_x,figsize_y))
                plt.xlabel('Electric field (V/cm)', fontsize=fontsize)
                plt.ylabel('Energy (kHz)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                
                if offset[iii] in ['min']:
                    plt.plot(Efield_array, 1000*( energy_plot_array[iii] - min(energy_plot_array[iii]) )) 
                if offset[iii] in ['middle']:
                    plt.plot(Efield_array, 1000*( energy_plot_array[iii] - np.mean(energy_plot_array[iii]) ))
                if offset[iii] in ['max']:
                    plt.plot(Efield_array, 1000*( energy_plot_array[iii] - max(energy_plot_array[iii]) )) 
                
                if save_fig is not None:
                    plt.savefig(save_fig  + str(iii) + '_energy_vs_E.pdf')                
                plt.show()
                plt.clf()

        if combined_plot:
            plt.figure(figsize=(figsize_x,figsize_y)) 
            plt.xlabel('Electric field (V/cm)', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize) 
            for iii in range(len(state_plot_index_array)): 
                if iii == 0:
                    plt.plot(Efield_array,  E_plot_array[iii], 'g-.')
                    plt.plot(Efield_array, B_plot_array[iii] , 'r-')
                if iii == 1:
                    plt.plot(Efield_array, B_plot_array[iii] , 'b:')
                
                
                if width_plot_E is not None:
                    if iii == 0:
                        plt.fill_between(Efield_array,  E_plot_array[iii] - width_plot_E[0] ,  E_plot_array[iii] + width_plot_E[1] , facecolor='green', alpha=0.2)

                        
                if width_plot_B is not None:
                    if iii == 0:
                        plt.fill_between(Efield_array,  B_plot_array[iii] - width_plot_B[0] ,  B_plot_array[iii] + width_plot_B[1] , facecolor='red', alpha=0.2)
                    
                    if iii == 1:
                        plt.fill_between(Efield_array,  B_plot_array[iii] - width_plot_B[0] ,  B_plot_array[iii] + width_plot_B[1] , facecolor='blue', alpha=0.2)        
                    
                    
            plt.legend(legend_combined, fontsize=fontsize)
            if save_fig_combined is not None:
                plt.savefig(save_fig_combined  + '_combined.pdf')  

            plt.show()
            plt.clf()  
            
            
        if coherence_time_plot:
            
            print('coherence time is calcurated assuming 1 mV/cm E field and 1 uG B field fluctuations')
            
            plt.figure(figsize=(figsize_x,figsize_y)) 
            plt.xlabel('Electric field (V/cm)', fontsize=fontsize)
            plt.ylabel('Coherence time (s)', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize) 
            for iii in range(len(state_plot_index_array)): 
                if iii == 0:
                    coherence_time_E = 1/(abs(E_plot_array[iii]*dipole_line_broadening_1mVcm_in_Hz) + abs(np.gradient(E_plot_array[iii])*1e-3*dipole_line_broadening_1mVcm_in_Hz))
                    coherence_time_B = 1/(abs(B_plot_array[iii]*gfactor_line_broadening_1uG_in_Hz) + abs(np.gradient(B_plot_array[iii])*1e-3*gfactor_line_broadening_1uG_in_Hz))
                    coherence_time_total = 1/((1/coherence_time_E) + (1/coherence_time_B))

            plt.plot(Efield_array, coherence_time_E , 'g-.')
            plt.plot(Efield_array, coherence_time_B , 'r-')
            plt.plot(Efield_array, coherence_time_total , 'b:')
            
            plt.yscale('log')
            
            if coherence_time_y_lim is not None:
                plt.ylim(coherence_time_y_lim[0], coherence_time_y_lim[1])
            
            plt.legend(legend_coherence_time, fontsize=fontsize)
            if save_fig_coherence_time is not None:
                plt.savefig(save_fig_coherence_time  + '_coherence_time.pdf')  

            plt.show()
            plt.clf()  
        
        
        if gradient_plot:
            print('gradient is calculated assuming that intepolation is turned on')
            plt.figure(figsize=(figsize_x,figsize_y)) 
            plt.xlabel('Electric field (V/cm)', fontsize=fontsize)
            plt.ylabel('slope of sensitivities (/(V/cm))', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize) 
            for iii in range(len(state_plot_index_array)): 
                if iii == 0:
                    plt.plot(Efield_array, len(interpolation)/(interpolation[-1]-interpolation[0])*np.gradient(E_plot_array[iii]) , 'g-.')
                    plt.plot(Efield_array, len(interpolation)/(interpolation[-1]-interpolation[0])*np.gradient(B_plot_array[iii]) , 'r-')
            
            plt.legend(['d slope','g slope'], fontsize=fontsize)

            plt.show()
            plt.clf()
            
        #return  B_plot_array, E_plot_array, CPV_plot_array, energy_plot_array, coherence_time_E, coherence_time_B, coherence_time_total, np.gradient(E_plot_array[iii]), np.gradient(B_plot_array[iii])
        return  B_plot_array, E_plot_array, CPV_plot_array, energy_plot_array, np.gradient(E_plot_array[iii]), np.gradient(B_plot_array[iii])
    
    

    
    def find_E_B_zero_crossings(self,Efield_array,Bz, EDM_or_MQM, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, neighbor_state_rejection = False, already_zero_crossed_checck = False):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            
        elif  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            



        B_zero_crossing = []
        E_zero_crossing = []
        
        already_crossed = []

        Is_this_first_Efield = True

        previous_Ez = 0
        
        
        for Ez in Efield_array:            

            print('E field (V/cm): ', Ez)
            print(' ')

            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]

            g_effs = []
            Orientations = []

            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)





            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            if Is_this_first_Efield:
                previous_ds = np.zeros(len(B_combis))
                previous_gs = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                    
                if '174' in self.iso_state:
                    
                    J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                    J1 = self.q_numbers['J'][np.argmax(evec1**2)]                    

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 

                if (Is_this_first_Efield is False) and (np.sign(E_combis[i][0] - E_combis[i][1]) != np.sign(previous_ds[i])) or (np.sign(B_combis[i][0] - B_combis[i][1]) != np.sign(previous_gs[i])):                    

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1



                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       



                    #if (Is_this_first_Efield is False) and ((M0 != previous_Ms[i][0]) or (M1 != previous_Ms[i][1])):
                        #print('Caution! M value is not the same as the previous M value! E field (V/cm) & cobination & state index: ', Ez, i, index_combis[i][0], index_combis[i][1])
                        #print('M now: ', M0, M1)
                        #print('M previous: ', previous_Ms[i][0], previous_Ms[i][1])
                        #if '174' not in self.iso_state:
                            #print('F1 previous: ', previous_F1s[i][0], previous_F1s[i][1])
                        #print('F previous: ', previous_Fs[i][0], previous_Fs[i][1])
                        #print('')

                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]):
                        flag = True
                        if neighbor_state_rejection:
                            flag = False
                            if abs(index_combis[i][0] - index_combis[i][1]) != 1:
                                flag = True
                        
                        if already_zero_crossed_checck:
                            if i in already_crossed:
                                flag = False
                                
                        
                        if flag:
                            if np.sign(E_combis[i][0] - E_combis[i][1]) != np.sign(previous_ds[i]):

                                print('state index',  index_combis[i])
                                print('previous d', previous_ds[i])
                                print('now d', E_combis[i][0] - E_combis[i][1])
                                print('')

                                Ez_at_zero = previous_Ez + (Ez - previous_Ez)*abs(previous_ds[i])/(abs(previous_ds[i]) + abs(E_combis[i][0] - E_combis[i][1]) )
                                g_at_zero = previous_gs[i] + (B_combis[i][0] - B_combis[i][1] - previous_gs[i])*abs(previous_ds[i])/(abs(previous_ds[i]) + abs(E_combis[i][0] - E_combis[i][1]) )

                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    E_zero_crossing.append([Ez_at_zero, index_combis[i],  g_at_zero, E_PTV_EDM_0 - E_PTV_EDM_1 ,  E_PTV_NSM_0 - E_PTV_NSM_1 , E_PTV_0 - E_PTV_1 ])
                                    already_crossed.append(i)


                            if np.sign(B_combis[i][0] - B_combis[i][1]) != np.sign(previous_gs[i]):

                                print('state index',  index_combis[i] )
                                print('previous g',previous_gs[i])
                                print('now g', B_combis[i][0] - B_combis[i][1]) 
                                print('')

                                Ez_at_zero = previous_Ez + (Ez - previous_Ez)*abs(previous_gs[i])/(abs(previous_gs[i]) + abs(B_combis[i][0] - B_combis[i][1]) )
                                d_at_zero = previous_ds[i] + (E_combis[i][0] - E_combis[i][1] - previous_ds[i])*abs(previous_gs[i])/(abs(previous_gs[i]) + abs(B_combis[i][0] - B_combis[i][1]) )

                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    B_zero_crossing.append([Ez_at_zero, index_combis[i],  d_at_zero, E_PTV_EDM_0 - E_PTV_EDM_1 ,  E_PTV_NSM_0 - E_PTV_NSM_1 , E_PTV_0 - E_PTV_1 ])
                                    already_crossed.append(i)


                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                
                if '174' not in self.iso_state:
                    previous_F1s[i][0] = F10
                    previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous_ds[i] = E_combis[i][0] - E_combis[i][1]
                previous_gs[i] = B_combis[i][0] - B_combis[i][1]
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False
                
            previous_Ez = Ez

                                            
             
        return  E_zero_crossing, B_zero_crossing
    
    
    
    
    
    
    
    
    def plot_EB_zero_crossing_B_scan(self,Bfield_array,Ez, EDM_or_MQM, state_plot_index_array, figsize_x = 15, figsize_y = 7.5, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, B_plot = True, E_plot = True, CPV_plot = True, E_energy_plot = True, title_defalut = True, ylabel_defalut = True, legend_defalut = True,  fontsize=20, state_track = False, save_fig = None, CPV_hlines = None, offset = 'middle'):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            
        elif  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
            H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
            H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            
        else:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)


        previous_Ms = np.zeros((len(state_plot_index_array),2))
        previous_F1s = np.zeros((len(state_plot_index_array),2))
        previous_Fs = np.zeros((len(state_plot_index_array),2))  
        
        
        B_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))
        E_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))
        CPV_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))
        if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
            EDM_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))
        if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
            EDM_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))                
            NSM_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))  
        energy_plot_array = np.zeros((len(state_plot_index_array), len(Bfield_array)))


        j = 0
        
        for Bz in Bfield_array:            

            print('B field (G): ', Bz)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            #B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            jj = 0
            for i in state_plot_index_array:
                
                plot_index = i

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                if CPV_plot:
                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1

                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1                    

                    
                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                
                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]
       

              

                if j > 0:
                    if (M0 != previous_Ms[jj][0]) or (M1 != previous_Ms[jj][1]):
                        print('Caution! M value is not the same as the previous M value! B field (G) & cobination & state index: ', Bz, i, index_combis[i][0], index_combis[i][1])
                        print('M now: ', M0, M1)
                        print('M previous: ', previous_Ms[jj][0], previous_Ms[jj][1])
                        if '174' not in self.iso_state:
                            print('F1 previous: ', previous_F1s[jj][0], previous_F1s[jj][1])
                        print('F previous: ', previous_Fs[jj][0], previous_Fs[jj][1])
                        print('')
                        
                        if state_track:
                            
                            print('state track is on. We will fix the state index.')
                            
                            overlaps = []
                            for e in range(len(evecs)):
                                if self.q_numbers['M'][np.argmax(evecs[e]**2)] == previous_Ms[jj][0]:
                                    overlaps.append(abs(evecs_old[index_combis[i][0]]@evecs[e].T))
                                else:
                                    overlaps.append(0)
                                
                            overlaps = np.array(overlaps)
                            state_track_index0 = np.argmax(overlaps)
                            
                            print('new state index for 0', state_track_index0)
                            
                        
                            overlaps = []
                            for e in range(len(evecs)):
                                if self.q_numbers['M'][np.argmax(evecs[e]**2)] == previous_Ms[jj][1]:
                                    overlaps.append(abs(evecs_old[index_combis[i][1]]@evecs[e].T))
                                else:
                                    overlaps.append(0)
                                        
                            overlaps = np.array(overlaps)
                            state_track_index1 = np.argmax(overlaps)  
                            
                            print('new state index for 1', state_track_index1)
                            
                            M0 = self.q_numbers['M'][np.argmax(evecs[state_track_index0]**2)]
                            M1 = self.q_numbers['M'][np.argmax(evecs[state_track_index1]**2)]
                            
                            print('M new', M0, M1)

                            if '174' not in self.iso_state:

                                G0 = self.q_numbers['G'][np.argmax(evecs[state_track_index0]**2)]
                                G1 = self.q_numbers['G'][np.argmax(evecs[state_track_index1]**2)]

                                F10 = self.q_numbers['F1'][np.argmax(evecs[state_track_index0]**2)]
                                F11 = self.q_numbers['F1'][np.argmax(evecs[state_track_index1]**2)]
                                
                                print('G new', G0, G1)
                                print('F1 new', F10, F11)

                            F0 = self.q_numbers['F'][np.argmax(evecs[state_track_index0]**2)]
                            F1 = self.q_numbers['F'][np.argmax(evecs[state_track_index1]**2)]
                            
                            print('F new', F0, F1)
                           
                            
                            
                            
                            for ee in range(len(index_combis)):
                                if (index_combis[ee][0] == state_track_index0) and (index_combis[ee][1] == state_track_index1):
                                    state_plot_index_array[jj] = ee
                                    plot_index = ee
                                    print('new state index for combination array', ee, state_plot_index_array[jj])
                                    print('')       
                                    
                            if CPV_plot:
                                
                                evec0 = evecs[state_track_index0]
                                evec1 = evecs[state_track_index1]  
                                
                                E_PTV_0 = evec0@H_PTV@evec0
                                E_PTV_1 = evec1@H_PTV@evec1

                                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                                    E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                                    E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                                    E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                                    E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       
                        
                        
                    if '174' not in self.iso_state:
                        if (F10 != previous_F1s[jj][0]) or (F11 != previous_F1s[jj][1]):
                            print('Caution! F1 value is not the same as the previous F1 value! B field (G) & state index: ', Bz, i)
                            print('F1 now: ', F10, F11)
                            print('F1 previous: ', previous_F1s[jj][0], previous_F1s[jj][1])
                            print('')

                    if (F0 != previous_Fs[jj][0]) or (F1 != previous_Fs[jj][1]):
                        print('Caution! F value is not the same as the previous F value! B field (G) & state index: ', Bz, i)
                        print('F now: ', F0, F1)
                        print('F previous: ', previous_Fs[jj][0], previous_Fs[jj][1])
                        print('')

                    
                plt.figure(i)
                
                
                
                if B_plot:
                    B_plot_array[jj][j] = B_combis[plot_index][0] - B_combis[plot_index][1]
                if E_plot:
                    E_plot_array[jj][j] = E_combis[plot_index][0] - E_combis[plot_index][1]
                if CPV_plot:
                    CPV_plot_array[jj][j] = E_PTV_0 - E_PTV_1
                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        EDM_plot_array[jj][j] = E_PTV_EDM_0 - E_PTV_EDM_1
                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        EDM_plot_array[jj][j] = E_PTV_EDM_0 - E_PTV_EDM_1
                        NSM_plot_array[jj][j] = E_PTV_NSM_0 - E_PTV_NSM_1
                        
                if E_energy_plot:
                    energy_plot_array[jj][j] = abs(evals[index_combis[plot_index][0]] - evals[index_combis[plot_index][1]])
          
                
                
    
                previous_Ms[jj][0] = M0
                previous_Ms[jj][1] = M1
                if '174' not in self.iso_state:
                    previous_F1s[jj][0] = F10
                    previous_F1s[jj][1] = F11
                previous_Fs[jj][0] = F0
                previous_Fs[jj][1] = F1 
                jj += 1
            
            
            evecs_old = evecs
            j += 1
        
        
        

        
        for iii in range(len(state_plot_index_array)): 
            
            plt.figure(figsize=(figsize_x,figsize_y))
            
            if title_defalut:
                if '174' not in self.iso_state:
                    plt.title('Differential g factor, dipole moment, and CPV sensitivity\nstate G = '+ str(G0) + ' and ' + str(G1) + ', F1 = ' + str(F10) + ' and ' + str(F11) + ', F = ' + str(F0) + ' and ' + str(F1) + ', MF = ' + str(M0) + ' and ' + str(M1) + ' (state combination index: ' + str(state_plot_index_array[iii]) + ', ' + str(index_combis[state_plot_index_array[iii]][0]) + ', ' + str(index_combis[state_plot_index_array[iii]][1]) + ')', fontsize=fontsize)
                elif '174' in self.iso_state:
                    plt.title('Differential g factor, dipole moment, and CPV sensitivity\nstate F = ' + str(F0) + ' and ' + str(F1) + ', MF = ' + str(M0) + ' and ' + str(M1) + ' (state combination index: ' + str(state_plot_index_array[iii]) + ', ' + str(index_combis[state_plot_index_array[iii]][0]) + ', ' + str(index_combis[state_plot_index_array[iii]][1]) + ')', fontsize=fontsize)
                    
            plt.xlabel('Magnetic field (G)', fontsize=fontsize)  
            
            if ylabel_defalut is not True:
                plt.ylabel(ylabel_defalut, fontsize=fontsize)
            else:
                plt.ylabel('g factor, normalized dipole moment, and <STn>', fontsize=fontsize)
            
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
            #plt.hlines( 0, min(Bfield_array), max(Bfield_array), linestyles='dashed' , colors= 'orange')

            color_array = ['green','blue','red']
            
            if B_plot:
                plt.plot(Bfield_array, B_plot_array[iii] , 'r-')
            if E_plot:
                plt.plot(Bfield_array,  E_plot_array[iii], 'b-.')
            if CPV_plot:
                plt.plot(Bfield_array,  CPV_plot_array[iii] , 'g-')
                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                    plt.plot(Bfield_array,  EDM_plot_array[iii] , 'b-.')
                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                    plt.plot(Bfield_array,  NSM_plot_array[iii] , 'b-.')
                    plt.plot(Bfield_array,  EDM_plot_array[iii] , 'r:')
                if CPV_hlines is not None:
                    for k in range(len(CPV_hlines[iii])):
                        plt.hlines( CPV_hlines[iii][k], min(Bfield_array), max(Bfield_array) , linestyles='dashed', colors = color_array[k])
                    
            if legend_defalut is not True:
                plt.legend(legend_defalut, fontsize=fontsize)
            else:
                plt.legend(['g','d','CPV'], fontsize=fontsize)  
            
            if save_fig is not None:
                plt.savefig(save_fig + str(iii) + '_gdCPV_vs_B.pdf')
            plt.show()
            plt.clf()            
            
            
            if E_energy_plot:
                plt.figure(figsize=(figsize_x,figsize_y))
                plt.xlabel('Magnetic field (G)', fontsize=fontsize)
                plt.ylabel('Energy (kHz)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                
                if offset[iii] in ['min']:
                    plt.plot(Bfield_array, 1000*( energy_plot_array[iii] - min(energy_plot_array[iii]) )) 
                if offset[iii] in ['middle']:
                    plt.plot(Bfield_array, 1000*( energy_plot_array[iii] - np.mean(energy_plot_array[iii]) ))
                if offset[iii] in ['max']:
                    plt.plot(Bfield_array, 1000*( energy_plot_array[iii] - max(energy_plot_array[iii]) )) 
                
                if save_fig is not None:
                    plt.savefig(save_fig  + str(iii) + '_energy_vs_B.pdf')                
                plt.show()
                plt.clf()

        return  B_plot_array, E_plot_array, CPV_plot_array, energy_plot_array

   
    
    def find_MQM_science_states(self,Efield_array,Bz, EDM_or_MQM, g_criteria = 10,  d_criteria = 10, CPV_criteria = -1, M_criteria = 10, M_specify = None, F1_specify = None,  G_specify = None, parity_specify = None, ground_states_isolation = None, level_diagram_show = False, stretch_check = False, frequency_criteria = None, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, neighbor_state_rejection = False, interpolation_number = 200, show_max_B_and_E_coherence_time = True, plot_coherence_time = False, chousei0 = 100, chousei1 = 100, figsize = (12,6), width = 0.75, minimum_calculation =  True):
        
        print('coherence time is calcurated assuming 1 mV/cm E field and 1 uG B field fluctuations')
        print(' ')

        if ground_states_isolation is not None:
            print('ground_states_isolation msut be specified in the following way [distance_from_the_state_to_consider, criteria_for_isolation_with_the_closest_state_with_different_M, criteria_for_isolation_with_frequency]')
            print(' ')
            
        if stretch_check:
            print('ground_states_isolation needs to be specfied to enable stretch_check')
            print('')
            
            
        if frequency_criteria is not None:
            print('frequency_criteria needs to be specfied in the following way: [freq_low, freq_high]')
            print('')
            
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
            
        elif  '171' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
            
        elif  '173' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            else:
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
                
                
        found_magic_index = []
        BE_magic_index = []

        Is_this_first_Efield = True

        previous_Ez = 0
        
        
        for Ez in Efield_array:            

            print('E field (V/cm): ', Ez)
            print(' ')

            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)

            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]

            g_effs = []
            Orientations = []

            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)





            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            if Is_this_first_Efield:
                previous_ds = np.zeros(len(B_combis))
                previous_gs = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):

                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]

                N0 = self.q_numbers['N'][np.argmax(evec0**2)]
                N1 = self.q_numbers['N'][np.argmax(evec1**2)]
                
                if '174' not in self.iso_state:

                    G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                    G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                    F11 = self.q_numbers['F1'][np.argmax(evec1**2)]
                    
                if '174' in self.iso_state:
                    
                    J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                    J1 = self.q_numbers['J'][np.argmax(evec1**2)]                    

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]   
                
                dnow = E_combis[i][0] - E_combis[i][1]
                gnow = B_combis[i][0] - B_combis[i][1]
                

                if Is_this_first_Efield is False:   
                    
                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1



                    if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1

                    if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                        E_PTV_EDM_0 = evec0@H_PTV_EDM@evec0
                        E_PTV_EDM_1 = evec1@H_PTV_EDM@evec1
                        E_PTV_NSM_0 = evec0@H_PTV_NSM@evec0
                        E_PTV_NSM_1 = evec1@H_PTV_NSM@evec1       


                    #The following is modified on Feb 18 2024
                    #if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]) and (previous_gs[i]*previous_ds[i]*dnow*gnow != 0):
                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) == M_criteria):  
                        
                        
                        Parity0 = evec0@self.Parity_mat@evec0
                        Parity1 = evec1@self.Parity_mat@evec1
                        
                        energy0 =  evals[index_combis[i][0]]
                        energy1 =  evals[index_combis[i][1]]
                        

                        flag_frequency = True
                        flag_neighbor = True
                        flag_M = True
                        flag_F1 = True
                        flag_G = True
                        flag_parity = True
                        flag_isolation = True
                        flag_stretch_state = True
                        
                        
                        if frequency_criteria is not None:
                            if (energy1 - energy0) < frequency_criteria[0] or (energy1 - energy0) > frequency_criteria[1]:
                                flag_frequency = False
                                
                                
                        if neighbor_state_rejection:
                            flag_neighbor = False
                            if abs(index_combis[i][0] - index_combis[i][1]) != 1:
                                flag_neighbor = True
                                
                                
                        if M_specify is not None:
                            flag_M = False
                            if (M0 in M_specify) and (M1 in M_specify):
                                flag_M = True

                        if F1_specify is not None:
                            flag_F1 = False
                            if (F10 in F1_specify) and (F11 in F1_specify):
                                flag_F1 = True
                                
                        if G_specify is not None:
                            flag_G = False
                            if (G0 in G_specify) and (G1 in G_specify):
                                flag_G = True
                                
                        if parity_specify is not None:
                            flag_parity = False
                            if (Parity0 >= parity_specify[0]) and (Parity0 <= parity_specify[1]) and (Parity1 >= parity_specify[0]) and (Parity1 <= parity_specify[1]):
                                flag_parity = True
                                
                                
                        if ground_states_isolation is not None:
                            
                            flag_stretch_state_0 = True
                            flag_stretch_state_1 = True
                            M_list_0 = []
                            M_list_1 = []
                            state_list_0 = []
                            state_list_1 = []
                            frequency_list = []
                            for k in range(len(evals)):
                                if (abs(evals[k] - energy0) <= ground_states_isolation[0]) and (abs(evals[k] - energy0) > 0):
                                    Mk = self.q_numbers['M'][np.argmax(evecs[k]**2)]
                                    
                                    state_list_0.append([evals[k],Mk])
                                    M_list_0.append(Mk)
                                        
                                    if Mk != -M0:
                                        if abs(evals[k] - energy0) < ground_states_isolation[1]:
                                            flag_isolation = False
                                            
                                
                                if (abs(evals[k] - energy1) <= ground_states_isolation[0]) and (abs(evals[k] - energy1) > 0):
                                    Mk = self.q_numbers['M'][np.argmax(evecs[k]**2)]
                                    
                                    state_list_1.append([evals[k],Mk])
                                    M_list_1.append(Mk)
                                        
                                    if Mk != -M0:
                                        if abs(evals[k] - energy1) < ground_states_isolation[1]:
                                            flag_isolation = False      
                                        
                            for l in range(len(state_list_0)):
                                for m in range(l+1, len(state_list_1)):
                                    if abs(state_list_1[m][1] - state_list_1[m][1]) <= M_criteria:
                                        frequency_list.append(state_list_1[m][0] - state_list_0[l][0])
                            
                            
                            for n in range(len(frequency_list)):
                                for p in range(n+1, len(frequency_list)):
                                    if  abs(frequency_list[p] - (energy1 - energy0)) < ground_states_isolation[2]:
                                        flag_isolation = False
                            
                            if M_list_0:
                                M_abs_max_0 = max(abs(np.array(M_list_0)))
                            if M_list_1:
                                M_abs_max_1 = max(abs(np.array(M_list_1)))

                            
                            if stretch_check:
                                if (M_abs_max_0 > abs(M0)) and (M_abs_max_1 > abs(M1)):
                                    flag_stretch_state = False
                                

                                
                        if flag_frequency*flag_neighbor*flag_M*flag_F1*flag_G*flag_parity*flag_isolation*flag_stretch_state:
                            
                            fg = interpolate.interp1d([previous_Ez, Ez], [previous_gs[i], gnow])
                            fd = interpolate.interp1d([previous_Ez, Ez], [previous_ds[i], dnow])

                            g_arrays = fg(np.linspace(previous_Ez, Ez, interpolation_number))
                            d_arrays = fd(np.linspace(previous_Ez, Ez, interpolation_number))
                            
                            minimum_g = min(abs(g_arrays))
                            minimum_d = min(abs(d_arrays))
                            
                            coherence_time_E = 1/(abs(d_arrays*dipole_line_broadening_1mVcm_in_Hz) + abs(np.gradient(d_arrays)*1e-3*dipole_line_broadening_1mVcm_in_Hz))
                            coherence_time_B = 1/(abs(g_arrays*gfactor_line_broadening_1uG_in_Hz) + abs(np.gradient(g_arrays)*1e-3*gfactor_line_broadening_1uG_in_Hz))
                            coherence_time_total = 1/((1/coherence_time_E) + (1/coherence_time_B))

                            max_coherence_time = max(coherence_time_total)
                            max_E_coherence_time = max(coherence_time_E)
                            max_B_coherence_time = max(coherence_time_B)

                            if minimum_calculation is False:
                                minimum_g = abs(B_combis[i][0] - B_combis[i][1])
                                minimum_d = abs(E_combis[i][0] - E_combis[i][1])
                            
                            if minimum_g <= g_criteria and minimum_d <= d_criteria:
                                
                                max_coherence_time_index = np.argmax(coherence_time_total)
                                max_E_coherence_time_index = np.argmax(coherence_time_E)
                                max_B_coherence_time_index = np.argmax(coherence_time_B)
                                

                                E_at_max_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_coherence_time_index]
                                E_at_max_E_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_E_coherence_time_index]
                                E_at_max_B_coherence_time = np.linspace(previous_Ez, Ez, interpolation_number)[max_B_coherence_time_index]
                                
                                g_at_max_coherence_time = fg(E_at_max_coherence_time)
                                d_at_max_coherence_time = fd(E_at_max_coherence_time)

                                g_at_max_E_coherence_time = fg(E_at_max_E_coherence_time)
                                d_at_max_E_coherence_time = fd(E_at_max_E_coherence_time)
                                g_at_max_B_coherence_time = fg(E_at_max_B_coherence_time)
                                d_at_max_B_coherence_time = fd(E_at_max_B_coherence_time)                                
                                
                                dgdE_at_max_coherence_time = interpolation_number/(Ez - previous_Ez)*np.gradient(g_arrays)[max_coherence_time_index]
                                
                                d_zero_crossed = False
                                g_zero_crossed = False
                                if np.sign(dnow) != np.sign(previous_ds[i]):
                                    d_zero_crossed = True
                                if np.sign(gnow) != np.sign(previous_gs[i]):
                                    g_zero_crossed = True

                                
                                if i not in found_magic_index:
                                    found_magic_index.append(i)

                                BE_magic_index.append([index_combis[i],Ez])                            

                                print('state index: ',index_combis[i], i)
                                print('energy (MHz): ', np.round(energy0, round), np.round(energy1, round))
                                print('Transition frequency (MHz): ', energy1 - energy0)    
                                print('N: ', N0, N1 )
                                if '174' in self.iso_state:
                                    print('J: ', J0, J1 )
                                if '174' not in self.iso_state:
                                    print('G: ', G0, G1 )
                                    print('F1: ', F10, F11 )
                                print('F: ', F0, F1 )
                                print('M_F: ', M0, M1 )
                                print('Parity: ', Parity0, Parity1 )
                                #print('energy separation with upper/lower state for state 0 : ', upper_level_position_0 - energy0, energy0  - lower_level_position_0 )
                                #print('energy separation with upper/lower state for state 1 : ', upper_level_position_1 - energy1, energy1  - lower_level_position_1 )
                                print('minimum g', minimum_g)
                                print('minimum d', minimum_d)
                                print('max coherence time between',previous_Ez, 'and', Ez, 'V/cm:', max_coherence_time,'sec')
                                print('max coherence time at', E_at_max_coherence_time, 'V/cm')
                                
                                if g_zero_crossed:
                                    print('g zero crossing!')
                                if d_zero_crossed:
                                    print('d zero crossing!')
                                    if plot_coherence_time:
                                        print('dg/dE at max coherence time:', dgdE_at_max_coherence_time)
                                        print('(dg/dE)/(CPV) at max coherence time:', dgdE_at_max_coherence_time/(E_PTV_0 - E_PTV_1),'(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                    #print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                
                                if plot_coherence_time:
                                    print('g at max coherence time:', g_at_max_coherence_time)
                                    print('d at max coherence time:', d_at_max_coherence_time)

                                    if show_max_B_and_E_coherence_time:
                                        print('max E coherence time at', E_at_max_E_coherence_time, 'V/cm')
                                        print('max E coherence time', max_E_coherence_time, 'sec')
                                        print('g at max E coherence time:', g_at_max_E_coherence_time)
                                        print('d at max E coherence time:', d_at_max_E_coherence_time)
                                        print('max B coherence time at', E_at_max_B_coherence_time, 'V/cm')
                                        print('max B coherence time', max_B_coherence_time, 'sec')
                                        print('g at max B coherence time:', g_at_max_B_coherence_time)
                                        print('d at max B coherence time:', d_at_max_B_coherence_time) 

                                print('At',Ez,'V/cm:')
                                print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ',np.round( B_combis[i][0] - B_combis[i][1], round))
                                print('polarization: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                                print('(Note that CPV sensitivities are at', Ez, 'V/cm)')
                                print('PTV shifts: ', E_PTV_0, E_PTV_1, ' difference: ', np.round(E_PTV_0 - E_PTV_1,round))
                                if  ('171' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                if  ('173' in self.iso_state) and (EDM_or_MQM == 'all'):
                                    print('EDM shifts: ', E_PTV_EDM_0, E_PTV_EDM_1, ' difference: ', np.round(E_PTV_EDM_0 - E_PTV_EDM_1,round))
                                    print('NSM shifts: ', E_PTV_NSM_0, E_PTV_NSM_1, ' difference: ', np.round(E_PTV_NSM_0 - E_PTV_NSM_1,round))
                                    print('MQM/EDM shifts: ', np.round( (E_PTV_0 - E_PTV_1)/(E_PTV_EDM_0 - E_PTV_EDM_1),round))

                                print(' ')
                                print(' ')
                                
                                if M_list_0:
                                    if M_list_1:
                                        if level_diagram_show:


                                            plt.figure(figsize=figsize )
                                            plt.xlim(-max(M_abs_max_0, M_abs_max_1)-1,max(M_abs_max_0, M_abs_max_1)+1)
                                            for q in range(len(state_list_1)):
                                                plt.hlines(state_list_1[q][0],state_list_1[q][1]-width/2,state_list_1[q][1]+width/2)
                                            plt.hlines(energy1,M1-width/2,M1+width/2, colors = 'red')
                                            plt.show()
                                            plt.clf()

                                            plt.figure(figsize=figsize )
                                            plt.xlim(-max(M_abs_max_0, M_abs_max_1)-1,max(M_abs_max_0, M_abs_max_1)+1)
                                            for q in range(len(state_list_0)):
                                                plt.hlines(state_list_0[q][0],state_list_0[q][1]-width/2,state_list_0[q][1]+width/2)
                                            plt.hlines(energy0,M0-width/2,M0+width/2, colors = 'red')
                                            plt.show() 
                                            plt.clf()

                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                
                if '174' not in self.iso_state:
                    previous_F1s[i][0] = F10
                    previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous_ds[i] = E_combis[i][0] - E_combis[i][1]
                previous_gs[i] = B_combis[i][0] - B_combis[i][1]
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False
                
            previous_Ez = Ez

        found_magic_index = np.array(found_magic_index)
        
        print('# of identified magic transitions',len(found_magic_index))                                            
             
        return  BE_magic_index
            
    
    
    
    

    def find_B_degeneracy_NSDPV(self,Bfield_array,Ez, EDM_or_MQM,  energy_criteria = 10, NSDPV_criteria = -1,  step_B = 1e-4, step_E = 1e-2, idx = None, round = None):
        
        '''
        if EDM_or_MQM != 'NSDPV':
            print('This function only accepts NSDPV for EDM_or_MQM')
        else:
        '''
        self.PTV_type = EDM_or_MQM
        H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)


        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        F1_is_same = True
        F_is_same = True
        
        for Bz in Bfield_array:            
            
            print('B field (V/cm): ', Bz)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            B_zero_crossing = []


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            NSDPV_shifts = np.zeros(len(index_combis))
            energy_differences = np.zeros(len(index_combis))
            
            if Is_this_first_Efield:
                previous = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))
                previous_energys = np.zeros(len(B_combis))     
                
                
            for i in range(len(NSDPV_shifts)):
                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]
                NSDPV_shifts[i] = evec1@H_PTV@evec0
                energy_differences[i] = evals[index_combis[i][1]] - evals[index_combis[i][0]]
                
                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 
                
            
                Parity0 = evec0@self.Parity_mat@evec0                    
                Parity1 = evec1@self.Parity_mat@evec1    
                
                if (Is_this_first_Efield is False) and (abs(energy_differences[i]) < energy_criteria) and (abs(NSDPV_shifts[i]) > NSDPV_criteria):                    


                    print('state index: ',index_combis[i], i)
                    print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                    print('Parity: ', Parity0, Parity1 )
                    print('G: ', G0, G1 )
                    print('F1: ', F10, F11 )
                    print('F: ', F0, F1 )
                    print('M_F: ', M0, M1 )
                    print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                    print('g factor difference previous: ',np.round( previous[i], round))
                    print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                    print('NSDPV shifts: ',  np.round(NSDPV_shifts[i],round))  
                    print(' ')
                        
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                previous_F1s[i][0] = F10
                previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous[i] = B_combis[i][0] - B_combis[i][1]
           
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            B_zero_crossing = np.array(B_zero_crossing)
            
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all
    

    
    
    def find_E_degeneracy_NSDPV(self,Efield_array,Bz, EDM_or_MQM,  energy_criteria = 10, NSDPV_criteria = -1,  step_B = 1e-4, step_E = 1e-2, idx = None, round = None):
        
        '''
        if EDM_or_MQM != 'NSDPV':
            print('This function only accepts NSDPV for EDM_or_MQM')
        else:
        '''
        self.PTV_type = EDM_or_MQM
        H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)


        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        F1_is_same = True
        F_is_same = True
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            B_zero_crossing = []


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            NSDPV_shifts = np.zeros(len(index_combis))
            energy_differences = np.zeros(len(index_combis))
            
            if Is_this_first_Efield:
                previous = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))
                previous_energys = np.zeros(len(B_combis))     
                
                
            for i in range(len(NSDPV_shifts)):
                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]
                NSDPV_shifts[i] = evec1@H_PTV@evec0
                energy_differences[i] = evals[index_combis[i][1]] - evals[index_combis[i][0]]
                
                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 
                
            
                Parity0 = evec0@self.Parity_mat@evec0                    
                Parity1 = evec1@self.Parity_mat@evec1    
                
                if (Is_this_first_Efield is False) and (abs(energy_differences[i]) < energy_criteria) and (abs(NSDPV_shifts[i]) > NSDPV_criteria):                    


                    print('state index: ',index_combis[i], i)
                    print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                    print('Parity: ', Parity0, Parity1 )
                    print('G: ', G0, G1 )
                    print('F1: ', F10, F11 )
                    print('F: ', F0, F1 )
                    print('M_F: ', M0, M1 )
                    print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                    print('g factor difference previous: ',np.round( previous[i], round))
                    print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                    print('NSDPV shifts: ',  np.round(NSDPV_shifts[i],round))  
                    print(' ')
                        
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                previous_F1s[i][0] = F10
                previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous[i] = B_combis[i][0] - B_combis[i][1]
           
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            B_zero_crossing = np.array(B_zero_crossing)
            
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all
    
    
    
    def g_eff_evecs_Arian(self,evals,evecs,Ez,Bz,step=1e-7):
        evals0,evecs0 = evals,evecs
        evals1,evecs1 = self.eigensystem(Ez,Bz+step,set_attr=False)
        order = state_ordering(evecs0,evecs1,round=self.round)
        # evecs1_ordered = evecs1[order,:]
        evals1_ordered = evals1[order]
        # sgn = np.sign(np.diag(evecs1_ordered@evecs0.T))
        # evecs1_ordered = (evecs1_ordered.T*sgn).T
        g_eff = []
        for E0,E1 in zip(evals0,evals1_ordered):
            g_eff.append((E1-E0)/(step*self.parameters['mu_B']))
        g_eff = np.array(g_eff)
        return g_eff


    
    
#Is this redundant?
    # def g_eff_EB(self,Ez,Bz,step=1e-7):
    #     self.eigensystem(Ez,Bz)
    #     g_eff = self.g_eff(step=step)
    #     return g_eff

    def trap_matrix(self,theta_trap=None,output=True,rank=2):
        if theta_trap is None:
            theta_trap = self.theta_trap
        H_trap = tensor_matrix(theta_trap,self.q_numbers,self.q_numbers,self.parameters,self.matrix_elements,rank=rank)
        self.H_trap = H_trap
        #Note does not include I_trap
        if output:
            return H_trap

    def trap_shift_EB(self,I_trap=None,theta_trap=None,Ez=None,Bz=None,step=0.1):
        recalc = False
        if Ez is None:
            Ez = self.E0
        if Bz is None:
            Bz = self.B0
        if I_trap is None:
            I_trap = self.I_trap
        if theta_trap is None:
            theta_trap = self.theta_trap
        evals,evecs = diagonalize(self.H_function(Ez,Bz,I_trap,theta_trap),round=self.round)
        return self.trap_shift_evecs(evals,sevecs,I0,theta,Ez,Bz,step=step)

    def trap_shift_evecs(self,evals,evecs,I0,theta,Ez,Bz,step=0.1):
        evals0,evecs0 = evals,evecs
        evals1,evecs1 = diagonalize(self.H_function(Ez,Bz,I_trap*(1-step),theta_trap),round=self.round)
        order = state_ordering(evecs0,evecs1,round=self.round)
        # evecs1_ordered = evecs1[order,:]
        evals1_ordered = evals1[order]
        # sgn = np.sign(np.diag(evecs1_ordered@evecs0.T))
        # evecs1_ordered = (evecs1_ordered.T*sgn).T
        shifts = []
        for E0,E1 in zip(evals0,evals1_ordered):
            shifts.append(E1-E0)
        shifts = np.array(shifts)
        self.trap_shifts = shifts
        return shifts

    def PTV_shift(self,EDM_or_MQM):
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
        self.H_PTV = H_PTV
        evals,evecs = self.evals0,self.evecs0
        PTV_shift = []
        for evec in evecs:
            E_PTV = evec@H_PTV@evec
            PTV_shift.append(E_PTV)
        PTV_shift = np.array(PTV_shift)
        return PTV_shift

    def PTV_shift_EB(self,Ez,Bz,EDM_or_MQM):
        self.eigensystem(Ez,Bz)
        PTV_shift = self.PTV_shift(EDM_or_MQM)
        return PTV_shift

    def plot_evals_EB(self,E_or_B, offset_middle = None, fontsize = 14, plot_title = True, idx=None,kV_kG=False, GHz=False,Freq=True):
        x_scale = {False: 1, True: 10**-3}[kV_kG]
        y_scale = {False: 1, True: 10**-3}[GHz]

        field,evals = {
            'E': (self.Ez,self.evals_E),
            'B': (self.Bz,self.evals_B)
        }[E_or_B]
        

        evals = evals.T

        if idx is not None:
            evals = evals[idx]

        x_label = {
            True: {'E': 'Electric field (kV/cm)', 'B': 'Magnetic field (kGauss)'}[E_or_B],
            False: {'E': 'Electric field (V/cm)', 'B': 'Magnetic field (Gauss)'}[E_or_B]
        }[kV_kG]

        y_label = {
            True: 'Energy (GHz)',
            False: 'Energy (MHz)'
        }[GHz]
        state_str = self.state_str

        EB_str = {
            'E': 'Stark Shifts',
            'B': 'Zeeman Shifts'
        }[E_or_B]

        title = state_str + ' ' + EB_str + r', $N={}$'.format(str(self.N_range)[1:-1])

        plt.figure(figsize=(10,7))
        for trace in evals:
            if offset_middle is not None:
                plt.plot(x_scale*field,y_scale*trace - y_scale*offset_middle)
            else:
                plt.plot(x_scale*field,y_scale*trace)
        plt.xlabel(x_label,fontsize=fontsize)
        plt.ylabel(y_label,fontsize=fontsize)
        if plot_title:
            plt.title(title,fontsize=fontsize)
        return


    def write_state(self,eval_i,Ez=None,Bz=None,show_PTV=False):
        if Ez is None and Bz is None:
            pass
        else:
            if Ez==self.E0 and Bz==self.B0:
                pass
            else:
                self.eigensystem(Ez,Bz,set_attr=True)
        i=eval_i
        if i<0:
            i = len(self.evals0)+i
        vector=self.evecs0[i]
        energy = self.evals0[i]
        print('E = {} MHz\n'.format(energy))
        if self.PTV0 is not None and show_PTV:
            print('{} Shift = {}\n'.format(self.PTV_type,self.PTV0[i]))
        #sum_sq = 0
        for index in np.nonzero(vector)[0]:
            v={q:self.q_numbers[q][index] for q in self.q_numbers}
            coeff = vector[index]
            if self.hunds_case == 'bBS':
                print(' {} |K={},N={},G={},F1={},F={},M={}> \n'.format(coeff,v['K'],v['N'],v['G'],v['F1'],v['F'],v['M']))
            elif self.hunds_case == 'bBJ':
                print(' {} |K={},N={},J={},F={},M={}> \n'.format(coeff,v['K'],v['N'],v['J'],v['F'],v['M']))
            elif self.hunds_case == 'aBJ':
                print(' {} |K={},\u03A3={},P={},J={},F={},M={}> \n'.format(coeff,v['K'],v['Sigma'],v['P'],v['J'],v['F'],v['M']))


    def PTV_Map(self,EDM_or_MQM,E_or_B='E', plot=False):
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
        self.H_PTV = H_PTV
        if E_or_B=='E':
            PTV_vs_E = []
            if self.evecs_E is None:
                print('Run StarkMap first')
                return None
            for evecs in self.evecs_E:
                PTV_shift = []
                for evec0 in evecs:
                    E_PTV = evec0@H_PTV@evec0
                    PTV_shift.append(np.round(E_PTV,self.round))
                PTV_shift = np.array(PTV_shift)
                PTV_vs_E.append(PTV_shift)
            PTV_vs_E = np.array(PTV_vs_E)
            self.PTV_E = PTV_vs_E
        elif E_or_B=='B':
            PTV_vs_B = []
            if self.evecs_B is None:
                print('Run ZeemanMap first')
                return None
            for evecs in self.evecs_B:
                PTV_shift = []
                for evec0 in evecs:
                    E_PTV = evec0@H_PTV@evec0
                    PTV_shift.append(np.round(E_PTV,self.round))
                PTV_shift = np.array(PTV_shift)
                PTV_vs_B.append(PTV_shift)
            PTV_vs_B = np.array(PTV_vs_B)
            self.PTV_B = PTV_vs_B
        if plot:
            self.plot_PTV(E_or_B)
        return


    def g_eff_Map(self,E_or_B='E',step=1e-7):
        if E_or_B=='E':
            if self.evecs_E is None:
                print('Run StarkMap first')
                return None
            g_eff_vs_E = []
            for i,evecs in enumerate(self.evecs_E):
                g_eff = self.g_eff_evecs(self.evals_E[i],evecs,self.Ez[i],self._Bz,step=step)
                g_eff_vs_E.append(g_eff)
            g_eff_vs_E = np.array(g_eff_vs_E)
            self.g_eff_E = g_eff_vs_E
            return g_eff_vs_E
        else:
            if self.evecs_B is None:
                print('Run ZeemanMap first')
                return None
            g_eff_vs_B = []
            for i,evecs in enumerate(self.evecs_B):
                g_eff = self.g_eff_evecs(self.evals_B[i],evecs,self._Ez,self.Bz[i],step=step)
                g_eff_vs_B.append(g_eff)
            g_eff_vs_B = np.array(g_eff_vs_B)
            self.g_eff_B = g_eff_vs_B
            return g_eff_vs_B


    def plot_PTV(self,E_or_B='E',fontsize=14, plot_title = True, kV_kG=False):

        if self.PTV_E is None and E_or_B=='E':
            print('Need to run PTV_Map first')
            return
        if self.PTV_B is None and E_or_B == 'B':
            print('Need to run PTV_BMap first')
            return

        x_scale = {False: 1, True: 10**-3}[kV_kG]

        x_label = {
            True: {'E': 'Electric field (kV/cm)', 'B': 'Magnetic field (kG)'}[E_or_B],
            False: {'E': 'Electric field (V/cm)', 'B': 'Magnetic field (G)'}[E_or_B]
        }[kV_kG]

        field,shifts = {'E': [self.Ez,self.PTV_E], 'B':[self.Bz,self.PTV_B]}[E_or_B]
        shifts = shifts.T # change primary index from E field to eigenvector

        y_label = {
            #'EDM': r'$\langle \Sigma \rangle$',
            #'MQM': r'$\langle S \cdot T \cdot n \rangle$'
            'EDM': r'eEDM sensitivity',
            'NSM': r'NSM sensitivity',
            'MQM': r'NMQM sensitivity',
            'NSDPV': r'NSDPV sensitivity',
        }[self.PTV_type]

        state_str =  self.state_str

        PTV_str = {
            'EDM': 'EDM Shifts',
            'NSM': 'NSM Shifts',
            'MQM': 'MQM Shifts',
            'NSDPV': 'NSDPV Shifts',
        }[self.PTV_type]


        title = state_str + ' ' + PTV_str + r', $N={}$'.format(str(self.N_range)[1:-1])

        plt.figure(figsize=(10,7))
        for trace in shifts:
            plt.plot(x_scale*field,trace)
        plt.xlabel(x_label,fontsize=fontsize)
        plt.ylabel(y_label,fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if plot_title:
            plt.title(title,fontsize=fontsize)
        return

    def filter_evecs(self,evecs,q_filter,filter_vals):
        idx_filtered = []
        for i,evec in enumerate(evecs):
            max_idx = np.argmax(evec**2)
            for val in filter_vals:
                if self.q_numbers[q_filter][max_idx]==val:
                    idx_filtered.append(i)
        idx_filtered=np.array(idx_filtered)
        return idx_filtered

    def display_levels(self,Ez,Bz,pattern_q,idx = None,label=True,label_off = 0.03,parity=False,label_q =None,width=0.75,figsize=(10,10),ylim=None, deltaE_label = 3000,alt_label=False):
        if label_q is None:
            label_q = self.q_str
        if 'M' not in label_q:
            label_q.append('M')
        if pattern_q not in label_q:
            label_q.append(pattern_q)
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=False)
        parities = np.array(self.parities)
        if idx is not None:
            evals = evals[idx]
            evecs = evecs[idx]
            parities = np.array(self.parities)[idx]
        if ylim is None:
            scale = abs(evals[-1]-evals[0])
            ylim = (evals[0]-0.1*scale,evals[-1]+0.1*scale)
        else:
            scale = abs(ylim[1]-ylim[0])
        fig = plt.figure(figsize=figsize)
        primary_q = {q:[] for q in self.q_str}
        M_bounds = []
        for evec in evecs:
            for q in label_q:
                max_q_val = self.q_numbers[q][np.argmax(evec**2)]
                primary_q[q].append(max_q_val)
                if q=='M':
                    M_bounds.append([(max_q_val-width/2),(max_q_val+width/2)])
        M_bounds = np.array(M_bounds).T
        plt.hlines(evals,*M_bounds,colors='b')
        plt.ylim(ylim)
    #     left,right = plt.xlim()
    #     plt.xlim(left-1,right+2)
        if label:
            off = label_off
            sign = 1
            label_q_no_M = [q for q in label_q if q!='M']
            prev_pattern = None
            prev_energy = 0
            labeled_energies = []
            for i,energy in enumerate(evals):
                current_pattern = primary_q[pattern_q][i]
                if (prev_pattern != current_pattern or (prev_pattern == current_pattern and abs(energy-prev_energy)>deltaE_label)) and (ylim[0]+scale*off < energy < ylim[1]-scale*off):
                    if energy not in labeled_energies:
                        label_str = r'$|$'
                        if parity:
                            label_str+='${}$,'.format({1: '+', -1:'-'}[parities[i]])
                        for q in label_q_no_M:
                            label_str+=r'${}$={},'.format(q,primary_q[q][i])
                        label_str = label_str[:-1]+r'$\rangle$'
                        if self.M_values=='custom':
                            loc = primary_q['M'][i]
                        elif self.M_values=='positive':
                            loc = abs(primary_q['M'][i] % 1)
                        else:
                            loc = 0
                        plt.annotate(label_str,(loc,energy-sign*off*scale),ha='center',fontsize=10)
                        labeled_energies.append(energy)
                        if alt_label: # alternate labels
                            sign*=-1
                prev_energy = energy
                prev_pattern = current_pattern
            bot,top = plt.ylim()
            plt.ylim(bot-0.03*scale,top+0.03*scale)
        plt.xlabel(r'$M_F$',fontsize=14)
        plt.ylabel('Energy (MHz)',fontsize=14)

        return

    def display_PTV(self, Ez,Bz,EDM_or_MQM,step_B = 1e-4, step_E = 1e-2, idx = None,width=0.75,figsize=(9,9), fontsize_for_values = 8,fontsize = 14, ylim=None,round=None, off = 0, scale2 = 0.03, plot_state_index = False, plot_only_one_CPV = False, plot_CPV = True):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            else:
                self.PTV_type = EDM_or_MQM
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
                
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
        

        g_effs = []
        Orientations = []
        for i in range(len(evals)):
            Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
            g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            
        Orientations = np.array(Orientations)
        g_effs = np.array(g_effs)
        
        if idx is not None:
            evals = evals[idx]
            evecs = evecs[idx]
            Orientations = Orientations[idx]
            g_effs = g_effs[idx]    
        
        
       
        if EDM_or_MQM == 'all':
            EDM_shifts = []
            NSM_shifts = []
            for evec0 in evecs:
                
                E_PTV_EDM = evec0@H_PTV_EDM@evec0 #should I complex conjugate?
                EDM_shifts.append(np.round(E_PTV_EDM,self.round))  
                
                E_PTV_NSM = evec0@H_PTV_NSM@evec0 #should I complex conjugate?
                NSM_shifts.append(np.round(E_PTV_NSM,self.round))   
            
            EDM_shifts = np.array(EDM_shifts)
            NSM_shifts = np.array(NSM_shifts)
                
        PTV_shifts = []   
        
        for evec0 in evecs:
            E_PTV = evec0@H_PTV@evec0 #should I complex conjugate?
            PTV_shifts.append(np.round(E_PTV,self.round))            
        
        
        PTV_shifts = np.array(PTV_shifts)

        self.PTV0 = PTV_shifts
            
        if ylim is None:
            scale = abs(evals[-1]-evals[0])
            ylim = (evals[0]-0.1*scale,evals[-1]+0.1*scale)
        else:
            scale = abs(ylim[1]-ylim[0])
        fig = plt.figure(figsize=figsize)
        plt.ylim(ylim)
        M_bounds = []
        M_vals = []
        for evec in evecs:
            M = self.q_numbers['M'][np.argmax(evec**2)]
            M_vals.append(M)
            M_bounds.append([(M-width/2),(M+width/2)])
        M_bounds = np.array(M_bounds).T
        plt.hlines(evals+off,*M_bounds)
       
        for i,shift in enumerate(PTV_shifts):
            if (ylim[0]+scale*off < evals[i] < ylim[1]-scale*off):
                orientation = Orientations[i]
                g_eff = g_effs[i]
                
                if EDM_or_MQM == 'all':
                    shift_EDM = EDM_shifts[i]
                    shift_NSM = NSM_shifts[i]
                    
                if round is not None:
                    if EDM_or_MQM == 'all':
                        shift_EDM = np.round(shift_EDM,round)
                        shift_NSM = np.round(shift_NSM,round)
                    shift = np.round(shift,round)
                    orientation = np.round(orientation,round)
                    g_eff = np.round(g_eff,round)


                #orientation = np.round(self.g_eff_EB(Ez,Bz,step=step)[i], round)
                #g_eff = np.round(self.dipole_EB(Ez,Bz,step=step)[i], round)
                
                if plot_only_one_CPV is False:
                    plt.annotate(g_eff,(M_vals[i],evals[i]+off+2*scale2),ha='center',fontsize=12)
                    plt.annotate(orientation,(M_vals[i],evals[i]+off+scale2),ha='center',fontsize=12)
                
                if plot_state_index:
                    plt.annotate(i,(M_vals[i],evals[i]+off+3*scale2),ha='center',fontsize=fontsize_for_values)
                    
                if plot_CPV:
                    plt.annotate(shift,(M_vals[i],evals[i]+off-scale2),ha='center',fontsize=fontsize_for_values)
                    
                if (EDM_or_MQM == 'all') and (plot_only_one_CPV is False):
                    plt.annotate(shift_NSM,(M_vals[i],evals[i]+off-2*scale2),ha='center',fontsize=12)
                    plt.annotate(shift_EDM,(M_vals[i],evals[i]+off-3*scale2),ha='center',fontsize=12)
                
        
        bot,top = plt.ylim()
        
        if EDM_or_MQM == 'all'and (plot_only_one_CPV is False):
            plt.annotate('top1: B shift\ntop2: E shift\nbottom1: MQM shift\nbottom2: NSM shift\nbottom3: EDM shift',(max(M_vals)-0.5,top-0.5),fontsize=12)
        elif  plot_only_one_CPV is False:
            plt.annotate('top1: B shift\ntop2: E shift\nbottom: PTV shift',(max(M_vals)-0.5,top-0.5),fontsize=12)
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)        
        plt.ylim(bot-0.3,top+0.3)
        plt.xlabel(r'$M_F$',fontsize=fontsize)
        plt.ylabel('Energy (MHz)',fontsize=fontsize)
        
        return



    def display_PTV_old(self,Ezs, Ez,Bz,EDM_or_MQM,step = 1e-5, idx = None,width=0.75,figsize=(9,9),ylim=None,round=None,Bzs = np.linspace(0,0.01,1001),g_eff_scale = 10000, orientation_scale = 10, off = 0, scale2 = 0.03):
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            if EDM_or_MQM == 'all':
                H_PTV_EDM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'EDM')
                H_PTV_NSM = self.library.PTV_builders[self.iso_state](self.q_numbers, 'NSM')
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, 'MQM')
            else:
                self.PTV_type = EDM_or_MQM
                H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
        
        B_close_index = 1     
        E_close_index = (np.abs(Ezs - Ez)).argmin()
        g_effs = []
        Orientations = []
        for i in range(len(evals)):
            Orientations.append(np.gradient(self.evals_E.T[i])[E_close_index]/(0.1*self.parameters['muE']))
            g_effs.append(np.gradient(self.evals_B.T[i])[B_close_index]/(1e-5*self.parameters['mu_B']))
            
        Orientations = np.array(Orientations)
        g_effs = np.array(g_effs)
        
        if idx is not None:
            evals = evals[idx]
            evecs = evecs[idx]
            Orientations = Orientations[idx]
            g_effs = g_effs[idx]    
        
        
       
        if EDM_or_MQM == 'all':
            EDM_shifts = []
            NSM_shifts = []
            for evec0 in evecs:
                
                E_PTV_EDM = evec0@H_PTV_EDM@evec0 #should I complex conjugate?
                EDM_shifts.append(np.round(E_PTV_EDM,self.round))  
                
                E_PTV_NSM = evec0@H_PTV_NSM@evec0 #should I complex conjugate?
                NSM_shifts.append(np.round(E_PTV_NSM,self.round))   
            
            EDM_shifts = np.array(EDM_shifts)
            NSM_shifts = np.array(NSM_shifts)
                
        PTV_shifts = []   
        
        for evec0 in evecs:
            E_PTV = evec0@H_PTV@evec0 #should I complex conjugate?
            PTV_shifts.append(np.round(E_PTV,self.round))            
        
        
        PTV_shifts = np.array(PTV_shifts)

        self.PTV0 = PTV_shifts
            
        if ylim is None:
            scale = abs(evals[-1]-evals[0])
            ylim = (evals[0]-0.1*scale,evals[-1]+0.1*scale)
        else:
            scale = abs(ylim[1]-ylim[0])
        fig = plt.figure(figsize=figsize)
        plt.ylim(ylim)
        M_bounds = []
        M_vals = []
        for evec in evecs:
            M = self.q_numbers['M'][np.argmax(evec**2)]
            M_vals.append(M)
            M_bounds.append([(M-width/2),(M+width/2)])
        M_bounds = np.array(M_bounds).T
        plt.hlines(evals,*M_bounds)
       
        for i,shift in enumerate(PTV_shifts):
            if (ylim[0]+scale*off < evals[i] < ylim[1]-scale*off):
                orientation = orientation_scale*Orientations[i]
                g_eff = g_eff_scale*g_effs[i]
                
                if EDM_or_MQM == 'all':
                    shift_EDM = EDM_shifts[i]
                    shift_NSM = NSM_shifts[i]
                    
                if round is not None:
                    if EDM_or_MQM == 'all':
                        shift_EDM = np.round(shift_EDM,round)
                        shift_NSM = np.round(shift_NSM,round)
                    shift = np.round(shift,round)
                    orientation = np.round(orientation,round)
                    g_eff = np.round(g_eff,round)
               

                #orientation = np.round(self.g_eff_EB(Ez,Bz,step=step)[i], round)
                #g_eff = np.round(self.dipole_EB(Ez,Bz,step=step)[i], round)
            
                plt.annotate(g_eff,(M_vals[i],evals[i]+off+2*scale2),ha='center',fontsize=12)
                plt.annotate(orientation,(M_vals[i],evals[i]+off+scale2),ha='center',fontsize=12)
                
                plt.annotate(shift,(M_vals[i],evals[i]+off-scale2),ha='center',fontsize=12)
                if EDM_or_MQM == 'all':
                    plt.annotate(shift_NSM,(M_vals[i],evals[i]+off-2*scale2),ha='center',fontsize=12)
                    plt.annotate(shift_EDM,(M_vals[i],evals[i]+off-3*scale2),ha='center',fontsize=12)
                
        bot,top = plt.ylim()
        
        if EDM_or_MQM == 'all':
            plt.annotate('top1: B shift\ntop2: E shift\nbottom1: MQM shift\nbottom2: NSM shift\nbottom3: EDM shift',(max(M_vals)-0.5,top-0.5),fontsize=12)
        else:
            plt.annotate('top1: B shift\ntop2: E shift\nbottom: PTV shift',(max(M_vals)-0.5,top-0.5),fontsize=12)
            
        plt.ylim(bot-0.3,top+0.3)
        plt.xlabel(r'$M_F$',fontsize=14)
        plt.ylabel('Energy (MHz)',fontsize=14)
        return
#        return self.g_eff_EB(Ez,Bz,step=step)
    
    
    def parity_EB(self,Ez,Bz,round=0):
        evals,evecs = self.eigensystem(Ez,Bz)
        parity_list = []
        for evec in evecs:
            P = np.round(evec@self.Parity_mat@evec,round)
            parity_list.append(P)
        return np.array(parity_list)

    def display_parity(self,Ez,Bz,round = 2,**kwargs):
        return self.display_property(Ez,Bz,self.parity_EB(Ez,Bz,round=round),**kwargs)

    def display_g_eff(self,Ez,Bz,step=1e-5,**kwargs):
        return self.display_property(Ez,Bz,self.g_eff_EB(Ez,Bz,step=step),**kwargs)

    def display_property(self,Ez,Bz,properties,idx = None,width=0.75,figsize=(9,9),ylim=None,round=None):
        if Ez==self.E0 and Bz==self.B0:
            evals,evecs = [self.evals0,self.evecs0]
        else:
            evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
        if idx is not None:
            evals = evals[idx]
            evecs = evecs[idx]
            properties = properties[idx]
        if ylim is None:
            scale = abs(evals[-1]-evals[0])
            ylim = (evals[0]-0.1*scale,evals[-1]+0.1*scale)
        else:
            scale = abs(ylim[1]-ylim[0])
        fig = plt.figure(figsize=figsize)
        plt.ylim(ylim)
        M_bounds = []
        M_vals = []
        for evec in evecs:
            M = self.q_numbers['M'][np.argmax(evec**2)]
            M_vals.append(M)
            M_bounds.append([(M-width/2),(M+width/2)])
        M_bounds = np.array(M_bounds).T
        plt.hlines(evals,*M_bounds)
        off = 0.03
        for i,prop in enumerate(properties):
            if (ylim[0]+scale*off < evals[i] < ylim[1]-scale*off):
                if round is not None:
                    prop = np.round(prop,round)
                plt.annotate(prop,(M_vals[i],evals[i]-off*scale),ha='center',fontsize=12)
        bot,top = plt.ylim()
        plt.ylim(bot-0.3,top+0.3)
        plt.xlabel(r'$M_F$',fontsize=14)
        plt.ylabel('Energy (MHz)',fontsize=14)
        return properties
#        return

    def convert_evecs(self,basis,evecs=None,Normalize=True,verbose=True):
        if evecs is None:
            evecs = self.evecs0
        current_case = self.hunds_case
        new_case = basis
        if new_case == current_case:
            print('Eigenvectors already in {} basis'.format(current_case))
            return evecs
        inputt = self.q_numbers
        output = self.alt_q_numbers[new_case]
        if ('a' in new_case and 'bBJ' in current_case) or ('bBJ' in new_case and 'a' in current_case):
            basis_matrix = self.library.basis_changers['a_bBJ'](inputt,output)
        elif ('decoupled' in new_case and 'b' in current_case):
            basis_matrix = self.library.basis_changers['b_decoupled'](inputt,output)
        elif ('decoupled' in new_case and 'a' in current_case and 'J' not in new_case):
            intermediate = self.alt_q_numbers['bBJ']
            basis_matrix = self.library.basis_changers['b_decoupled'](intermediate,output)@self.library.basis_changers['a_bBJ'](inputt,intermediate)
        elif ('recoupled' in new_case and 'a' in current_case):
            intermediate1 = self.alt_q_numbers['bBJ']
            intermediate2 = self.alt_q_numbers['decoupled']
            basis_matrix = self.library.basis_changers['recouple_J'](intermediate2,output)@self.library.basis_changers['b_decoupled'](intermediate1,intermediate2)@self.library.basis_changers['a_bBJ'](inputt,intermediate1)
        elif ('decouple_I' in new_case and 'b' in current_case):
            basis_matrix = self.library.basis_changers['b_I_decoupled'](inputt,output)
        elif ('a' in new_case and 'bBS' in current_case) or ('bBS' in new_case and 'a' in current_case):
            intermediate = self.alt_q_numbers['bBJ']
            if 'bBS' in current_case:
                basis_matrix = self.library.basis_changers['a_bBJ'](intermediate,output)@self.library.basis_changers['bBS_bBJ'](inputt,intermediate)
            else:
                basis_matrix = self.library.basis_changers['bBS_bBJ'](inputt,intermediate)@self.library.basis_changers['a_bBJ'](intermediate,output)
        elif ('bBS' in new_case and 'bBJ' in current_case) or ('bBJ' in new_case and 'bBS' in current_case):
            basis_matrix = self.library.basis_changers['bBS_bBJ'](inputt,output)
        converted_evecs = []
        for i in range(len(evecs)):
            converted_evecs.append(basis_matrix@evecs[i])
        converted_evecs = np.array(converted_evecs)
        if Normalize:
            for i,evec in enumerate(converted_evecs):
                converted_evecs[i]/=evec@evec
        if verbose:
            print('Successfully converted eigenvectors from {} to {}'.format(current_case,new_case))
        return converted_evecs

    def gen_state_str(self,vector_idx,evecs=None,basis=None,label_q=None,parity=False,single=False,thresh=0.01,show_coeff=True,new_line=False,round=None,frac=''):
        q_numbers = self.q_numbers
        if label_q == None:
            label_q = self.q_str
        if round is None:
            round=self.round
        if evecs is None:
            evecs = self.evecs0
        if basis is not None:
            if basis in self.hunds_case:
                pass
            elif 'decoupled' in basis:
                q_numbers = self.alt_q_numbers['decoupled']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['decoupled'])
                evecs = self.convert_evecs('decoupled',evecs=evecs,verbose=False)
            elif 'a' in basis:
                q_numbers = self.alt_q_numbers['aBJ']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['aBJ'])
                evecs = self.convert_evecs('aBJ',evecs=evecs,verbose=False)
            elif 'bBJ' in basis:
                q_numbers = self.alt_q_numbers['bBJ']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['bBJ'])
                evecs = self.convert_evecs('bBJ',evecs=evecs,verbose=False)
            elif 'recouple' in basis:
                q_numbers = self.alt_q_numbers['recouple_J']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['recouple_J'])
            elif 'decouple_I' in basis:
                q_numbers = self.alt_q_numbers['decouple_I']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['decouple_I'])
            elif 'bBS' in basis:
                q_numbers = self.alt_q_numbers['bBS']
                if label_q == None:
                    label_q = list(self.alt_q_numbers['bBS'])
                evecs = self.convert_evecs('bBS',evecs=evecs,verbose=False)
        full_label = r''
        vector = deepcopy(evecs[vector_idx])
        vector[abs(vector)<thresh] = 0
        nonzero_idx = np.nonzero(vector)[0]
        if single:
            max_idx = abs(vector).argmax()
            nonzero_idx = [max_idx]
        i=0
        for i,index in enumerate(nonzero_idx):
            coeff = np.round(vector[index],round)
            sign = {True: '+', False: '-'}[coeff > 0]
            sign0 = {True: ' ', False: sign}[sign=='+']
            if show_coeff:
                label_str = r'$\,{}\,{}|'.format({True: sign0, False: sign}[i==0],abs(coeff))
            else:
                if single:
                    label_str = r'$|'
                else:
                    label_str = r'$\,{}|'.format({True: sign0, False: sign}[i==0])
                    if i==0:
                        label_str = r'$'+label_str[3:]
            if new_line:
                label_str = r'$'+label_str
            val = {q:q_numbers[q][index] for q in label_q}
            if parity:
                label_str+= r'{},'.format({1:'+',-1:'-'}[self.parities[vector_idx]])
            for q in label_q:
                _q = {True: '\u039B', False: q}[q=='L']
                _q = {True: '\u03A3', False: q}[q=='Sigma']
                _q = {True: '\u03A9', False: q}[q=='Omega']
                if (abs(val[q]) % 1) !=0:
                    label_str+=r'{}=\{}frac{{{}}}{{{}}},'.format(_q,frac,*val[q].as_integer_ratio())
                else:
                    label_str+=r'{}={},'.format(_q,int(val[q]))
            label_str = label_str[:-1]+r'\rangle\,$'
            if new_line:
                label_str+=r'$'
            full_label+=label_str
        return full_label

    def select_q(self,q_dict,evecs=None,parity=None):
        if evecs == None:
            evecs = self.evecs0
        idx = []
        for i in range(len(evecs)):
            match = True
            for q_string in q_dict:
                if not isinstance(q_dict[q_string],list):
                    q_dict[q_string] = [q_dict[q_string]]
                if self.q_numbers[q_string][np.argmax(evecs[i]**2)] in q_dict[q_string]:
                    match*=True
                else:
                    match*=False
                if parity is not None:
                    if parity=='+' and self.parities[i]<0:
                        match*=False
                    if parity=='-' and self.parities[i]>0:
                        match*=False
            if match==True:
                idx.append(i)
        return np.array(idx)

    def EB_grid(self,Ez,Bz,E_or_B_first = 'E',reverse=False,interp=False,method='torch',output=False,evecs=False,PTV=False,trap_shifts=False,order_states=True,EDM_or_MQM='EDM'):
        self.eigensystem(Ez[0],Bz[0])
        N_evals = len(self.evals0)
        evec_dim = len(self.evecs0[0])
        if self.trap == False:
            trap_shifts=False
        N_Bz = len(Bz)
        N_Ez = len(Ez)
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' in self.iso_state or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)
        if trap_shifts:
            H_trap= self.trap_matrix()
        evals_EB = np.zeros((N_evals, N_Ez, N_Bz))
        if evecs:
            evecs_EB = np.zeros((N_evals, N_Ez, N_Bz,evec_dim))
        if PTV:
            PTV_EB = np.zeros((N_evals, N_Ez, N_Bz))
        if trap_shifts:
            shifts_EB = np.zeros((N_evals,N_Ez,N_Bz))
        if reverse:
            Ez = Ez[::-1]
            Bz = Bz[::-1]
        if E_or_B_first == 'E':
            evals_E,evecs_E = self.StarkMap(Ez,Bz[0],output=True,write_attribute=False,method=method,order=order_states)
            for i in range(N_Ez):
                evals_B, evecs_B = self.ZeemanMap(Bz, Ez_val = Ez[i],output=True,write_attribute=False,method=method,initial_evecs=evecs_E[i],order=order_states)
                for j in range(N_Bz):
                    evals_EB[:,i,j] = evals_B[j]
                    if evecs:
                        evecs_EB[:,i,j,:] = evecs_B[j]
                    if PTV:
                        PTV_EB[:,i,j] = np.round(np.diagonal(evecs_B[j]@H_PTV@evecs_B[j].T),self.round)
                        #I think it has to be evecs.T from trial and error
                    if trap_shifts:
                        shifts_EB[:,i,j] = np.round(np.diagonal(evecs_B[j]@H_trap@evecs_B[j].T),self.round)
        else:
            evals_B,evecs_B = self.ZeemanMap(Bz,Ez[0],output=True,write_attribute=False,method=method,order=order_states)
            for i in range(N_Bz):
                evals_E, evecs_E = self.StarkMap(Ez, Bz_val = Bz[i], output=True,write_attribute=False,method=method,initial_evecs = evecs_B[i],order=order_states)
                for j in range(N_Ez):
                    evals_EB[:,j,i] = evals_E[j]
                    if evecs:
                        evecs_EB[:,j,i,:] = evecs_E[j]
                    if PTV:
                        PTV_EB[:,j,i] = np.round(np.diagonal(evecs_E[j]@H_PTV@evecs_E[j].T),self.round)
                    if trap_shifts:
                        shifts_EB[:,j,i] = np.round(np.diagonal(evecs_E[j]@H_trap@evecs_E[j].T),self.round)
        if reverse:
            evals_EB = np.flip(evals_EB,(1,2))
            order = np.argsort(evals_EB[:,0,0])
            print(order)
            evals_EB = evals_EB[order,:,:]
        self.evals_EB = evals_EB
        if evecs:
            if reverse:
                evecs_EB = np.flip(evecs_EB,(1,2))[order,:,:,:]
            self.evecs_EB = evecs_EB
        if PTV:
            if reverse:
                PTV_EB = np.flip(PTV_EB, (1,2))[order,:,:]
            self.PTV_EB = PTV_EB
        if trap_shifts:
            if reverse:
                shifts_EB = np.flip(shifts_EB, (1,2))[order,:,:]
            self.shifts_EB = shifts_EB
        if output and not (evecs or PTV):
            result = evals_EB
            return [result]
        elif output and (evecs or PTV or trap_shifts):
            result = [evals_EB]
            if PTV:
                result.append(PTV_EB)
            if evecs:
                result.append(evecs_EB)
            if trap_shifts:
                result.append(shifts_EB)
            return result
        else:
            return


    def generate_parities(self,evecs=None,ret=True):
        if evecs is None:
            evals,evecs = self.eigensystem(0,1e-6,set_attr=False)
        P_matrix = self.Parity_mat
        parities = []
        for evec in evecs:
            parity = np.round(evec@P_matrix@evec,0)
            parities.append(parity)
        self.parities = parities
        if ret:
            return parities
        return
    
    

    def find_antimagic(self,Efield_array,Bz, EDM_or_MQM, B_criteria = 10, E_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        BE_magic_index_all = []
        PTV_shifts_magic_all = []    
        
        found_magic_index = []
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            BE_magic_index = []
            PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))

            for i in range(len(B_combis)):
                if (abs(B_combis[i][0] - B_combis[i][1]) < B_criteria) and (abs(E_combis[i][0] - E_combis[i][1]) > E_criteria):                    

                    evec0 = evecs[index_combis[i][0]]
                    evec1 = evecs[index_combis[i][1]]

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1
                    
                    M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                    M1 = self.q_numbers['M'][np.argmax(evec1**2)]

                    F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                    F1 = self.q_numbers['F'][np.argmax(evec1**2)] 
                    
                    if '174' in self.iso_state:
                        J0 = self.q_numbers['J'][np.argmax(evec0**2)]
                        J1 = self.q_numbers['J'][np.argmax(evec1**2)]
                    
                    if '174' not in self.iso_state:
                        F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                        F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                        G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                        G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                    if (abs(E_PTV_0 - E_PTV_1) < CPV_criteria) and (abs(M0 - M1) <= M_criteria):
                        
                        
                        if i not in found_magic_index:
                            found_magic_index.append(i)
                        
                        BE_magic_index.append([index_combis[i],Ez])
                        

                    
                        if round is not None:
                            E_PTV_0 = np.round( E_PTV_0, round)
                            E_PTV_1 = np.round( E_PTV_1, round)

                        PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                        print('state index: ',index_combis[i], i)
                        print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                        if '174' in self.iso_state:
                            print('J: ', J0, J1 )
                        if '174' not in self.iso_state:
                            print('G: ', G0, G1 )
                            print('F1: ', F10, F11 )
                        print('F: ', F0, F1 )
                        print('M_F: ', M0, M1 )
                        print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ',np.round( B_combis[i][0] - B_combis[i][1], round))
                        print('polarization: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                        print('PTV shifts: ', E_PTV_0, E_PTV_1, ' difference: ', np.round(E_PTV_0 - E_PTV_1,round))  
                        print(' ')


            BE_magic_index = np.array(BE_magic_index)
            PTV_shifts_magic = np.array(PTV_shifts_magic)
            
            BE_magic_index_all.append(BE_magic_index)
            PTV_shifts_magic_all.append(PTV_shifts_magic)
            
            
        BE_magic_index_all = np.array(BE_magic_index_all)
        PTV_shifts_magic_all = np.array(PTV_shifts_magic_all)
        
        found_magic_index = np.array(found_magic_index)
        
        print('# of identified all insensitive transitions',len(found_magic_index))
        
        return found_magic_index, BE_magic_index_all, PTV_shifts_magic_all
    
         

    def find_B_zero_crossing_old(self,Efield_array,Bz, EDM_or_MQM,  E_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, F1_same_as_previous_check = False, F_same_as_previous_check = False):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        #BE_magic_index_all = []
        #PTV_shifts_magic_all = []
        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        F1_is_same = True
        F_is_same = True
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            
            if Is_this_first_Efield:
                previous = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):
                
                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 
                
                if (Is_this_first_Efield is False) and (np.sign(B_combis[i][0] - B_combis[i][1]) != np.sign(previous[i])) and (abs(E_combis[i][0] - E_combis[i][1]) < E_criteria):                    

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1


                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]):
                                                
                        if (F10 != previous_F1s[i][0]) or (F11 != previous_F1s[i][1]):
                            print('Caution! F1 value is not the same as the previous F1 value! F1 now: ', F10, F11)
                            print('F1 previous: ', previous_F1s[i][0], previous_F1s[i][1])
                            print('')
                            
                            if F1_same_as_previous_check:
                                F1_is_same = False
                            
                        if (F0 != previous_Fs[i][0]) or (F1 != previous_Fs[i][1]):
                            print('Caution! F value is not the same as the previous F value! F now: ', F0, F1)
                            print('F previous: ', previous_Fs[i][0], previous_Fs[i][1])
                            print('')

                            if F_same_as_previous_check:
                                F_is_same = False
                                
                        if (F1_is_same is False) or (F_is_same is False):
                            F1_is_same = True
                            F_is_same = True
                        
                        else:
                            B_zero_crossing.append([Ez, index_combis[i], np.round(previous[i], round), np.round( B_combis[i][0] - B_combis[i][1], round)])



                            if round is not None:
                                E_PTV_0 = np.round( E_PTV_0, round)
                                E_PTV_1 = np.round( E_PTV_1, round)


                            #PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                            print('state index: ',index_combis[i], i)
                            print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                            print('G: ', G0, G1 )
                            print('F1: ', F10, F11 )
                            print('F: ', F0, F1 )
                            print('M_F: ', M0, M1 )
                            print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                            print('g factor difference previous: ',np.round( previous[i], round))
                            print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                            print('PTV shifts (a factor of two): ', 2*E_PTV_0, 2*E_PTV_1, ' difference: ', np.round(2*(E_PTV_0 - E_PTV_1),round))  
                            print(' ')
                        
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                previous_F1s[i][0] = F10
                previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous[i] = B_combis[i][0] - B_combis[i][1]
           
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            #BE_magic_index = np.array(BE_magic_index)
            B_zero_crossing = np.array(B_zero_crossing)
            
            #BE_magic_index_all.append(BE_magic_index)
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all
        
    def find_E_zero_crossing_old(self,Efield_array,Bz, EDM_or_MQM,  B_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None, F1_same_as_previous_check = False, F_same_as_previous_check = False):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        #BE_magic_index_all = []
        #PTV_shifts_magic_all = []
        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        F1_is_same = True
        F_is_same = True
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            
            if Is_this_first_Efield:
                previous = np.zeros(len(B_combis))
                previous_Ms = np.zeros((len(B_combis),2))
                previous_F1s = np.zeros((len(B_combis),2))
                previous_Fs = np.zeros((len(B_combis),2))

            for i in range(len(B_combis)):
                
                evec0 = evecs[index_combis[i][0]]
                evec1 = evecs[index_combis[i][1]]

                M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                
                G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                G1 = self.q_numbers['G'][np.argmax(evec1**2)]

                F10 = self.q_numbers['F1'][np.argmax(evec0**2)]
                F11 = self.q_numbers['F1'][np.argmax(evec1**2)]

                F0 = self.q_numbers['F'][np.argmax(evec0**2)]
                F1 = self.q_numbers['F'][np.argmax(evec1**2)]                 
                
                if (Is_this_first_Efield is False) and (np.sign(E_combis[i][0] - E_combis[i][1]) != np.sign(previous[i])) and (abs(B_combis[i][0] - B_combis[i][1]) < B_criteria):                    

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1


                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]):
                                                
                        if (F10 != previous_F1s[i][0]) or (F11 != previous_F1s[i][1]):
                            print('Caution! F1 value is not the same as the previous F1 value! F1 now: ', F10, F11)
                            print('F1 previous: ', previous_F1s[i][0], previous_F1s[i][1])
                            print('')
                            
                            if F1_same_as_previous_check:
                                F1_is_same = False
                            
                        if (F0 != previous_Fs[i][0]) or (F1 != previous_Fs[i][1]):
                            print('Caution! F value is not the same as the previous F value! F now: ', F0, F1)
                            print('F previous: ', previous_Fs[i][0], previous_Fs[i][1])
                            print('')

                            if F_same_as_previous_check:
                                F_is_same = False
                                
                        if (F1_is_same is False) or (F_is_same is False):
                            F1_is_same = True
                            F_is_same = True
                        
                        else:
                            B_zero_crossing.append([Ez, index_combis[i], np.round(previous[i], round), np.round( E_combis[i][0] - E_combis[i][1], round)])



                            if round is not None:
                                E_PTV_0 = np.round( E_PTV_0, round)
                                E_PTV_1 = np.round( E_PTV_1, round)


                            #PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                            print('state index: ',index_combis[i], i)
                            print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                            print('G: ', G0, G1 )
                            print('F1: ', F10, F11 )
                            print('F: ', F0, F1 )
                            print('M_F: ', M0, M1 )
                            print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                            print('dipole previous: ',np.round( previous[i], round))
                            print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                            print('PTV shifts (a factor of two): ', 2*E_PTV_0, 2*E_PTV_1, ' difference: ', np.round(2*(E_PTV_0 - E_PTV_1),round))  
                            print(' ')
                        
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                previous_F1s[i][0] = F10
                previous_F1s[i][1] = F11
                previous_Fs[i][0] = F0
                previous_Fs[i][1] = F1       
                previous[i] = E_combis[i][0] - E_combis[i][1]
           
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            #BE_magic_index = np.array(BE_magic_index)
            B_zero_crossing = np.array(B_zero_crossing)
            
            #BE_magic_index_all.append(BE_magic_index)
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all
                
    def find_E_zero_crossing_old_old(self,Efield_array,Bz, EDM_or_MQM,  B_criteria = 10, CPV_criteria = -1, M_criteria = 10, step_B = 1e-4, step_E = 1e-2, idx = None, round = None):
        
        if '174' in self.iso_state or '40' in self.iso_state:
            self.PTV_type = 'EDM'
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers)
        elif '173' or '171' in self.iso_state:
            self.PTV_type = EDM_or_MQM
            H_PTV = self.library.PTV_builders[self.iso_state](self.q_numbers, EDM_or_MQM)

            
            
        #BE_magic_index_all = []
        #PTV_shifts_magic_all = []
        B_zero_crossing_all = []
        
        Is_this_first_Efield = True
        
        for Ez in Efield_array:            
            
            print('E field (V/cm): ', Ez)
            print(' ')
            
            if Ez==self.E0 and Bz==self.B0:
                evals,evecs = [self.evals0,self.evecs0]
            else:
                evals,evecs = self.eigensystem(Ez,Bz, set_attr=True)
                
            if idx is not None:
                evals = evals[idx]
                evecs = evecs[idx]
                
            g_effs = []
            Orientations = []
                
            if idx is None:
                for i in range(len(evals)):
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])
            else:
                for i in idx:
                    Orientations.append(self.dipole_EB(Ez,Bz,step=step_E)[i])
                    g_effs.append(self.g_eff_EB(Ez,Bz,step=step_B)[i])          

            Orientations = np.array(Orientations)
            g_effs = np.array(g_effs)
            
            
            
            #BE_magic_index = []
            B_zero_crossing = []
            #PTV_shifts_magic = [] 


            index = range(len(g_effs))
            index_combis = np.array(list(it.combinations(index,2)))
            B_combis = np.array(list(it.combinations(g_effs,2)))
            E_combis = np.array(list(it.combinations(Orientations,2)))
            
            if Is_this_first_Efield:
                previous = np.zeros(len(E_combis))
                previous_Ms = np.zeros((len(E_combis),2))

            for i in range(len(E_combis)):
                
                if (Is_this_first_Efield is False) and (np.sign(E_combis[i][0] - E_combis[i][1]) != np.sign(previous[i])) and (abs(B_combis[i][0] - B_combis[i][1]) < B_criteria):                    

                    evec0 = evecs[index_combis[i][0]]
                    evec1 = evecs[index_combis[i][1]]

                    E_PTV_0 = evec0@H_PTV@evec0
                    E_PTV_1 = evec1@H_PTV@evec1
                    
                    M0 = self.q_numbers['M'][np.argmax(evec0**2)]
                    M1 = self.q_numbers['M'][np.argmax(evec1**2)]
                    


                    if (abs(E_PTV_0 - E_PTV_1) > CPV_criteria) and (abs(M0 - M1) <= M_criteria) and (M0 == previous_Ms[i][0]) and (M1 == previous_Ms[i][1]):
                        
                        G0 = self.q_numbers['G'][np.argmax(evec0**2)]
                        G1 = self.q_numbers['G'][np.argmax(evec1**2)]
                    
                        #BE_magic_index.append(index_combis[i])
                        B_zero_crossing.append([Ez, index_combis[i], np.round(previous[i], round), np.round( E_combis[i][0] - E_combis[i][1], round)])
                        

                    
                        if round is not None:
                            E_PTV_0 = np.round( E_PTV_0, round)
                            E_PTV_1 = np.round( E_PTV_1, round)
                            

                        #PTV_shifts_magic.append([E_PTV_0,E_PTV_1]) 

                        print('state index: ',index_combis[i], i)
                        print('energy: ', np.round(evals[index_combis[i][0]], round), np.round(evals[index_combis[i][1]], round))
                        print('G: ', G0, G1 )
                        print('M_F: ', M0, M1 )
                        print('g factor: ',np.round( B_combis[i][0], round),np.round( B_combis[i][1], round), ' difference: ', np.round( B_combis[i][0] - B_combis[i][1], round))
                        print('dipole: ',np.round( E_combis[i][0], round),np.round( E_combis[i][1], round), ' difference: ', np.round(E_combis[i][0] - E_combis[i][1], round))
                        print('dipole difference previous: ',np.round( previous[i], round))
                        print('PTV shifts (a factor of two): ', 2*E_PTV_0, 2*E_PTV_1, ' difference: ', np.round(2*(E_PTV_0 - E_PTV_1),round))  
                        print(' ')
                        
                        
                previous[i] = E_combis[i][0] - E_combis[i][1]
                previous_Ms[i][0] = M0
                previous_Ms[i][1] = M1
                
                
            if Is_this_first_Efield:
                Is_this_first_Efield = False

            #BE_magic_index = np.array(BE_magic_index)
            B_zero_crossing = np.array(B_zero_crossing)
            
            #BE_magic_index_all.append(BE_magic_index)
            B_zero_crossing_all.append(B_zero_crossing)
            
            
        #BE_magic_index_all = np.array(BE_magic_index_all)
        B_zero_crossing_all = np.array(B_zero_crossing_all)
        
        return B_zero_crossing_all



#here, ground = final, excited = initial
def branching_ratios(Ground, Excited,Ez, Bz, Normalize=False):
    G_evals,G_evecs = Ground.eigensystem(Ez,Bz)
    G_qn = Ground.q_numbers
    E_evals,E_evecs = Excited.eigensystem(Ez,Bz)
    E_qn = Excited.q_numbers
    if 'a' not in Ground.hunds_case:
        G_evecs = Ground.convert_evecs('aBJ',Normalize=Normalize)
        G_qn = Ground.alt_q_numbers['aBJ']
    if 'a' not in Excited.hunds_case:
        E_evecs = Excited.convert_evecs('aBJ',Normalize=Normalize)
        E_qn = Excited.alt_q_numbers['aBJ']
    TDM_matrix = Excited.library.TDM_builders[Excited.iso_state](E_qn,G_qn)
    BR_matrix = (G_evecs@TDM_matrix@E_evecs.T)**2
    return BR_matrix

def XA_branching_ratios(X,A,Ez,Bz,Normalize=False): # must be in case a
    A.eigensystem(Ez,Bz)
    X.eigensystem(Ez,Bz)
    X_evecs_a = X.convert_evecs('aBJ',Normalize=Normalize)
    TDM_matrix = A.library.TDM_builders[A.iso_state](A.q_numbers,X.alt_q_numbers['aBJ'])
    BR_matrix = (X_evecs_a@TDM_matrix@A.evecs0.T)**2
    return BR_matrix


#Here, ground = initial, excited = final
def Calculate_TDMs(p,Ground, Excited, Ez, Bz, q=[-1,0,1],Normalize=False):
    if type(q)!=list:
        q = [q]
    G_evals,G_evecs = Ground.eigensystem(Ez,Bz)
    G_qn = Ground.q_numbers
    E_evals,E_evecs = Excited.eigensystem(Ez,Bz)
    E_qn = Excited.q_numbers
    if 'a' not in Ground.hunds_case:
        G_evecs = Ground.convert_evecs('aBJ',Normalize=Normalize)
        G_qn = Ground.alt_q_numbers['aBJ']
    if 'a' not in Excited.hunds_case:
        E_evecs = Excited.convert_evecs('aBJ',Normalize=Normalize)
        E_qn = Excited.alt_q_numbers['aBJ']
    TDM_matrix = Excited.library.TDM_p_builders[Excited.iso_state](p,q,G_qn,E_qn)
    TDM_p = (E_evecs@TDM_matrix@G_evecs.T)
    return TDM_p

def Calculate_TDM_evecs(p,G_evecs,Ground, E_evecs,Excited, q=[-1,0,1],Normalize=False):
    G_qn = Ground.q_numbers
    E_qn = Excited.q_numbers
    if 'a' not in Ground.hunds_case:
        G_evecs = Ground.convert_evecs('aBJ',evecs = G_evecs,Normalize=Normalize)
        G_qn = Ground.alt_q_numbers['aBJ']
    if 'a' not in Excited.hunds_case:
        E_evecs = Excited.convert_evecs('aBJ',evecs = E_evecs,Normalize=Normalize)
        E_qn = Excited.alt_q_numbers['aBJ']
    TDM_matrix = Excited.library.TDM_p_builders[Excited.iso_state](p,q,G_qn,E_qn)
    TDM_p = (E_evecs@TDM_matrix@G_evecs.T)
    return TDM_p

def Calculate_forbidden_TDM_evecs(p,G_evecs,Ground, E_evecs,Excited, scale=1,Normalize=False):
    G_qn = Ground.q_numbers
    E_qn = Excited.q_numbers
    if 'a' not in Ground.hunds_case:
        G_evecs = Ground.convert_evecs('aBJ',evecs = G_evecs,Normalize=Normalize)
        G_qn = Ground.alt_q_numbers['aBJ']
    if 'a' not in Excited.hunds_case:
        E_evecs = Excited.convert_evecs('aBJ',evecs = E_evecs,Normalize=Normalize)
        E_qn = Excited.alt_q_numbers['aBJ']
    TDM_matrix = Excited.library.TDM_p_forbidden_builders[Excited.iso_state](p,[-1,1],G_qn,E_qn) + scale*Excited.library.TDM_p_forbidden_builders[Excited.iso_state](p,[0],G_qn,E_qn)
    TDM_p = (E_evecs@TDM_matrix@G_evecs.T)
    return TDM_p

def Calculate_forbidden_TDMs(p,Ground, Excited, Ez, Bz, scale=1,Normalize=False):
    G_evals,G_evecs = Ground.eigensystem(Ez,Bz)
    G_qn = Ground.q_numbers
    E_evals,E_evecs = Excited.eigensystem(Ez,Bz)
    E_qn = Excited.q_numbers
    if 'a' not in Ground.hunds_case:
        G_evecs = Ground.convert_evecs('aBJ',Normalize=Normalize)
        G_qn = Ground.alt_q_numbers['aBJ']
    if 'a' not in Excited.hunds_case:
        E_evecs = Excited.convert_evecs('aBJ',Normalize=Normalize)
        E_qn = Excited.alt_q_numbers['aBJ']
    TDM_matrix = 0*Excited.library.TDM_p_forbidden_builders[Excited.iso_state](p,[-1,1],G_qn,E_qn) + scale*Excited.library.TDM_p_forbidden_builders[Excited.iso_state](p,[0],G_qn,E_qn)
    TDM_p = (E_evecs@TDM_matrix@G_evecs.T)
    return TDM_p


def state_ordering(evecs_old,evecs_new,round=8):
    overlap = np.round(abs(evecs_old@evecs_new.T),round)     #Essentially a matrix of the fidelities: |<phi|psi>|
    #calculate trace distance
    # for o in overlap:
    #     for _o in o:
    #         if (_o>1):
    #             print('OVERLAP BIGGER THAN 1', _o)
    # dist = abs(1-overlap)/step**2
    # ordering = np.array([dist[i,:].argmin() for i in range(len(evecs_old))])  #python
    # ordering = np.argmin(dist,axis=1) #numpy
    ordering = np.argmax(overlap,axis=1) #numpy
    return ordering

def order_eig(evals,evecs):
    order = np.argsort(evals)
    evecs_ordered =evecs[order,:]
    evals_ordered = evals[order]
    return evals_ordered,evecs_ordered

def diagonalize_batch(matrix_array,method='torch',round=10):
    if method == 'numpy':
        w,v = npLA.eigh(matrix_array)
    elif method == 'scipy':
        w,v = sciLA.eigh(matrix_array)
    elif method == 'torch':
        if not TORCH_AVAILABLE:
            print("Warning: torch not available, using numpy for batch diagonalization")
            w,v = npLA.eigh(matrix_array)
        else:
            w,v = torch.linalg.eigh(torch.from_numpy(matrix_array))
            # Use detach().cpu().numpy() to safely convert tensor to numpy
            w = w.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
    evals_batch = np.round(w,round)
    evecs_batch= np.round(np.transpose(v,[0,2,1]),round)
    return evals_batch,evecs_batch


def diagonalize(matrix,method='torch',order=False, Normalize=False,round=10):
    if method == 'numpy':
        w,v = npLA.eigh(matrix)
    elif method == 'scipy':
        w,v = sciLA.eigh(matrix)
    elif method == 'torch':
        if not TORCH_AVAILABLE:
            print("Warning: torch not available, using numpy for diagonalization")
            w,v = npLA.eigh(matrix)
        else:
            w,v = torch.linalg.eigh(torch.from_numpy(matrix))
            # Use detach().cpu().numpy() to safely convert tensor to numpy
            w = w.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
    evals = np.round(w,round)
    evecs = np.round(v.T,round)
    # if Normalize:
    #     for i,evec in enumerate(evecs):
    #         evecs[i]/=evec@evec
    if order:
        idx_order = np.argsort(evals)
        evecs = evecs[idx_order,:]
        evals = evals[idx_order]
    return evals,evecs


# Simple in-memory cache for eigensystems computed for a given state and (Ez,Bz)
_eigensystem_cache = {}

def compute_eigensystems_for_state(state, EzBz_pairs, method='torch', round=10, use_cache=True):
    """
    Compute eigenpairs for multiple (Ez,Bz) pairs for a given `state`.

    - `EzBz_pairs` is an iterable of (Ez, Bz) tuples.
    - Uses the state's `H_function(Ez,Bz)` to build Hamiltonians, stacks them,
      and calls `diagonalize_batch` once for all matrices.
    - Optionally caches results in `_eigensystem_cache` keyed by
      (state.iso_state, Ez, Bz, method, round).

    Returns a list of (evals, evecs) tuples in the same order as input pairs.
    """
    pairs = list(EzBz_pairs)
    results = [None] * len(pairs)
    to_stack = []
    to_stack_idx = []

    for i, (Ez, Bz) in enumerate(pairs):
        key = (getattr(state, 'iso_state', None), float(Ez), float(Bz), method, int(round))
        if use_cache and key in _eigensystem_cache:
            results[i] = _eigensystem_cache[key]
        else:
            # build H for this (Ez,Bz)
            try:
                H = state.H_function(Ez, Bz)
            except TypeError:
                # some H_function signatures include additional args (I_trap, theta_trap)
                H = state.H_function(Ez, Bz, getattr(state, 'I_trap', None), getattr(state, 'theta_trap', None))
            to_stack.append(np.array(H, dtype=float))
            to_stack_idx.append((i, key))

    if to_stack:
        H_stack = np.stack(to_stack, axis=0)
        evals_batch, evecs_batch = diagonalize_batch(H_stack, method=method, round=round)
        for j, (i, key) in enumerate(to_stack_idx):
            val = (evals_batch[j], evecs_batch[j])
            if use_cache:
                _eigensystem_cache[key] = val
            results[i] = val

    return results


def build_H_stack_from_state(state, EzBz_pairs):
    """Return a numpy array of stacked Hamiltonians for given (Ez,Bz) pairs."""
    Hs = []
    for Ez, Bz in EzBz_pairs:
        try:
            H = state.H_function(Ez, Bz)
        except TypeError:
            H = state.H_function(Ez, Bz, getattr(state, 'I_trap', None), getattr(state, 'theta_trap', None))
        Hs.append(np.array(H, dtype=float))
    if not Hs:
        return np.empty((0,0,0), dtype=float)
    return np.stack(Hs, axis=0)

