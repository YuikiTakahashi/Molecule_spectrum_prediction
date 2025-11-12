import matrix_elements as me
import quantum_numbers as qn
import hamiltonian_builders as ham
from molecule_parameters import all_params
from functools import partial

class Molecule_Library(object):

    '''
    This class is primarily used as a library of functions and properties used
    to model the structure of YbOH. The class attributes are:

    matrix_elements: dictionary of relevant matrix elements, indexed by the
    YbOH state and isotope (ie, '174X000')

    cases: dictionary of the Hund's cases used for each state. The matrix
    elements are calculated in this basis.

    H_builders: dictionary of functions that construct the Hamiltonian matrix
    for a given isotope and state.

    PTV_builders: dictionary of functions that construct Hamiltonians for
    PT-violating effects.

    q_number_builders: dictionary of functions that construct the quantum
    numbers for a given state and isotope.
    The quantum numbers are represented as a dictionary, with the dict key
    giving the angular momentum name, and the dict value containing an array of
    the eigenvalues of each basis vector.
    For example, {'J': [0,1,1,1], 'M': [0,-1,0,1]}

    parameters: dictionary of the Hamiltonian constants associated with each
    state and isotope.

    Lambda: projection of angular momentum on the internuclear axis, excluding
    electron spin.
    '''

    def __init__(self,molecule,I_spins,M_values,P_values,trap):
        self.I_spins = I_spins
        self.M_values=M_values
        self.P_values=P_values
        self.trap = trap
        self.molecule = molecule
        self.parameters = all_params[self.molecule]
        if len(P_values)==1:
            self.parameters['ASO'] = 0
        self.matrix_elements = self.collect_all_matrix_elements(I_spins,M_values)
        self.cases = self.collect_all_cases()
        self.H_builders = self.collect_all_H_builders()
        self.PTV_builders = self.collect_all_PTV_builders()
        self.q_number_builders = self.collect_all_q_number_builders(I_spins, P_values)
        self.K = self.collect_all_K()
        self.basis_changers = self.collect_change_basis(I_spins)
        self.TDM_builders = self.collect_TDM()
        self.alt_q_number_builders = self.collect_alt_q(I_spins,P_values)
        self.TDM_p_builders = self.collect_p_TDM()
        self.all_parity = self.collect_parity()
        self.TDM_p_forbidden_builders = self.collect_p_TDM_forbidden()
        self.TDM_forbidden_builders = self.collect_TDM_forbidden()


    def collect_all_cases(self):
        if self.molecule=='YbOH':
            all_cases={
            '174X000': 'bBJ',
            '174X010': 'bBJ',
            '173X000': 'bBS',
            '173X010': 'bBS',
            '171X000': 'bBS',
            '171X010': 'bBS',
            '174A000': 'aBJ',
            '173A000': 'aBJ',
            '171A000': 'aBJ'
            }
        if self.molecule=='CaOH':
            all_cases={
            '40X000':'bBJ',
            '40X010':'bBJ',
            '40A000':'aBJ',
            '40A010':'aBJ',
            '40B000':'bBJ',
            }
        if self.molecule=='YbF':
            all_cases={
            '174X000': 'bBJ',
            '173X000': 'bBS',
            '171X000': 'bBS',
            '174A000': 'aBJ',
            '173A000': 'aBJ',
            '171A000': 'aBJ'
            }            
        return all_cases

    def collect_all_K(self):
        if self.molecule == 'YbOH':
            all_K={
            '174X000': 0,
            '174X010': 1,
            '173X000': 0,
            '171X000': 0,
            '173X010': 1,
            '171X010': 1,
            '174A000': 1,
            '173A000': 1,
            '171A000': 1
            }
        if self.molecule == 'CaOH':
            all_K ={
            '40X000': 0,
            '40X010': 1,
            '40A000': 1,
            '40A010': 2,
            '40B000':0,
            }
        if self.molecule == 'YbF':
            all_K={
            '174X000': 0,
            '173X000': 0,
            '171X000': 0,
            '174A000': 2,
            '173A000': 1,
            '171A000': 1
            }            
        return all_K


    def collect_all_matrix_elements(self,I_spins,M_values):
        iH = self.I_spins[-1]
        IM = self.I_spins[0]


        bBJ_even_X_matrix_elements={
        # Fine Structure
        'N^2': me.Rot_bBJ,                 # N^2 Rotation
        'N.S': me.SR_bBJ,                  # N.S Spin Rotation
        'l doubling': me.lD_bBJ,           # Effective l doubling
        'l doubling p':me.p_lD_bBJ,
        'l doubling q':me.q_lD_bBJ,
        'NzSz': me.NzSz_bBJ,               # NzSz for bending mode

        # Hydrogen Hyperfine
        'I.S': me.IS_bBJ,                  # I.S Fermi Contact Interaction
        'T2_0(I,S)': me.T2IS_bBJ,          # I S dipolar interaction
        'Iz': me.Iz_bBJ,                   # I.n projection of I on internuclear axis n
        'Sz': me.Sz_bBJ,                   # S.n projection of S on internuclear axis n
        }

        if M_values != 'none':
            ext_fields = {
            # External Fields
            'ZeemanZ': me.ZeemanZ_bBJ,         # Zeeman interaction with lab z magnetic field
            'StarkZ': me.StarkZ_bBJ            # Stark interaction with lab z electric field
            }
            bBJ_even_X_matrix_elements.update(ext_fields)
        if self.trap:
            trap_shifts = {
            'T0_azz': me.T00_alphaZZ_bBJ,
            'T0_axxyy': me.T00_alphaXXYY_bBJ,
            'T20_azz': partial(me.T20p_alphaZZ_bBJ,p=0),
            'T21_azz': partial(me.T20p_alphaZZ_bBJ,p=1),
            'T2-1_azz': partial(me.T20p_alphaZZ_bBJ,p=-1),
            'T22_azz': partial(me.T20p_alphaZZ_bBJ,p=2),
            'T2-2_azz': partial(me.T20p_alphaZZ_bBJ,p=-2),
            'T20_axxyy': partial(me.T20p_alphaXXYY_bBJ,p=0),
            'T21_axxyy': partial(me.T20p_alphaXXYY_bBJ,p=1),
            'T2-1_axxyy': partial(me.T20p_alphaXXYY_bBJ,p=-1),
            'T22_axxyy': partial(me.T20p_alphaXXYY_bBJ,p=2),
            'T2-2_axxyy': partial(me.T20p_alphaXXYY_bBJ,p=-2),
            }
            bBJ_even_X_matrix_elements.update(trap_shifts)
        for term,element in bBJ_even_X_matrix_elements.items():       #iterate through, substitute hydrogen proton value
            bBJ_even_X_matrix_elements[term] = partial(element,I=iH)




        bBS_odd_X_matrix_elements={
        # Fine Structure
        'N^2': me.Rot_bBS,                 # N^2 Rotation
        'N.S': me.SR_bBS,                  # N.S Spin Rotation
        'l doubling': me.lD_bBS,           # Effective l doubling
        'l doubling p':me.p_lD_bBS,
        'l doubling q':me.q_lD_bBS,
        'NzSz': me.NzSz_bBS,               # NzSz for bending mode

        # Yb Hyperfine
        'IM.S': me.ISM_bBS,              # IM.S Fermi Contact Interaction
        'T2_0(IM,S)': me.T2ISM_bBS,      # q=0 molecule frame electron-IM dipole-dipole interaction (sqrt6 * (3IzSz - I.S))
        'T2_0(IM^2)': me.T2QM_bBS,       # q=0 molecule frame Yb electric quadrupole interaction (sqrt6 * (3Iz^2 - I^2))

        # Hydrogen Hyperfine
        'IH.S': me.ISH_bBS,                # IH.S Fermi Contact Interaction for Hydrogen
        'T2_0(IH,S)': me.T2ISH_bBS        # q=0 molecule frame electron-IH dipole-dipole interaction (sqrt6 * (3IzSz - I.S))
        }

        if M_values != 'none':
            ext_fields = {
            # External Fields
            'ZeemanZ': me.ZeemanZ_bBS,         # Zeeman interaction with lab z magnetic field
            'StarkZ': me.StarkZ_bBS            # Stark interaction with lab z electric field
            }
            bBS_odd_X_matrix_elements.update(ext_fields)
        for term, element in bBS_odd_X_matrix_elements.items():      #substitue nuclear spin values
             bBS_odd_X_matrix_elements[term] = partial(element,iH=iH,I=IM)

        # bBS_odd_X_matrix_elements={
        # # Fine Structure
        # 'N^2': me.Rot_bBS,                 # N^2 Rotation
        # 'N.S': me.SR_bBS,                  # N.S Spin Rotation
        # 'l-doubling': me.lD_bBS,           # Effective l doubling
        # 'NzSz': me.NzSz_bBS,               # NzSz for bending mode
        #
        # # Yb Hyperfine
        # 'IM.S': me.ISM_bBS,              # IM.S Fermi Contact Interaction
        # 'T2_0(IM,S)': me.T2ISM_bBS,      # q=0 molecule frame electron-IM dipole-dipole interaction (sqrt6 * (3IzSz - I.S))
        # # 'T2_0(IM^2)': me.T2QM_bBS,       # q=0 molecule frame Yb electric quadrupole interaction (sqrt6 * (3Iz^2 - I^2))
        #
        # # Hydrogen Hyperfine
        # 'IH.S': me.ISH_bBS,                # IH.S Fermi Contact Interaction for Hydrogen
        # 'T2_0(IH,S)': me.T2ISH_bBS        # q=0 molecule frame electron-IH dipole-dipole interaction (sqrt6 * (3IzSz - I.S))
        #
        # }
        #
        # if M_values != 'none':
        #     ext_fields = {
        #     # External Fields
        #     'ZeemanZ': me.ZeemanZ_bBS,         # Zeeman interaction with lab z magnetic field
        #     'StarkZ': me.StarkZ_bBS            # Stark interaction with lab z electric field
        #     }
        #     bBS_odd_X_matrix_elements.update(ext_fields)
        # for term, element in bBS_odd_X_matrix_elements.items():      #substitue nuclear spin values
        #      bBS_odd_X_matrix_elements[term] = partial(element,iH=iH,I=IM)

        
        
        aBJ_even_A_matrix_elements={
        # Fine Structure
        'N^2': me.Rot_even_aBJ,                 # N^2 Rotation
        'SO': me.SO_even_aBJ,                   # Spin-Orbit
        'Lambda Doubling p+2q': me.LambdaDoubling_p2q_even_aBJ,          # Effective Lambda doubling
        'Lambda Doubling q':me.LambdaDoubling_q_even_aBJ,

        # Hydrogen Hyperfine
        'I.S': me.IS_even_aBJ,                   # I.S Fermi Contact Interaction
        'IzSz': me.IzSz_even_aBJ,                # I.n*S.n projection of I and S on internuclear axis n
        'T2_0(IS)': me.T2IS_even_aBJ            # q=0 molecule frame electron-IH dipole-dipole interaction (sqrt6 * (3IzSz - I.S))
        }


        if M_values != 'none':
            ext_fields = {
            # External Fields
            'ZeemanLZ': me.ZeemanLZ_even_aBJ,
            'ZeemanSZ': me.ZeemanSZ_even_aBJ,
            'ZeemanParityZ': me.ZeemanParityZ_even_aBJ,
            'StarkZ': me.StarkZ_even_aBJ,            # Stark interaction with lab z electric field
            }
            aBJ_even_A_matrix_elements.update(ext_fields)

        for term, element in aBJ_even_A_matrix_elements.items():      #substitue nuclear spin values
             aBJ_even_A_matrix_elements[term] = partial(element,I=iH)
                
                
                

        aBJ_even_A_matrix_elements_3Delta1={
        # Fine Structure
        'J^2': me.Rot_even_aBJ_3Delta1,                 # J^2 Rotation
        'Omega Doubling': me.OmegaDoubling_even_aBJ_3Delta1,          # Effective Omega doubling

        # Fluoride Hyperfine
        'A_parallel HF': me.A_parallel_HF_even_aBJ_3Delta1,                   # A_parallel magnetic HF parameter for HfF+ and ThF+

            
        # External Fields
        'StarkZ': me.StarkZ_even_aBJ_3Delta1,
        'Zeeman eff Z': me.Zeeman_eff_even_aBJ_3Delta1,                          # Effective Zeeman term
            
        }
        
        for term, element in aBJ_even_A_matrix_elements_3Delta1.items():      #substitue nuclear spin values
             aBJ_even_A_matrix_elements_3Delta1[term] = partial(element,I=iH)
                
                
        aBJ_odd_A_matrix_elements={
        # Fine Structure
        'N^2': me.Rot_odd_aBJ,                 # N^2 Rotation
        'SO': me.SO_odd_aBJ,                    # Spin Orbit
        'Lambda-Doubling': me.LambdaDoubling_odd_aBJ,       #Lambda doubling

        # Yb Hypeerfine
        'IzLz_M': me.ILM_odd_aBJ,
        'T2_2(IS)_M': me.T2q2_ISM_odd_aBJ,
        'T2_0(II)_M': me.T2q0_IM_odd_aBJ,


        # Hydrogen Hyperfine
        # 'IS_H': ,                   # I.S Fermi Contact Interaction
        # 'Iz_H': ,                   # I.n projection of I on internuclear axis n
        # 'Sz_H': ,                   # S.n projection of S on internuclear axis n
        }

        if M_values != 'none':
            ext_fields = {
            # External Fields
            'ZeemanLZ': me.ZeemanLZ_odd_aBJ,
            'ZeemanSZ': me.ZeemanSZ_odd_aBJ,
            'ZeemanParityZ': me.ZeemanParityZ_odd_aBJ,
            'StarkZ': me.StarkZ_odd_aBJ,            # Stark interaction with lab z electric field
            }
            aBJ_odd_A_matrix_elements.update(ext_fields)
        for term, element in aBJ_odd_A_matrix_elements.items():      #substitue nuclear spin values
             aBJ_odd_A_matrix_elements[term] = partial(element,iH=iH,I=IM)

        # aBJ_odd_A_matrix_elements={
        # # Fine Structure
        # 'N^2': me.Rot_odd_aBJ,                 # N^2 Rotation
        # 'SO': me.SO_odd_aBJ,                    # Spin Orbit
        # 'Lambda-Doubling': me.LambdaDoubling_odd_aBJ,       #Lambda doubling
        #
        # # Yb Hypeerfine
        # 'IzLz_M': me.ILM_odd_aBJ,
        # 'T2_2(IS)_M': me.T2q2_ISM_odd_aBJ,
        # # 'T2_0(II)_M': me.T2q0_IM_odd_aBJ,
        #
        #
        # # Hydrogen Hyperfine
        # # 'IS_H': ,                   # I.S Fermi Contact Interaction
        # # 'Iz_H': ,                   # I.n projection of I on internuclear axis n
        # # 'Sz_H': ,                   # S.n projection of S on internuclear axis n
        # }
        #
        #
        # if M_values != 'none':
        #     ext_fields = {
        #     # External Fields
        #     'ZeemanLZ': me.ZeemanLZ_odd_aBJ,
        #     'ZeemanSZ': me.ZeemanSZ_odd_aBJ,
        #     'ZeemanParityZ': me.ZeemanParityZ_odd_aBJ,
        #     'StarkZ': me.StarkZ_odd_aBJ,            # Stark interaction with lab z electric field
        #     }
        #     aBJ_odd_A_matrix_elements.update(ext_fields)
        # for term, element in aBJ_odd_A_matrix_elements.items():      #substitue nuclear spin values
        #      aBJ_odd_A_matrix_elements[term] = partial(element,iH=iH,I=IM)

        if self.molecule=='YbOH':
            all_matrix_elements={
            '174X000': bBJ_even_X_matrix_elements,
            '174X010': bBJ_even_X_matrix_elements,
            '173X000': bBS_odd_X_matrix_elements,
            '173X010': bBS_odd_X_matrix_elements,
            '174A000': aBJ_even_A_matrix_elements,
            '173A000': aBJ_odd_A_matrix_elements,
            '171A000': aBJ_odd_A_matrix_elements,
            '171X000': bBS_odd_X_matrix_elements,
            '171X010': bBS_odd_X_matrix_elements
            }
        if self.molecule=='CaOH':
            all_matrix_elements = {
            '40X000': bBJ_even_X_matrix_elements,
            '40X010': bBJ_even_X_matrix_elements,
            '40A000': aBJ_even_A_matrix_elements,
            '40A010': aBJ_even_A_matrix_elements,
            '40B000': bBJ_even_X_matrix_elements,
            }
        if self.molecule=='YbF':
            all_matrix_elements={
            '174X000': bBJ_even_X_matrix_elements,
            '173X000': bBS_odd_X_matrix_elements,
            '174A000': aBJ_even_A_matrix_elements_3Delta1,
            '173A000': aBJ_odd_A_matrix_elements,
            '171A000': aBJ_odd_A_matrix_elements,
            '171X000': bBS_odd_X_matrix_elements,
            }
        return all_matrix_elements

    def collect_all_H_builders(self):
        if self.molecule == 'YbOH':
            H_builders = {
                '174X000': ham.H_even_X,
                '174X010': ham.H_even_X,
                '173X000': ham.H_odd_X,
                '173X010': ham.H_odd_X,
                '174A000': ham.H_even_A,
                '173A000': ham.H_odd_A,
                '171A000': ham.H_odd_A,
                '171X000': ham.H_odd_X,
                '171X010': ham.H_odd_X
                }
        if self.molecule == 'CaOH':
            H_builders = {
                '40X000': ham.H_even_X,
                '40X010': partial(ham.H_even_X,trap=self.trap),
                '40A000': ham.H_even_A,
                '40B000': ham.H_even_X,
                # '40A010': ham.H_even_A
                }
        if self.molecule == 'YbF':
            H_builders = {
                '174X000': ham.H_even_X,
                '173X000': ham.H_odd_X,
                '174A000': ham.H_even_3Delta1,
                '173A000': ham.H_odd_A,
                '171A000': ham.H_odd_A,
                '171X000': ham.H_odd_X,
                }            
        for key in H_builders:
            H_builders[key] = partial(H_builders[key],params = self.parameters[key],matrix_elements = self.matrix_elements[key])
        return H_builders

    def collect_all_q_number_builders(self,I_spins,P_values):
        if self.molecule == 'YbOH':
            q_number_builders = {
                '174X000': partial(qn.q_numbers_even_bBJ, K_mag=0),
                '174X010': partial(qn.q_numbers_even_bBJ, K_mag=1),
                '173X000': partial(qn.q_numbers_bBS, K_mag=0),
                '173X010': partial(qn.q_numbers_bBS, K_mag=1),
                '174A000': partial(qn.q_numbers_even_aBJ, K_mag=1,P_values=P_values),
                '173A000': partial(qn.q_numbers_odd_aBJ, K_mag=1,P_values=P_values),
                '171X000': partial(qn.q_numbers_bBS, K_mag=0),
                '171X010': partial(qn.q_numbers_bBS, K_mag=1),
                '171A000': partial(qn.q_numbers_odd_aBJ, K_mag=1,P_values=P_values),
                # '174aBJ': qn.q_numbers_174_aBJ,
                # '174bBJ': qn.q_numbers_bBJ,
                }
        if self.molecule=='CaOH':
            q_number_builders = {
                '40X000': partial(qn.q_numbers_even_bBJ, K_mag=0),
                '40X010': partial(qn.q_numbers_even_bBJ,K_mag=1),
                '40A000': partial(qn.q_numbers_even_aBJ, K_mag=1,P_values=P_values),
                '40A010': partial(qn.q_numbers_even_aBJ, K_mag=2,P_values=P_values),
                '40B000': partial(qn.q_numbers_even_bBJ, K_mag=0),
                }
        if self.molecule == 'YbF':
            q_number_builders = {
                '174X000': partial(qn.q_numbers_even_bBJ, K_mag=0),
                '173X000': partial(qn.q_numbers_bBS, K_mag=0),
                # Yuiki is using '174A00' in YbF for 3Delta1 state
                '174A000': partial(qn.q_numbers_even_aBJ, K_mag=2,S=1,P_values=P_values),
                '173A000': partial(qn.q_numbers_odd_aBJ, K_mag=1,P_values=P_values),
                '171X000': partial(qn.q_numbers_bBS, K_mag=0),
                '171A000': partial(qn.q_numbers_odd_aBJ, K_mag=1,P_values=P_values),
                }            
        for key,builder in q_number_builders.items():
            q_number_builders[key] = partial(builder,I_list=self.I_spins)
        return q_number_builders

    def collect_all_PTV_builders(self):
        IM = self.I_spins[0]
        iH = self.I_spins[-1]

        if self.molecule == 'YbOH':
            PTV_builders = {
                '174X000': ham.build_PTV_bBJ,
                '174X010': ham.build_PTV_bBJ,
                '173X000': ham.build_PTV_bBS,
                '173X010': partial(ham.build_PTV_bBS,IM=IM,iH=iH),
                '171X010': partial(ham.build_PTV_bBS,IM=IM,iH=iH)
            }
        if self.molecule == 'CaOH':
            PTV_builders = {
                '40X000': ham.build_PTV_bBJ,
                '40X010': ham.build_PTV_bBJ,
            }
        if self.molecule == 'YbF':
            PTV_builders = {
                '174X000': ham.build_PTV_bBJ,
                '173X000': partial(ham.build_PTV_bBS,IM=IM,iH=iH),
                '171X000': partial(ham.build_PTV_bBS,IM=IM,iH=iH),
                '174A000': ham.build_PTV_aBJ,
            }            
        return PTV_builders

    def collect_change_basis(self,I_spins):
        all_change_basis = {
            'a_bBJ': ham.convert_abBJ,
            'b_decoupled': partial(ham.decouple_b,I=max(I_spins)), #WiLL NOT WORK for case bBS
            'bBS_bBJ': partial(ham.convert_bbBS,I=max(I_spins)),
            'recouple_J': partial(ham.recouple_J,I=max(I_spins)),
            'b_I_decoupled': partial(ham.decouple_b_I,I=max(I_spins)),
            # 'A000_B010': partial(ham.build_operator,matrix_element = me.HRTSO_A000_B010),
            # 'A000_uA010': partial(ham.build_operator,matrix_element = me.HRTSO_A000_uA010),
            # 'A000_kA010': partial(ham.build_operator,matrix_element = me.HRTSO_A000_kA010),
            # 'X010_vibronic': partial(ham.build_operator,matrix_element = me.convert_X010_vibronic),
            # 'A000_vibronic': partial(ham.build_operator,matrix_element = me.convert_A000_vibronic)
        }
        return all_change_basis

    def collect_p_TDM(self):
        IM = self.I_spins[0]
        iH = self.I_spins[-1]
        if self.M_values != 'none':
            if self.molecule == 'YbOH':
                p_TDM = {
                    '174A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '174X010': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '173A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_odd_aBJ,iH=iH,I=IM)),
                    '171A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_odd_aBJ,iH=iH,I=IM)),
                    '173X010': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_odd_bBS,iH=iH,I=IM)),
                }
            if self.molecule == 'CaOH':
                p_TDM = {
                    '40A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '40B000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '40X010': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '40X000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                } 
                
            if self.molecule == 'YbF':
                p_TDM = {
                    '174A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '174X010': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_even_aBJ,I=iH)),
                    '173A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_odd_aBJ,iH=iH,I=IM)),
                    '171A000': partial(ham.build_p_TDM_aBJ,TDM_matrix_element=partial(me.TDM_p_odd_aBJ,iH=iH,I=IM))
                }
                
            return p_TDM
        else:
            return None

    def collect_p_TDM_forbidden(self):
        IM = self.I_spins[0]
        iH = self.I_spins[-1]
        if self.M_values != 'none':
            if self.molecule == 'YbOH':
                p_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ_forbidden,TDM_matrix_element=partial(me.TDM_p_vibronic_aBJ,I=iH)),
                    'uA010': partial(ham.build_TDM_aBJ_forbidden,TDM_matrix_element=partial(me.TDM_p_uA010_aBJ,I=iH)),
                    'kA010': partial(ham.build_TDM_aBJ_forbidden,TDM_matrix_element=partial(me.TDM_p_kA010_aBJ,I=iH)),
                    'B010': partial(ham.build_TDM_aBJ_forbidden,TDM_matrix_element=partial(me.TDM_p_B010_aBJ,I=iH)),
                }
            if self.molecule == 'CaOH':
                p_TDM = {}
            if self.molecule == 'YbF':
                p_TDM = {}                
                
            return p_TDM
        else:
            return None

    def collect_TDM(self):
        IM = self.I_spins[0]
        iH = self.I_spins[-1]
        if self.M_values != 'none':
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                    '173A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ,iH=iH,I=IM)),
                    '171A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ,iH=iH,I=IM))
                }
            if self.molecule == 'CaOH':
                all_TDM = {
                    '40A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                    '40B000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                }
            if self.molecule == 'YbF':
                all_TDM = {}                
        else:
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    '173A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ_noM,iH=iH,I=IM)),
                    '171A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ_noM,iH=iH,I=IM))
                }
            if self.molecule == 'CaOH':
                all_TDM = {
                    '40A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    '40B000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    }
            if self.molecule == 'YbF':
                all_TDM = {}                   
        return all_TDM

    def collect_TDM_forbidden(self):
        IM = self.I_spins[0]
        iH = self.I_spins[-1]
        if self.M_values != 'none':
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_operator,matrix_element=partial(me.TransitionDipole_even_aBJ_vibronic,I=iH)),
                }
            if self.molecule == 'CaOH':
                all_TDM = {}
            if self.molecule == 'YbF':
                all_TDM = {}                
        else:
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ_forbidden,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM_vibronic,I=iH)),
                }
            if self.molecule == 'CaOH':
                all_TDM = {
                    }
            if self.molecule == 'YbF':
                all_TDM = {}                  
        return all_TDM

    def collect_TDM(self):
        iH = self.I_spins[-1]
        IM = self.I_spins[0]
        if self.M_values != 'none':
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                    '173A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ,iH=iH,I=IM)),
                    '171A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ,iH=iH,I=IM))
                }
            if self.molecule == 'CaOH':
                all_TDM = {
                    '40A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                    '40B000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ,I=iH)),
                }
                
            if self.molecule == 'YbF':
                all_TDM = {}     
                
        else:
            if self.molecule == 'YbOH':
                all_TDM = {
                    '174A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    '173A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ_noM,iH=iH,I=IM)),
                    '171A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_odd_aBJ_noM,iH=iH,I=IM))
                }
            if self.molecule == 'CaOH':
                all_TDM = {
                    '40A000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    '40B000': partial(ham.build_TDM_aBJ,TDM_matrix_element=partial(me.TransitionDipole_even_aBJ_noM,I=iH)),
                    }
                
            if self.molecule == 'YbF':
                all_TDM = {} 
                
        return all_TDM

    def collect_parity(self):
        iH = self.I_spins[-1]
        IM = self.I_spins[0]
        if self.molecule=='YbOH':
            all_parity = {
                '174X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBJ,I=iH)),
                '174X010': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBJ,I=iH)),
                '171X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                '173X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                '171X010': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                '173X010': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                '174A000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_aBJ,I=iH)),
                '173A000': partial(ham.build_operator,matrix_element=partial(me.Parity_odd_L_aBJ,I=IM,iH=iH)),
                '171A000': partial(ham.build_operator,matrix_element=partial(me.Parity_odd_L_aBJ,I=IM,iH=iH)),
            }
        if self.molecule == 'CaOH':
            all_parity = {
                '40X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBJ,I=iH)),
                '40X010': partial(ham.build_operator,matrix_element=partial(me.Parity_l_bBJ,I=iH)),
                '40B010': partial(ham.build_operator,matrix_element=partial(me.Parity_l_bBJ,I=iH)),
                '40A000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_aBJ,I=iH)),
                '40B000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_aBJ,I=iH))
            }
        if self.molecule=='YbF':
            all_parity = {
                '174X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBJ,I=iH)),
                '171X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                '173X000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_bBS,I=IM,iH=iH)),
                # Yuiki is using '174A00' in YbF for 3Delta1 state
                '174A000': partial(ham.build_operator,matrix_element=partial(me.Parity_L_aBJ_3Delta1,I=iH)),
                '173A000': partial(ham.build_operator,matrix_element=partial(me.Parity_odd_L_aBJ,I=IM,iH=iH)),
                '171A000': partial(ham.build_operator,matrix_element=partial(me.Parity_odd_L_aBJ,I=IM,iH=iH)),
            }            
        return all_parity

    def collect_alt_q(self,I_spins,P_values):
        if self.molecule=='YbOH':
            alt_q_builders = {
                '174X000': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins)
                    },
                '174X010': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=1,P_values=[1/2,3/2],I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins),
                    'recouple_J': partial(qn.q_numbers_decoupled_mJ, K_mag=1,I_list = I_spins),
                    'decouple_I': partial(qn.q_numbers_decoupled_mJ, K_mag=1,I_list = I_spins),
                    'vibronic': partial(qn.q_numbers_vibronic_even_aBJ,l_mag=1,L_mag=0,I_list=I_spins,P_values=[1/2,3/2])
                    },
                '173X000': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=0,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled,K_mag=0,I_list = I_spins)
                    },
                '173X010': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=1,P_values=[1/2,3/2],I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    },
                '174A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_even_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins),
                    # 'vibronic': partial(qn.q_numbers_vibronic_even_aBJ,l_mag=0,L_mag=1,I_list=I_spins,P_values=P_values),
                    # 'uA010': partial(qn.q_numbers_vibronic_even_aBJ,l_mag=1,L_mag=1,Omega_values=[1/2], K_values=[0],I_list=I_spins,P_values=P_values),
                    # 'kA010': partial(qn.q_numbers_vibronic_even_aBJ,l_mag=1,L_mag=1,Omega_values=[3/2], K_values=[0],I_list=I_spins,P_values=P_values),
                    # 'B010': partial(qn.q_numbers_vibronic_even_aBJ,l_mag=1,L_mag=0,I_list=I_spins,P_values=P_values),
                    },
                '173A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    },
                '171X000': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=0,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins)
                    },
                '171X010': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=1,I_list = I_spins,P_values=[1/2,3/2]),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    },
                '171A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    }
                }
        if self.molecule == 'CaOH':
            alt_q_builders = {
                '40X000': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins),
                    'bBS': partial(qn.q_numbers_bBS, K_mag=0,I_list = I_spins),
                    },
                '40X010': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=1,P_values=[1/2,3/2],I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins),
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    },
                '40A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_even_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins),
                    'recouple_J': partial(qn.q_numbers_decoupled_mJ, K_mag=1,I_list = I_spins)
                    },
                '40B000': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins)
                    },
                }
        if self.molecule=='YbF':
            alt_q_builders = {
                '174X000': {
                    'aBJ': partial(qn.q_numbers_even_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins)
                    },

                '173X000': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=0,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled,K_mag=0,I_list = I_spins)
                    },

                '174A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_even_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins),
                    },
                '173A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    },
                '171X000': {
                    'aBJ': partial(qn.q_numbers_odd_aBJ, K_mag=0,I_list = I_spins,P_values=[1/2]),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=0,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=0,I_list = I_spins)
                    },

                '171A000': {
                    'bBS': partial(qn.q_numbers_bBS, K_mag=1,I_list = I_spins),
                    'bBJ': partial(qn.q_numbers_odd_bBJ, K_mag=1,I_list = I_spins),
                    'decoupled': partial(qn.q_numbers_decoupled, K_mag=1,I_list = I_spins)
                    }
                }            
        return alt_q_builders
