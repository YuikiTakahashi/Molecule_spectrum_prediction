def set_state_parameters(state, updates=None):
    """Update the molecule parameters, rebuild the Hamiltonian, and refresh eigenvectors."""
    updates = updates or {}
    new_params = {**BASE_PARAMETERS}
    new_params.update(updates)

    state.parameters = new_params
    state.library.parameters[state.iso_state] = new_params
    
    # Rebuild H_builder with new parameters (H_builders are partial functions with params/matrix_elements already bound)
    # We need to create a new partial function with the updated parameters
    from functools import partial
    import hamiltonian_builders as ham
    
    # Get the original H builder function from the partial
    existing_builder = state.library.H_builders[state.iso_state]
    if hasattr(existing_builder, 'func'):
        H_builder_func = existing_builder.func
    else:
        # Fallback: determine function based on iso_state
        if state.iso_state in ['174X000', '174X010']:
            H_builder_func = ham.H_even_X
        elif state.iso_state in ['173X000', '173X010', '171X000', '171X010']:
            H_builder_func = ham.H_odd_X
        elif state.iso_state in ['174A000']:
            H_builder_func = ham.H_even_A
        elif state.iso_state in ['173A000', '171A000']:
            H_builder_func = ham.H_odd_A
        else:
            raise ValueError(f"Unknown H_builder for {state.iso_state}")
    
    # Create new partial with updated parameters
    new_H_builder = partial(H_builder_func, 
                           params=new_params, 
                           matrix_elements=state.matrix_elements)
    
    # Now call it with the remaining arguments (don't pass params/matrix_elements again)
    if state.trap:
        state.H_function, state.H_symbolic = new_H_builder(
            state.q_numbers,
            M_values=state.M_values,
            precision=state.round,
            theta_num=state.theta_num
        )
    else:
        state.H_function, state.H_symbolic = new_H_builder(
            state.q_numbers,
            M_values=state.M_values,
            precision=state.round
        )
    
    state.eigensystem(0, 1e-8, order=True, method="torch", set_attr=True)
    state.generate_parities(state.evecs0)
    return state

