import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as Estimator_sim
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from circuits import (
    circuit8qubits_statevector,
    circuit8qubits_projector,
    circuit8qubits_psiU00,
    circuit8qubits_measureZ
)

"""
Simulation Module

Contains functions to simulate evolution of system with statevector and noiseless simulation of quantum computer
Also has functions to run on IBM devices, and compare this to the circuit simulation
"""

# Define expectation value operators
SINGLET_OP = SparsePauliOp(["IIIIIIII", "-IIIXXIII", "-IIIYYIII", "-IIIZZIII"], coeffs=0.25)
ZEFF_OP = SparsePauliOp(['IIIIZIII', 'IIIZIIII'], coeffs=[0.5+0.j, -0.5+0.j])
XEFF_OP = SparsePauliOp(['IIIXXIII', 'IIIYYIII'], coeffs=[0.5+0.j, 0.5-0.j])

OPERATOR_LABELS = {
    'singlet': r'$|\langle S|\psi\rangle |^2$',
    'Zeff': r'$\langle \psi |Z_{eff}| \psi \rangle$',
    'Xeff': r'$\langle \psi |X_{eff}| \psi \rangle$'
}


def get_operator(op_name):
    """
    Retrieve a Sparse Pauli operator for observables to measure.
    
    Parameters:
        op_name (str): Name of the operator to retrieve ('singlet', 'Zeff', or 'Xeff')
    
    Returns:
        SparsePauliOp: The requested operator
        
    Raises:
        ValueError: If op_name is not one of the supported operators
    """
    if op_name == 'singlet':
        return SINGLET_OP
    elif op_name == 'Zeff':
        return ZEFF_OP
    elif op_name == 'Xeff':
        return XEFF_OP
    else:
        raise ValueError(f"Unknown operator: {op_name}")

def calculate_timesteps(T, d):
    """
    Calculate evenly spaced time steps based on total evolution time and number of steps.
    
    Parameters:
        T (float): Total evolution time
        d (int): Number of time steps
    
    Returns:
        list: Array of time points [0, dt, 2*dt, ..., (d-1)*dt] where dt = T/d
    """
    dt = T / d if d > 0 else 0
    return [i * dt for i in range(d)]

def plot_expectation_values(ts, exp_values, operator, compare=None, sv_evals=None, 
                           noiseless_evals=None, savefile_path=None, savefig=False):
    """
    Plot expectation values with optional comparison to other simulation methods.
    
    Parameters:
        ts (list): List of time points
        exp_values (dict or list): Expectation values to plot (dictionary for multiple operators)
        operator (str): Operator to plot ('all', 'singlet', 'Zeff', or 'Xeff')
        compare (str, optional): Comparison mode ('statevector', 'noiseless', or 'both')
        sv_evals (dict, optional): State vector simulation results for comparison
        noiseless_evals (dict, optional): Noiseless simulation results for comparison
        savefile_path (str, optional): Path prefix for saving figures
        savefig (bool, optional): Whether to save figures to disk (default: False)
    
    Returns:
        None: Displays plots and optionally saves them to disk
    """
    if operator == 'all':
        ops = ['singlet', 'Zeff', 'Xeff']
        fig = plt.figure(figsize=(10, 15))
        
        for i, op in enumerate(ops):
            ax = fig.add_subplot(3, 1, i+1)
            ax.plot(ts, exp_values[op], label='Results')
            
            if compare in ['statevector', 'both'] and sv_evals:
                ax.plot(ts, sv_evals[op], label='Statevector sim')
                
            if compare in ['noiseless', 'both'] and noiseless_evals:
                ax.plot(ts, noiseless_evals[op], label='Noiseless sim')
                
            
            ax.set_ylabel(OPERATOR_LABELS[op])
            ax.set_xlabel('t')
            ax.legend()
            
        fig.tight_layout()
        
    else:
        plt.figure()
        plt.plot(ts, exp_values[operator] if isinstance(exp_values, dict) else exp_values, 
                label='Results')
        
        if compare in ['statevector', 'both'] and sv_evals:
            plt.plot(ts, sv_evals[operator], label='Statevector sim')
            
        if compare in ['noiseless', 'both'] and noiseless_evals:
            plt.plot(ts, noiseless_evals[operator], label='Noiseless sim')
            
        plt.xlabel('$t$')
        plt.ylabel(OPERATOR_LABELS[operator])
        plt.legend()
    
    if savefig and savefile_path:
        plt.savefig(f"{savefile_path}.svg")
    
    plt.show()

def statevector_simulate(data, savefile_path=None, circuit=circuit8qubits_statevector, 
                        T=1, dJ=0.5, d=0, one_qubits=[0, 2, 4, 6], 
                        operator='all', show=True, savefig=False):
    """
    Perform ideal statevector simulation of quantum circuits without noise.
    
    This function simulates quantum circuits using the statevector method, which
    provides a complete quantum state representation. It calculates expectation
    values for specified operators and can save/plot the results.
    
    Parameters:
        data (list): Input data for circuit construction
        savefile_path (str, optional): Path prefix for saving results
        circuit (function, optional): Circuit construction function to use
        T (float, optional): Total evolution time (default: 1)
        dJ (float, optional): Coupling parameter (default: 0.5)
        d (int, optional): Number of time steps (default: 0)
        one_qubits (list, optional): Indices of qubits initialized to |1⟩ (default: [0, 2, 4, 6])
        operator (str, optional): Operator to calculate expectation values for
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
                'all': all of the operators (default)
                None: no expecation value calculated 
        show (bool, optional): Whether to display plots (default: True)
        savefig (bool, optional): Whether to save figures to disk (default: False)
    
    Returns:
        tuple: (exp_values, sv_data)
            - exp_values (dict): Dictionary of expectation values for each operator
            - sv_data (dict): Raw statevector data from simulation
    
    Raises:
        ValueError: If operator is not one of the supported options
    """
    if operator not in ['all', 'singlet', 'Zeff', 'Xeff', None]:
        raise ValueError("operator must be one of: 'all', 'singlet', 'Zeff', 'Xeff' or None")
    
    backend = AerSimulator(method='statevector')
    dt = T / d if d > 0 else 0
    
    qc = transpile(circuit(data, d, dJ, dt, one_qubits), backend)
    result = backend.run(qc.decompose(), shots=None).result()
    sv_data = result.data(0)
    del sv_data['counts']
    
    if savefile_path:
        pd.DataFrame(sv_data).to_csv(f"{savefile_path}.csv")
    
    if operator is None:
        return None, sv_data
    
    # Calculate expectation values
    exp_values = {'singlet': [], 'Zeff': [], 'Xeff': []}
    ts = calculate_timesteps(T, d)
    
    for i in range(d):
        state_vector = np.asarray(sv_data[f'd = {i}'])
        for op_name in exp_values:
            op = get_operator(op_name)
            exp_values[op_name].append(
                np.real(state_vector.conj().T @ np.asarray(op) @ state_vector)
            )
    
    exp_values['t'] = ts
    
    if show and d > 0:
        plot_expectation_values(ts, exp_values, operator, savefile_path=savefile_path, savefig=savefig)
    
    return exp_values, sv_data

def noiseless_simulate(data, savefile_path=None, T=1, dJ=0.5, max_d=40, shots=1000,
                      method='direct', one_qubits=[0, 2, 4, 6], 
                      operator='all', show=True, savefig=False):
    """
    Perform noiseless circuit simulation using either direct measurement or estimator.
    
    This function simulates quantum circuits using a shot-based approach but without
    hardware noise models. It supports two simulation methods: direct measurement or
    using the Estimator primitive.
    
    Parameters:
        data (list): Input data for circuit construction
        savefile_path (str, optional): Path prefix for saving results
        T (float, optional): Total evolution time (default: 1)
        dJ (float, optional): Coupling parameter (default: 0.5)
        max_d (int, optional): Maximum number of time steps (default: 40)
        shots (int, optional): Number of measurement shots per circuit (default: 1000)
        method (str, optional): Simulation method ('direct' or 'estimator') (default: 'direct')
        one_qubits (list, optional): Indices of qubits initialized to |1⟩ (default: [0, 2, 4, 6])
        operator (str, optional): Operator to calculate expectation values for
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
                'all': all of the operators (default)
        show (bool, optional): Whether to display plots (default: True)
        savefig (bool, optional): Whether to save figures to disk (default: False)
    
    Returns:
        dict: Dictionary of expectation values for each operator and time point
    
    Raises:
        ValueError: If method is not 'direct' or 'estimator'
    """
    if method not in ['direct', 'estimator']:
        raise ValueError("method must be either 'direct' or 'estimator'")
    
    backend = AerSimulator()
    dt = T / max_d
    exp_values = {'singlet': [], 'Zeff': [], 'Xeff': []}
    
    if method == 'estimator':
        estimator = Estimator_sim(run_options={'shots': shots})
        observables = [get_operator(op) for op in ['singlet', 'Zeff', 'Xeff']]
        
        for d in range(max_d):
            psi = circuit8qubits_projector(data, d, dJ, dt, one_qubits).decompose()
            result = estimator.run([psi, psi, psi], observables).result().values
            
            for i, op in enumerate(['singlet', 'Zeff', 'Xeff']):
                exp_values[op].append(result[i])
    
    elif method == 'direct':
        for d in range(max_d):
            # Singlet measurement
            singlet_circuit = transpile(circuit8qubits_psiU00(data, d, dJ, dt, one_qubits), backend)
            singlet_result = backend.run(singlet_circuit.decompose(), shots=shots).result()
            singlet_count = singlet_result.get_counts()
            exp_values['singlet'].append(singlet_count.get('01', 0) / shots)
            
            # Z-effective measurement
            z_circuit = transpile(circuit8qubits_measureZ(data, d, dJ, dt, one_qubits), backend)
            z_result = backend.run(z_circuit.decompose(), shots=shots).result()
            z_count = z_result.get_counts()
            
            if "01" in z_count:
                exp_values['Zeff'].append((z_count.get('10', 0) - z_count.get('01', 0)) / shots)
            else:
                exp_values['Zeff'].append(z_count.get('10', 0) / shots)
            
            # X-effective measurement (using estimator)
            psi = circuit8qubits_projector(data, d, dJ, dt, one_qubits).decompose()
            estimator = Estimator_sim(run_options={'shots': shots})
            x_result = estimator.run(psi, XEFF_OP).result().values
            exp_values['Xeff'].append(x_result[0])
    
    ts = calculate_timesteps(T, max_d)
    exp_values['t'] = ts
    
    if show and max_d > 0:
        plot_expectation_values(ts, exp_values, operator, savefile_path=savefile_path, savefig=savefig)
    
    return exp_values

def run_one_job(data, shots, d, T, dJ, backend, run_type='sampler', op=None, one_qubits=[0, 2, 4, 6]):
    """
    Submit a single quantum circuit job to IBM Quantum for execution.
    
    This function constructs an appropriate quantum circuit based on the specified
    operator, optimizes it for the target backend, and submits it for execution.
    
    Parameters:
        data (list): Input data for circuit construction
        shots (int): Number of measurement shots
        d (int): Time step index
        T (float): Total evolution time
        dJ (float): Coupling parameter
        backend (str): IBM Quantum backend to use
        run_type (str, optional): Runtime primitive to use ('sampler' or 'estimator') (default: 'sampler')
        op (str, optional): Operator to measure 
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
                 None: no expecation value calculated (default)
        one_qubits (list, optional): Indices of qubits initialized to |1⟩ (default: [0, 2, 4, 6])
    
    Returns:
        str: Job ID assigned by IBM Quantum
    
    Notes:
        - For 'Xeff' operator, estimator is used regardless of run_type
        - Different circuit factories are used depending on the operator
    """
    # Force estimator for Xeff operator
    if op == 'Xeff':
        run_type = 'estimator'
    
    dt = T / d
    
    # Select appropriate circuit based on operator
    circuit_factories = {
        'singlet': circuit8qubits_psiU00,
        'Zeff': circuit8qubits_measureZ,
        'Xeff': circuit8qubits_projector
    }
    
    circuit_factory = circuit_factories.get(op, circuit8qubits_projector)
    actual_circuit = circuit_factory(data, d, dJ, dt, one_qubits)
    
    # Optimize circuit for the backend
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pass_manager.run(actual_circuit)
    
    if run_type == 'sampler':
        sampler = Sampler(mode=backend)
        job = sampler.run([(isa_circuit)], shots=shots)
    else:  # estimator
        estimator = Estimator(mode=backend)
        estimator.options.default_shots = shots
        operator = get_operator(op)
        mapped_op = operator.apply_layout(isa_circuit.layout)
        job = estimator.run([(isa_circuit, mapped_op)])
    
    print(f">>> Job ID: {job.job_id()}")
    return job.job_id()

def run_many_jobs(data, backend, max_d, T, dJ=0.5, shots=1000, run_type='sampler', op=None, one_qubits=[0, 2, 4, 6]):
    """
    Submit multiple quantum circuit jobs to IBM Quantum for parallel execution.
    
    This function constructs a series of quantum circuits for different time steps,
    optimizes them for the target backend, and submits them as a batch for execution.
    
    Parameters:
        data (list): Input data for circuit construction
        backend (str): IBM Quantum backend to use
        max_d (int): Maximum number of time steps
        T (float): Total evolution time
        dJ (float, optional): Coupling parameter (default: 0.5)
        shots (int, optional): Number of measurement shots per circuit (default: 1000)
        run_type (str, optional): Runtime primitive to use ('sampler' or 'estimator') (default: 'sampler')
        op (str, optional): Operator to measure 
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
                None: no expecation value calculated 
        one_qubits (list, optional): Indices of qubits initialized to |1⟩ (default: [0, 2, 4, 6])
    
    Returns:
        str: Job ID assigned by IBM Quantum
    
    Notes:
        - For 'Xeff' operator, estimator is used regardless of run_type
        - Different circuit factories are used depending on the operator
        - Circuits for all time steps (0 to max_d) are submitted in a single batch
    """
    # Force estimator for Xeff operator
    if op == 'Xeff':
        run_type = 'estimator'
    
    dt = T / max_d
    
    # Select appropriate circuit based on operator
    circuit_factories = {
        'singlet': circuit8qubits_psiU00,
        'Zeff': circuit8qubits_measureZ,
        'Xeff': circuit8qubits_projector
    }
    
    circuit_factory = circuit_factories.get(op, circuit8qubits_projector)
    actual_circuits = [circuit_factory(data, d, dJ, dt, one_qubits) for d in range(max_d + 1)]
    
    # Optimize circuits for the backend
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuits = [pass_manager.run(circuit) for circuit in actual_circuits]
    
    if run_type == 'sampler':
        sampler = Sampler(mode=backend)
        job = sampler.run(isa_circuits, shots=shots)
    else:  # estimator
        estimator = Estimator(mode=backend)
        estimator.options.default_shots = shots
        operator = get_operator(op)
        pubs = [(circuit, operator.apply_layout(circuit.layout)) for circuit in isa_circuits]
        job = estimator.run(pubs)
    
    print(f">>> Job ID: {job.job_id()}")
    return job.job_id()

def get_results(job_id, op, max_d, T, shots, run_type='sampler', plot=False):
    """
    Retrieve and process results from a completed IBM Quantum job.
    
    This function connects to IBM Quantum service, retrieves job results, and
    processes them to calculate expectation values for the specified operator.
    
    Parameters:
        job_id (str): Job ID from a previously submitted job
        op (str): Operator that was measured 
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
                None: no expecation value calculated 
        max_d (int): Maximum number of time steps
        T (float): Total evolution time
        shots (int): Number of measurement shots used
        run_type (str, optional): Runtime primitive used ('sampler' or 'estimator') (default: 'sampler')
        plot (bool, optional): Whether to display plots (default: False)
    
    Returns:
        tuple: (exps, ts, results)
            - exps (list): Calculated expectation values
            - ts (list): Time points
            - results: Raw results from IBM Quantum
    
    Notes:
        - Requires interactive input of API token
        - Processes results differently based on the operator and run_type
    """
    api_token = input("api token:")
    service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
    
    dt = T / max_d
    job = service.job(job_id)
    results = job.result()
    
    # Force estimator for Xeff operator
    if op == 'Xeff':
        run_type = 'estimator'
    
    exps = []
    
    if run_type == 'sampler':
        for result in results:
            key = next(iter(result.data))
            count = result.data[key].get_counts()
            
            if op == 'singlet':
                exp = count.get('01', 0) / shots
            elif op == 'Zeff':
                if "01" in count:
                    exp = (count.get('10', 0) - count.get('01', 0)) / shots
                else:
                    exp = count.get('10', 0) / shots
            
            exps.append(exp)
    else:  # estimator
        for result in results:
            exp = result.data.evs
            exps.append(exp)
    
    ts = calculate_timesteps(T, max_d + 1)
    
    if plot:
        plot_expectation_values(ts, exps, op)
    
    return exps, ts, results

def full_IBM_run(data, backend, max_d, T, dJ=0.5, shots=1000, run_type='sampler', 
                op='singlet', one_qubits=[0, 2, 4, 6], compare=None, 
                savefile_path=None, plot=True, save=False):
    """
    Execute a complete quantum simulation workflow on IBM Quantum hardware.
    
    This function performs an end-to-end simulation: submitting jobs to IBM Quantum,
    retrieving results, calculating expectation values, optionally comparing with
    other simulation methods, and visualizing the results.
    
    Parameters:
        data (list): Input data for circuit construction
        backend (str): IBM Quantum backend to use
        max_d (int): Maximum number of time steps
        T (float): Total evolution time
        dJ (float, optional): Coupling parameter (default: 0.5)
        shots (int, optional): Number of measurement shots per circuit (default: 1000)
        run_type (str, optional): Runtime primitive to use ('sampler' or 'estimator') (default: 'sampler')
        op (str, optional): Operator to measure expectation value of
            Options:
                'singlet': the operator |s><s|, where |s> = 1/sqrt(2) * (|01>-|10>) (default)
                'Zeff': effective Z measurement, Zeff = |10><10|-|01><01|
                'Xeff': effective X measurement, xeff = |10><01|+|01><10|
        one_qubits (list, optional): Indices of qubits initialized to |1⟩ (default: [0, 2, 4, 6])
        compare (str, optional): Comparison mode ('statevector', 'noiseless', or 'both')
        savefile_path (str, optional): Path prefix for saving results
        plot (bool, optional): Whether to display plots (default: True)
        save (bool, optional): Whether to save results and figures to disk (default: False)
    
    Returns:
        tuple: (exps, ts, results, sv_evals, noiseless_evals)
            - exps (list): Calculated expectation values from IBM Quantum
            - ts (list): Time points
            - results: Raw results from IBM Quantum
            - sv_evals (dict or None): State vector simulation results if requested
            - noiseless_evals (dict or None): Noiseless simulation results if requested
    
    Notes:
        - This is the highest-level function that orchestrates the entire workflow
        - Results can be compared with ideal statevector and/or noiseless simulations
        - Results and figures can be saved to disk for later analysis
    """
    job_id = run_many_jobs(data, backend, max_d, T, dJ, shots, run_type, op, one_qubits)
    exps, ts, results = get_results(job_id, op, max_d, T, shots, run_type)
    
    if save:
        np.save(f"{savefile_path}_results_{job_id}.npy", np.array([exps, ts]))
    
    sv_evals = None
    noiseless_evals = None
    
    if compare in ['statevector', 'both']:
        sv_evals, _ = statevector_simulate(
            data, savefile_path, circuit=circuit8qubits_statevector,
            T=T, dJ=dJ, d=max_d+1, one_qubits=one_qubits,
            operator=op, show=False, savefig=save
        )
    
    if compare in ['noiseless', 'both']:
        noiseless_evals = noiseless_simulate(
            data, savefile_path, T, dJ, max_d+1, shots=shots,
            method='direct', one_qubits=one_qubits,
            operator=op, show=False, savefig=save
        )
    
    if plot:
        plot_expectation_values(ts, exps, op, compare, sv_evals, noiseless_evals, savefile_path, savefig=save)
    
    return exps, ts, results, sv_evals, noiseless_evals