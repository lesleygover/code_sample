import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

"""
Quantum Time Evolution Module

contains functions for simulation of bath dynamics and interaction with the state.
It implements Trotterized time evolution with XX+YY gates and single qubit rotations.
"""

def decomp_p(matrix):
    """
    Calculate which Pauli matrix and its coefficient from a 2x2 matrix.
    
    Parameters:
        matrix (np.ndarray): A 2x2 numpy array representing a quantum operation
     
    Returns:
        A tuple containing:
            - coefficient: The real coefficient of the Pauli matrix
            - index: The index representing the Pauli matrix (0=I, 1=X, 2=Y, 3=Z)
         
    Raises:
        ValueError: If the input matrix is not 2x2
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Input matrix must be 2x2, got {matrix.shape}")
        
    i_comp = (matrix[0, 0] + matrix[1, 1]) / 2
    z_comp = (matrix[0, 0] - matrix[1, 1]) / 2
    x_comp = (matrix[0, 1] + matrix[1, 0]) / 2
    y_comp = (-matrix[0, 1] + matrix[1, 0]) * -1j / 2
    
    comps = [i_comp, x_comp, y_comp, z_comp]
    
    for i, comp in enumerate(comps):
        if np.round(comp, 5) != 0:
            return np.real(comp), i

def time_evolve(data, qubits, dt, output_type = 'gate'):
    """
    Produces circuit of Trotterised time evolution consisting of exp(θ(XX+YY)) and single qubit rotation gates.
    
    Parameters:
        data (list): Array of locations of gates and rotation angles
        qubits (int): Number of qubits required
        dt (float): The time evolution step size
        output_type (str): Determines the output format:
            'gate': Returns a composite gate (default)
            'circuit': Returns the full circuit
            'decomposed_circuit': Returns the fully decomposed circuit
      
    Returns:
        The time evolution circuit in the specified format
       
    Raises:
        ValueError: If qubits < 1, dt is invalid, or output_type is invalid
    """
    # Input validation
    if not isinstance(qubits, int) or qubits < 1:
        raise ValueError(f"Number of qubits must be a positive integer, got {qubits}")
    
    if not isinstance(dt, (int, float)):
        raise ValueError(f"Time step dt must be a number, got {dt}")
        
    if output_type not in ['gate', 'circuit', 'decomposed_circuit']:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'gate', 'circuit', or 'decomposed_circuit'")
    
    # Build circuit
    qr = QuantumRegister(qubits)
    circuit = QuantumCircuit(qr, name='evolution')
    
    # Add gates

    for i in range(len(data[0])):
        # Two-qubit gate (XX+YY)
        if len(data[0][i][0][0][0]) == 2:
            q1 = data[0][i][0][0][0][0] - 1
            q2 = data[0][i][0][0][0][1] - 1
            
            # Validate qubit indices
            if not (0 <= q1 < qubits and 0 <= q2 < qubits):
                raise ValueError(f"Qubit indices {q1},{q2} out of range for {qubits} qubits")
                
            circuit.append(XXPlusYYGate(dt * 4), (qr[q1], qr[q2]))
            
        # Single-qubit gate
        elif len(data[0][i][0][0][0]) == 1:
            q = data[0][i][0][0][0][0] - 1
            
            # Validate qubit index
            if not (0 <= q < qubits):
                raise ValueError(f"Qubit index {q} out of range for {qubits} qubits")
                
            θ, P = decomp_p(data[0][i][0][1])
            
            if P == 1:
                circuit.rx(2 * θ * dt, qr[q])
            elif P == 2:
                circuit.ry(2 * θ * dt, qr[q])
            elif P == 3:
                circuit.rz(2 * θ * dt, qr[q])
        else:
            raise ValueError(f"Unexpected gate format: {data[0][i][0][0][0]}")

    
    # Convert to gate
    evol = circuit.to_instruction()
    
    # Return requested format
    if output_type == 'circuit':
        return circuit
    elif output_type == 'decomposed_circuit':
        return circuit.decompose()
    else:  # gate
        return evol


def interaction_op(dt, dJ, output_type = 'gate'):
    """
    Produces interaction operator exp(-i(XX+YY+ZZ)dJdt).
    
    Parameters:
        dt (float): Time step
        dJ (float): Interaction strength
        output_type (str): Determines the output format:
                    'gate': Returns a composite gate (default)
                    'circuit': Returns the full circuit
                    'decomposed_circuit': Returns the fully decomposed circuit
        
    Returns:
        The interaction operator in the specified format
        
    Raises:
        ValueError: If dt or dJ are invalid, or output_type is invalid
    """
    # Input validation
    if not isinstance(dt, (int, float)):
        raise ValueError(f"Time step dt must be a number, got {dt}")
        
    if not isinstance(dJ, (int, float)):
        raise ValueError(f"Interaction strength dJ must be a number, got {dJ}")
        
    if output_type not in ['gate', 'circuit', 'decomposed_circuit']:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'gate', 'circuit', or 'decomposed_circuit'")
    
    # Build circuit
    qr = QuantumRegister(2)
    circuit = QuantumCircuit(qr, name=f'interaction op: dt = {dt:.2f}, dJ = {dJ:.2f}')
    
    # Create and append the Pauli evolution gate
    operator = SparsePauliOp(["XX", "YY", "ZZ"])
    evo = PauliEvolutionGate(operator, time=dt * dJ)
    circuit.append(evo, (qr[0], qr[1]))
    
    # Convert to gate
    int_op = circuit.to_instruction()
    
    # Return requested format
    if output_type == 'circuit':
        return circuit
    elif output_type == 'decomposed_circuit':
        return circuit.decompose()
    else:  # gate
        return int_op


def RTE(data, qubits, dt, dJ, output_type = 'gate'):
    """
    Performs full time evolution chunk by combining local time evolution with interaction operations.
    
    Parameters:
        data (dict): The data dictionary for the time evolution bath containing:
              - 'rand_circ_ls': Random circuit list for the first half
              - 'rand_circ_ls_rvs': Random circuit list for the second half
        qubits (int): Number of qubits
        dt (float): Time step
        dJ: Interaction strength
        output_type (str): Determines the output format:
                    'gate': Returns a composite gate (default)
                    'circuit': Returns the full circuit
                    'decomposed_circuit': Returns the fully decomposed circuit
        
    Returns:
        The full time evolution in the specified format
        
    Raises:
        ValueError: If inputs are invalid or required data keys are missing
    """
    # Input validation
    if not isinstance(qubits, int) or qubits < 4:
        raise ValueError(f"Number of qubits must be a positive integer >= 4, got {qubits}")
    
    if not isinstance(dt, (int, float)):
        raise ValueError(f"Time step dt must be a number, got {dt}")
        
    if not isinstance(dJ, (int, float)):
        raise ValueError(f"Interaction strength dJ must be a number, got {dJ}")
        
    if output_type not in ['gate', 'circuit', 'decomposed_circuit']:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'gate', 'circuit', or 'decomposed_circuit'")
    
    # Check for required data keys
    required_keys = ['rand_circ_ls', 'rand_circ_ls_rvs']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in data dictionary")
    
    # Ensure qubits is even for proper splitting
    if qubits % 2 != 0:
        raise ValueError(f"Number of qubits must be even, got {qubits}")
    
    # Build circuit
    qr = QuantumRegister(qubits)
    circuit = QuantumCircuit(qr, name=f'RTE: dt = {dt:.2f}, dJ = {dJ:.2f}')
    
    # Calculate indices
    mid = qubits // 2
    mid_minus_1 = mid - 1
    mid_minus_2 = mid - 2
    mid_plus_1 = mid + 1
    
    # Apply time evolution to first half
    circuit.append(
        time_evolve(data['rand_circ_ls'], mid_minus_1, dt, output_type='gate'),
        qr[:mid_minus_1]
    )
    
    # Apply time evolution to second half
    circuit.append(
        time_evolve(data['rand_circ_ls_rvs'], mid_minus_1, dt, output_type='gate'),
        qr[mid_plus_1:]
    )
    
    # Apply interaction operators
    circuit.append(
        interaction_op(dt, 1, output_type='decomposed_circuit'),
        (qr[mid_minus_1], qr[mid])
    )  # Ulocal
    
    circuit.append(
        interaction_op(dt, dJ, output_type='decomposed_circuit'),
        (qr[mid_minus_2], qr[mid_minus_1])
    )
    
    circuit.append(
        interaction_op(dt, dJ, output_type='decomposed_circuit'),
        (qr[mid], qr[mid_plus_1])
    )
    
    # Convert to gate
    rte = circuit.to_instruction()
    
    # Return requested format
    if output_type == 'circuit':
        return circuit
    elif output_type == 'decomposed_circuit':
        return circuit.decompose()
    else:  # gate
        return rte
