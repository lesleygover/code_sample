import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

"""
Quantum bath state preparation

Contains functions for generating quantum circuits and gates for preparing the system bath
Generates circuits using Pauli string exponentials of the form exp(-iθP), 
where P is a Pauli string operator and θ is the coefficient of the Pauli string.
"""

def validate_Hl_pauli_data(data, element_number):
    """
    Validates the format and content of Pauli string data.
    
    Parameters:
        data (list): List of lists where each inner list contains a coefficient followed by Pauli indices
        element_number (str): Index of the element to validate
    
    Raises:
        ValueError: If data format is invalid or Pauli indices are not in [0,1,2,3]
        IndexError: If element_number is out of range
    """
    if not isinstance(data, list) or not data:
        raise ValueError("Data must be a non-empty list of lists")
    
    if element_number < 0 or element_number >= len(data):
        raise IndexError(f"Element number {element_number} out of range (0-{len(data)-1})")
    
    element = data[element_number]
    if not isinstance(element, list) or len(element) < 2:
        raise ValueError(f"Element {element_number} must be a list with at least one coefficient and one Pauli index")
    
    # Validate Pauli indices (should be 0=I, 1=X, 2=Y, 3=Z)
    for i, pauli_idx in enumerate(element[1:]):
        if pauli_idx not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid Pauli index {pauli_idx} at position {i+1} in element {element_number}. Must be 0(I), 1(X), 2(Y), or 3(Z)")



def exp_pauli(data, element_number, output_type = 'gate'):
    """
    Produces a gate or circuit for exp(-iθP_i), where P_i is a Pauli string operator and θ is the coefficient.
    
    Parameters:
        data (list): List of lists where each inner list contains a coefficient followed by Pauli indices.
              Format: [[coef, p1, p2, ...], [coef, p1, p2, ...], ...]
              Where each pi is 0(I), 1(X), 2(Y), or 3(Z)
        element_number (int): Index of the element in data to process
        output_type (str): Type of output to return, either 'gate' or 'circuit' (default: 'gate')
    
    Returns:
        Either a Qiskit gate instruction or a full QuantumCircuit representing exp(-iθP_i)
        
    Raises:
        ValueError: If data format is invalid or output_type is not recognized
    """
    # Validate inputs
    validate_Hl_pauli_data(data, element_number)
    
    if output_type not in ['gate', 'circuit']:
        raise ValueError(f"Output type must be 'gate' or 'circuit', got '{output_type}'")
    
    # Define number of qubits
    qubits = len(data[element_number]) - 1
    
    # Get list of pauli strings in order
    paulis = [int(i) for i in data[element_number][1:]]
    
    # Define Pauli names for circuit labeling
    pauli_names = ['I', 'X', 'Y', 'Z']
    circuit_name = [pauli_names[i] for i in paulis]
    
    # Get coefficient θ
    θ = data[element_number][0]
    
    # Build quantum circuit
    qr = QuantumRegister(qubits)
    circuit = QuantumCircuit(qr, name=f'exp(-i·{θ:.2f}·{"·".join(circuit_name)})')

    # Identify qubits that need CNOTs (all non-identity Paulis)
    cnot_indices = [i for i, pauli in enumerate(paulis) if pauli != 0]
    
    if not cnot_indices:
        # If only identity operators, circuit is identity (no operation)
        if output_type == 'circuit':
            return circuit
        else:
            return circuit.to_instruction()
    
    # Add rotation gates to set as I,X,Y, or Z
    for i, pauli in enumerate(paulis):
        if pauli == 1:    # X basis
            circuit.h(qr[i])
        elif pauli == 2:  # Y basis
            circuit.rx(np.pi/2, qr[i])
        # No rotation needed for Z basis (3) or Identity (0)
    
    # Add CNOT chain for entanglement (if more than one non-identity Pauli)
    for i in range(len(cnot_indices) - 1):
        circuit.cx(qr[cnot_indices[i]], qr[cnot_indices[i+1]])
    
    # Add Rz gate
    if cnot_indices:
        circuit.rz(-2 * θ, qr[cnot_indices[-1]])
    
    # Add reverse CNOT chain
    for i in reversed(range(len(cnot_indices) - 1)):
        circuit.cx(qr[cnot_indices[i]], qr[cnot_indices[i+1]])
    
    # Add second set of rotation gates to set as I,X,Y, or Z
    for i, pauli in enumerate(paulis):
        if pauli == 1:    # X basis
            circuit.h(qr[i])
        elif pauli == 2:  # Y basis
            circuit.rx(-np.pi/2, qr[i])
        # No rotation needed for Z basis or Identity
    
    # Return appropriate output format
    if output_type == 'circuit':
        return circuit
    else:
        return circuit.to_instruction()

def state_prep(data,output_type = 'gate', reverse = False):
    """
    Produces a quantum gate/circuit for state preparation by combining multiple Pauli exponentials.
        The state preparation circuit is built from a sequence of Pauli exponentials
        exp(-iθP_i), where P_i is a Pauli string operator and θ is the coefficient.
    
    Parameters:
        data (list): List of lists where each inner list contains a coefficient followed by Pauli indices.
              Format: [[coef, p1, p2, ...], [coef, p1, p2, ...], ...]
              Where each pi is 0(I), 1(X), 2(Y), or 3(Z)
        output_type (str): Type of output to return (default: 'gate')
             Options:
             - 'gate': Returns a composite gate
             - 'circuit': Returns a circuit with composite gates
             - 'decomposed_circuit': Returns a fully decomposed circuit
        reverse (bool): If True, applies operations in reverse qubit order (default: False)
    
    Returns:
        Either a Qiskit gate instruction or a QuantumCircuit representing the state preparation circuit
        
    Raises:
        ValueError: If data format is invalid or output_type is not recognized
    """
    # Validate inputs
    if not isinstance(data, list) or not data:
        raise ValueError("Data must be a non-empty list of lists")
    
    if output_type not in ['gate', 'circuit', 'decomposed_circuit']:
        raise ValueError(f"Output type must be 'gate', 'circuit', or 'decomposed_circuit', got '{output_type}'")
    
    if not data[0] or len(data[0]) < 2:
        raise ValueError("First element of data must contain at least one coefficient and one Pauli index")

    # Get number of qubits from the first element
    qubits = len(data[0]) - 1
    
    # Build up circuit
    qr = QuantumRegister(qubits)
    circuit = QuantumCircuit(qr, name='state_prep')
    
    for i in range(len(data)):
        # Generate Pauli exponential gate
        pauli_gate = exp_pauli(data, i, 'gate')
        
        # Apply gate to circuit with proper qubit ordering
        if reverse:
            circuit.append(pauli_gate, reversed(list(qr)))
        else:
            circuit.append(pauli_gate, qr)
    
    # Return appropriate output format
    if output_type == 'circuit':
        return circuit
    elif output_type == 'decomposed_circuit':
        return circuit.decompose()
    else:  # 'gate'
        return circuit.to_instruction()
    