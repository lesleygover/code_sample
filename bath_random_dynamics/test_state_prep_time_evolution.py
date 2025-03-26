import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit, QuantumRegister
"""
Tests for the modules state_prep and time_evolution
"""
#########
#test state_prep.py
#########

from state_prep import validate_Hl_pauli_data, exp_pauli, state_prep


class TestValidatePauliData(unittest.TestCase):
    """Tests for the validate_pauli_data function."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        # Valid data with different Pauli operators
        data = [
            [0.5, 1, 2, 3],  # [coef, X, Y, Z]
            [0.25, 0, 1, 2]  # [coef, I, X, Y]
        ]
        # Should not raise any exceptions
        validate_Hl_pauli_data(data, 0)
        validate_Hl_pauli_data(data, 1)
    
    def test_empty_data(self):
        """Test validation with empty data."""
        with self.assertRaises(ValueError):
            validate_Hl_pauli_data([], 0)
    
    def test_invalid_data_type(self):
        """Test validation with invalid data types."""
        with self.assertRaises(ValueError):
            validate_Hl_pauli_data("not a list", 0)
        
        with self.assertRaises(ValueError):
            validate_Hl_pauli_data([{"not": "a list"}], 0)
    
    def test_out_of_range_element(self):
        """Test validation with out-of-range element index."""
        data = [[0.5, 1, 2, 3]]
        
        with self.assertRaises(IndexError):
            validate_Hl_pauli_data(data, 1)  # Only one element exists at index 0
        
        with self.assertRaises(IndexError):
            validate_Hl_pauli_data(data, -1)  # Negative index
    
    def test_invalid_pauli_index(self):
        """Test validation with invalid Pauli indices."""
        # Pauli index should be 0, 1, 2, or 3
        data = [[0.5, 1, 4, 3]]  # 4 is invalid
        
        with self.assertRaises(ValueError):
            validate_Hl_pauli_data(data, 0)
        
        data = [[0.5, -1, 2, 3]]  # -1 is invalid
        
        with self.assertRaises(ValueError):
            validate_Hl_pauli_data(data, 0)


class TestExpPauli(unittest.TestCase):
    """Tests for the exp_pauli function."""
    
    def test_invalid_output_type(self):
        """Test exp_pauli with invalid output type."""
        data = [[0.5, 1, 2, 3]]
        
        with self.assertRaises(ValueError):
            exp_pauli(data, 0, output_type="invalid")
    
    def test_identity_only(self):
        """Test exp_pauli with only identity operators."""
        data = [[0.5, 0, 0, 0]]  # All identity (I) operators
        
        # Should return circuit with no operations
        result_gate = exp_pauli(data, 0, 'gate')
        result_circuit = exp_pauli(data, 0, 'circuit')
        
        self.assertEqual(result_circuit.size(), 0, "Circuit should have no operations for identity-only Paulis")
        
    def test_x_rotation(self):
        """Test exp_pauli with X rotation on a single qubit."""
        data = [[np.pi/4, 1]]  # Equivalent to Rx(pi/2) on qubit 0
        
        circuit = exp_pauli(data, 0, 'circuit')
        self.assertGreater(circuit.size(), 0, "Circuit should have operations for X rotation")
        
        # Check the first and last gate are Hadamard
        gates = circuit.data
        self.assertEqual(gates[0][0].name, 'h', "First gate should be Hadamard for X rotation")
        self.assertEqual(gates[-1][0].name, 'h', "Last gate should be Hadamard for X rotation")
    
    def test_y_rotation(self):
        """Test exp_pauli with Y rotation on a single qubit."""
        data = [[np.pi/4, 2]]  # Equivalent to Ry(pi/2) on qubit 0
        
        circuit = exp_pauli(data, 0, 'circuit')
        
        # Check the first gate is Rx(pi/2) and last gate is Rx(-pi/2)
        gates = circuit.data
        self.assertEqual(gates[0][0].name, 'rx', "First gate should be rx for Y rotation")
        self.assertEqual(gates[-1][0].name, 'rx', "Last gate should be rx for Y rotation")
    
    def test_z_rotation(self):
        """Test exp_pauli with Z rotation on a single qubit."""
        data = [[np.pi/4, 3]]  # Equivalent to Rz(pi/2) on qubit 0
        
        circuit = exp_pauli(data, 0, 'circuit')
        
        # Should have just the Rz gate
        gates = circuit.data
        self.assertEqual(gates[0][0].name, 'rz', "Should have rz gate for Z rotation")
    
    def test_multi_qubit_pauli(self):
        """Test exp_pauli with multi-qubit Pauli string."""
        data = [[0.25, 1, 2, 3]]  # X⊗Y⊗Z
        
        circuit = exp_pauli(data, 0, 'circuit')
        
        # Check circuit has CNOT gates for entanglement
        gate_names = [gate[0].name for gate in circuit.data]
        self.assertIn('cx', gate_names, "Multi-qubit Pauli should have CNOT gates")
    
    def test_output_types(self):
        """Test different output types of exp_pauli."""
        data = [[0.5, 1, 2, 3]]
        
        gate = exp_pauli(data, 0, 'gate')
        circuit = exp_pauli(data, 0, 'circuit')
        
        self.assertTrue(hasattr(gate, 'name'), "Gate output should have a name attribute")
        self.assertIsInstance(circuit, QuantumCircuit, "Circuit output should be a QuantumCircuit")


class TestStatePrep(unittest.TestCase):
    """Tests for the state_prep function."""
    
    def test_invalid_data(self):
        """Test state_prep with invalid data."""
        with self.assertRaises(ValueError):
            state_prep([], 'gate')
        
        with self.assertRaises(ValueError):
            state_prep("not a list", 'gate')
        
        with self.assertRaises(ValueError):
            state_prep([[]], 'gate')
    
    def test_invalid_output_type(self):
        """Test state_prep with invalid output type."""
        data = [[0.5, 1, 2, 3]]
        
        with self.assertRaises(ValueError):
            state_prep(data, output_type="invalid")
    
    def test_output_types(self):
        """Test different output types of state_prep."""
        data = [[0.5, 1, 2, 3], [0.25, 3, 2, 1]]
        
        gate = state_prep(data, 'gate')
        circuit = state_prep(data, 'circuit')
        decomposed = state_prep(data, 'decomposed_circuit')
        
        self.assertTrue(hasattr(gate, 'name'), "Gate output should have a name attribute")
        self.assertIsInstance(circuit, QuantumCircuit, "Circuit output should be a QuantumCircuit")
        self.assertIsInstance(decomposed, QuantumCircuit, "Decomposed circuit should be a QuantumCircuit")
        
        # Decomposed circuit should have more gates than composite circuit
        self.assertGreater(decomposed.size(), circuit.size(), 
                          "Decomposed circuit should have more gates than composite circuit")
    
    
    def test_multiple_pauli_strings(self):
        """Test state_prep with multiple Pauli strings."""
        data = [
            [0.5, 1, 2, 3],   # X⊗Y⊗Z with θ=0.5
            [0.25, 3, 2, 1],  # Z⊗Y⊗X with θ=0.25
            [0.1, 0, 3, 0]    # I⊗Z⊗I with θ=0.1
        ]
        
        circuit = state_prep(data, 'circuit')
        
        # Check that the circuit contains the expected number of instructions
        # Each Pauli string should contribute at least one gate
        self.assertGreaterEqual(len(circuit.data), len(data), 
                               "Circuit should have at least one instruction per Pauli string")


class TestCircuitEquivalence(unittest.TestCase):
    """Tests to verify circuit equivalence and correctness."""
    
    def test_x_gate_equivalence(self):
        """Test that exp(i*pi/4*X) is equivalent to Rx(pi/2)."""
        # Create test circuit with our exp_pauli
        data = [[np.pi/4, 1]]  # X rotation with θ=pi/4
        pauli_circuit = exp_pauli(data, 0, 'circuit')
        
        # Create reference circuit with direct Rx
        qr = QuantumRegister(1)
        reference = QuantumCircuit(qr)
        reference.rx(-np.pi/2, 0)
        
        # Compare unitary matrices (approximately)
        try:
            from qiskit.quantum_info import Operator
            pauli_unitary = Operator(pauli_circuit).data
            reference_unitary = Operator(reference).data
            
            np.testing.assert_almost_equal(pauli_unitary, reference_unitary, decimal=10,
                                         err_msg="exp(i*pi/4*X) should be equivalent to Rx(-pi/2)")
        except ImportError:
            self.skipTest("qiskit.quantum_info not available for unitary comparison")
    
    def test_z_gate_equivalence(self):
        """Test that exp(i*pi/4*Z) is equivalent to Rz(-pi/2)."""
        # Create test circuit with our exp_pauli
        data = [[np.pi/4, 3]]  # Z rotation with θ=pi/4
        pauli_circuit = exp_pauli(data, 0, 'circuit')
        
        # Create reference circuit with direct Rz
        qr = QuantumRegister(1)
        reference = QuantumCircuit(qr)
        reference.rz(-np.pi/2, 0)
        
        # Compare unitary matrices (approximately)
        try:
            from qiskit.quantum_info import Operator
            pauli_unitary = Operator(pauli_circuit).data
            reference_unitary = Operator(reference).data
            
            np.testing.assert_almost_equal(pauli_unitary, reference_unitary, decimal=10,
                                         err_msg="exp(-i*pi/4*Z) should be equivalent to Rz(pi/2)")
        except ImportError:
            self.skipTest("qiskit.quantum_info not available for unitary comparison")



#########
#test time_evolution.py
#########

"""
Unit tests for the time_evolution module.

This test suite covers the core functionality of the refactored time_evolution.py
including input validation, error handling, and basic functionality.
"""

# Import the module to be tested
import time_evolution as te
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XXPlusYYGate, PauliEvolutionGate


class TestPauliDecomposition(unittest.TestCase):
    """Test cases for the Pauli decomposition function."""

    def test_identity_matrix(self):
        """Test decomposition of identity matrix."""
        matrix = np.array([[1, 0], [0, 1]])
        coef, index = te.decomp_p(matrix)
        self.assertAlmostEqual(coef, 1.0)
        self.assertEqual(index, 0)  # Identity matrix

    def test_x_matrix(self):
        """Test decomposition of X Pauli matrix."""
        matrix = np.array([[0, 1], [1, 0]])
        coef, index = te.decomp_p(matrix)
        self.assertAlmostEqual(coef, 1.0)
        self.assertEqual(index, 1)  # X matrix

    def test_y_matrix(self):
        """Test decomposition of Y Pauli matrix."""
        matrix = np.array([[0, -1j], [1j, 0]])
        coef, index = te.decomp_p(matrix)
        self.assertAlmostEqual(coef, 1.0)
        self.assertEqual(index, 2)  # Y matrix

    def test_z_matrix(self):
        """Test decomposition of Z Pauli matrix."""
        matrix = np.array([[1, 0], [0, -1]])
        coef, index = te.decomp_p(matrix)
        self.assertAlmostEqual(coef, 1.0)
        self.assertEqual(index, 3)  # Z matrix

    def test_scaled_matrix(self):
        """Test decomposition of scaled Pauli matrix."""
        matrix = np.array([[0, 2], [2, 0]])  # 2*X
        coef, index = te.decomp_p(matrix)
        self.assertAlmostEqual(coef, 2.0)
        self.assertEqual(index, 1)  # X matrix

    def test_invalid_matrix_size(self):
        """Test error handling for invalid matrix sizes."""
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            te.decomp_p(matrix)


class TestTimeEvolve(unittest.TestCase):
    """Test cases for the time_evolve function."""

    def setUp(self):
        """Set up test data."""
        # Create a simple test data structure
        self.test_data = [
            [
                [
                    [
                        [[1]],np.array([[0, 1], [1, 0]])
                    ]
                ]
            ]
        ]
        
        # Create a two-qubit gate test data
        self.two_qubit_data = [
            [  
                [  
                    [  
                        [[1, 2]],  # Two-qubit gate on qubits 1 and 2
                        None  # Not used for two-qubit gates
                    ]
                    
                ]
            ]
        ] 

    def test_invalid_qubits(self):
        """Test error handling for invalid qubit count."""
        with self.assertRaises(ValueError):
            te.time_evolve(self.test_data, 0, 0.1)
        
        with self.assertRaises(ValueError):
            te.time_evolve(self.test_data, -1, 0.1)

    def test_invalid_dt(self):
        """Test error handling for invalid time step."""
        with self.assertRaises(ValueError):
            te.time_evolve(self.test_data, 1, "not a number")

    def test_invalid_output_type(self):
        """Test error handling for invalid output type."""
        with self.assertRaises(ValueError):
            te.time_evolve(self.test_data, 1, 0.1, output_type="invalid")

    def test_single_qubit_gate(self):
        """Test adding a single-qubit gate to the circuit."""
        result = te.time_evolve(self.test_data, 2, 0.1, output_type="circuit")
        
        # Check if the circuit has the correct number of qubits
        self.assertEqual(result.num_qubits, 2)
        
        # Check if an rx gate was added
        has_rx = any(instr.operation.name == 'rx' for instr in result.data)
        self.assertTrue(has_rx)

    def test_two_qubit_gate(self):
        """Test adding a two-qubit gate to the circuit."""
        result = te.time_evolve(self.two_qubit_data, 3, 0.1, output_type="circuit")
        
        # Check if the circuit has the correct number of qubits
        self.assertEqual(result.num_qubits, 3)
        
        # Check if an XXPlusYYGate was added
        has_xxplusyy = any(isinstance(instr.operation, XXPlusYYGate) for instr in result.data)
        self.assertTrue(has_xxplusyy)

    def test_out_of_range_qubit(self):
        """Test error handling for qubit indices out of range."""
        invalid_data = [
            [
                [
                    [
                        [[10]],  # Qubit 10 is out of range for a 3-qubit system
                    ],
                    np.array([[0, 1], [1, 0]])
                ]
            ]
        ]
        
        with self.assertRaises(ValueError):
            te.time_evolve(invalid_data, 3, 0.1)


class TestInteractionOp(unittest.TestCase):
    """Test cases for the interaction_op function."""

    def test_invalid_dt(self):
        """Test error handling for invalid time step."""
        with self.assertRaises(ValueError):
            te.interaction_op("not a number", 1.0)

    def test_invalid_j(self):
        """Test error handling for invalid interaction strength."""
        with self.assertRaises(ValueError):
            te.interaction_op(0.1, "not a number")

    def test_invalid_output_type(self):
        """Test error handling for invalid output type."""
        with self.assertRaises(ValueError):
            te.interaction_op(0.1, 1.0, output_type="invalid")

    def test_circuit_creation(self):
        """Test creation of the interaction operator circuit."""
        result = te.interaction_op(0.1, 1.0, output_type="circuit")
        
        # Check if the circuit has exactly 2 qubits
        self.assertEqual(result.num_qubits, 2)
        
        # Check if a PauliEvolutionGate was added
        has_evolution_gate = any(isinstance(instr.operation, PauliEvolutionGate) for instr in result.data)
        self.assertTrue(has_evolution_gate)


class TestRTE(unittest.TestCase):
    """Test cases for the RTE function."""

    def setUp(self):
        """Set up test data."""
        # Create a simple test data structure
        self.test_data = {
            'rand_circ_ls': [
                [  
                    [  
                        [  
                            [[1]],  
                        ],
                        np.array([[0, 1], [1, 0]])  # X matrix
                    ]
                ]
            ],
            'rand_circ_ls_rvs': [
                [  
                    [  
                        [  
                            [[1]],  # Single qubit gate on qubit 1
                        ],
                        np.array([[0, 1], [1, 0]])  # X matrix
                    ]
                ]
            ]
        }

    def test_invalid_qubits(self):
        """Test error handling for invalid qubit count."""
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, 3, 0.1, 1.0)  # Needs at least 4 qubits
        
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, -1, 0.1, 1.0)  # Negative qubit count

    def test_odd_qubits(self):
        """Test error handling for odd qubit count."""
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, 5, 0.1, 1.0)  # Odd number of qubits

    def test_missing_data_keys(self):
        """Test error handling for missing data keys."""
        with self.assertRaises(ValueError):
            te.RTE({'rand_circ_ls': []}, 4, 0.1, 1.0)  # Missing 'rand_circ_ls_rvs'
            
        with self.assertRaises(ValueError):
            te.RTE({'rand_circ_ls_rvs': []}, 4, 0.1, 1.0)  # Missing 'rand_circ_ls'

    def test_invalid_dt(self):
        """Test error handling for invalid time step."""
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, 4, "not a number", 1.0)

    def test_invalid_j(self):
        """Test error handling for invalid interaction strength."""
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, 4, 0.1, "not a number")

    def test_invalid_output_type(self):
        """Test error handling for invalid output type."""
        with self.assertRaises(ValueError):
            te.RTE(self.test_data, 4, 0.1, 1.0, output_type="invalid")

    @patch('time_evolution.time_evolve')
    @patch('time_evolution.interaction_op')
    def test_circuit_creation(self, mock_interaction_op, mock_time_evolve):
        """Test creation of the RTE circuit with mocks."""
        # Configure mocks
        mock_time_evolve.return_value = QuantumCircuit(1).to_instruction()
        mock_interaction_op.return_value = QuantumCircuit(2).to_instruction()
        
        # Call RTE
        result = te.RTE(self.test_data, 4, 0.1, 1.0, output_type="circuit")
        
        # Check if the circuit has the correct number of qubits
        self.assertEqual(result.num_qubits, 4)
        
        # Verify that time_evolve was called twice
        self.assertEqual(mock_time_evolve.call_count, 2)
        
        # Verify that interaction_op was called three times
        self.assertEqual(mock_interaction_op.call_count, 3)

if __name__ == '__main__':
    unittest.main()