'''
Currently contains:
- State Ansatzs:
    - Superconducting one used for Google's device for time evolution (`StateAnsatzXZ`)
    - Ion Trap one inspired by QSCOUT's native gates (`StateAnsatzQscout`)
- Circuit generation:
    - generate iMPS circuits
    - overlap calculation circuits
    - time evolution circuits
'''
import cirq
from cirq import two_qubit_matrix_to_ion_operations
from jaqalpaq.core import circuitbuilder
import numpy as np
from qscout.v1.std.jaqal_gates import ALL_GATES as std
from qscout.v1.zz.jaqal_gates import ACTIVE_GATES as zz
from quten.transpile import buildStatementsFromCirq


std.update(zz) # Add ZZ gate to QSCOUT gate models


class StateAnsatzXZ(cirq.Gate):
    '''
    Circuit representing single state unitary `U` restricted to `rx`,`rz` and
    `CNOT` rotations.
    '''
    param_len = 8

    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.rx(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[1]),
                cirq.rx(self.Psi[3]).on(qubits[1]),
                cirq.CNOT(*qubits),
                cirq.rz(self.Psi[4]).on(qubits[0]),
                cirq.rx(self.Psi[5]).on(qubits[0]),
                cirq.rz(self.Psi[6]).on(qubits[1]),
                cirq.rx(self.Psi[7]).on(qubits[1]),
                cirq.CNOT(*qubits),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U', 'U']


class StateAnsatzQscout(cirq.Gate):
    '''
    Circuit representing single state unitary `U` inspired by native gateset of
    QSCOUT's ion trap device.
    '''
    param_len = 12

    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
            cirq.PhasedXPowGate(phase_exponent=self.Psi[0],
                                exponent=self.Psi[1],
                                global_shift=-0.5).on(qubits[0]),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[2],
                                exponent=self.Psi[3],
                                global_shift=-0.5).on(qubits[1]),
            cirq.ZZPowGate(exponent=self.Psi[4],
                           global_shift=-0.5).on(*qubits),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[5],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[0]),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[6],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[1]),
            cirq.ZZPowGate(exponent=self.Psi[7],
                           global_shift=-0.5).on(*qubits),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[8],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[0]),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[9],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[1]),
            cirq.ZPowGate(exponent=self.Psi[10]).on(qubits[0]),
            cirq.ZPowGate(exponent=self.Psi[11]).on(qubits[1])
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U', 'U']


class StateAnsatzReducedQscout(cirq.Gate):
    '''
    Circuit representing single state unitary `U` inspired by native gateset of
    QSCOUT's ion trap device.

    This is a reduced parameterisation with 8 parameters that works well for
    demonstrating the dynamical phase transition in the TFIM.
    '''
    param_len = 8

    def __init__(self, Psi):
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
            cirq.PhasedXPowGate(phase_exponent=self.Psi[0],
                                exponent=self.Psi[1],
                                global_shift=-0.5).on(qubits[0]),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[2],
                                exponent=self.Psi[3],
                                global_shift=-0.5).on(qubits[1]),
            cirq.ZZPowGate(exponent=self.Psi[4],
                           global_shift=-0.5).on(*qubits),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[5],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[0]),
            # cirq.PhasedXPowGate(phase_exponent=0.,
            #                    exponent=0.5,
            #                    global_shift=-0.5).on(qubits[1]),
            cirq.XPowGate(exponent=0.5, global_shift=-0.5).on(qubits[1]),
            cirq.ZZPowGate(exponent=self.Psi[6],
                           global_shift=-0.5).on(*qubits),
            cirq.PhasedXPowGate(phase_exponent=self.Psi[7],
                                exponent=0.5,
                                global_shift=-0.5).on(qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U', 'U']

    def decomposeJaqal(self, builder, qubits):
        q0 = qubits[0]
        q1 = qubits[1]
        rescaledPsi = [np.pi*p for p in self.Psi]

        builder.gate('R', q0, rescaledPsi[0], rescaledPsi[1])
        builder.gate('R', q1, rescaledPsi[2], rescaledPsi[3])
        builder.gate('ZZ', q0, q1, rescaledPsi[4])
        builder.gate('R', q0, rescaledPsi[5], 0.5*np.pi)
        builder.gate('R', q1, 0.0, 0.5*np.pi)
        builder.gate('ZZ', q0, q1, rescaledPsi[6])
        builder.gate('R', q1, rescaledPsi[7], 0.5*np.pi)

        return builder

    def decomposeJaqalInverse(self, builder, qubits):
        q0 = qubits[0]
        q1 = qubits[1]
        rescaledPsi = [np.pi*p for p in self.Psi]

        builder.gate('R', q1, rescaledPsi[7] + np.pi, 0.5*np.pi)
        builder.gate('ZZ', q0, q1, -1*rescaledPsi[6])
        builder.gate('R', q1, np.pi, 0.5*np.pi)
        builder.gate('R', q0, rescaledPsi[5]+np.pi, 0.5*np.pi)
        builder.gate('ZZ', q0, q1, -1*rescaledPsi[4])
        builder.gate('R', q1, rescaledPsi[2]+np.pi, rescaledPsi[3])
        builder.gate('R', q0, rescaledPsi[0]+np.pi, rescaledPsi[1])

        return builder


def generateMPSCircuit(θ, N, Theta=None, Ne=0, Q=None, Ansatz=StateAnsatzXZ,
                       offset=0):
    """
    Generate cirq MPS Circuit Ansatz with environment

    Parameters
    ----------
    θ: State parameterisations
    N: Number of sites
    Theta: Env parameterisation
    Ne: Number of environments
    Q: List of qubits to apply
    Ansatz: State Ansatz class to use
    offset: offset from environment side
    """
    if Q is None:
        Q = cirq.LineQubit.range(N + Ne + 1)

    c = cirq.Circuit()

    for i in range(Ne):
        ops = [cirq.decompose_once(Ansatz(Theta).on( Q[-2-i], Q[-1-i]))]
        c.append(ops)

    for i in range(N):
        ops = [cirq.decompose_once(Ansatz(θ).on(Q[-2-i-Ne-offset],
                                   Q[-1-i-Ne-offset]))]
        c.append(ops)
    return c


def generateMPSCircuitJaqal(θ, N, builder=None, θEnv=None, Ne=0, Q=None,
                            Ansatz=StateAnsatzXZ, offset=0):
    """
    Generate Jaqal circuit builder representing qMPS circuit with environment

    Parameters
    ----------
    θ: State parameterisations
    N: Number of sites
    Theta: Env parameterisation
    Ne: Number of environments
    Q: List of qubits to apply
    Ansatz: State Ansatz class to use
    offset: offset from environment side
    """
    if builder is not None:
        assert Q is not None, 'Need to pass qubits'

    if builder is None:
        builder = circuitbuilder.CircuitBuilder(native_gates=std)
        Q = builder.register('q', N + Ne + 1)
        builder.gate('prepare_all')

    qLen = len(Q)
    ParameterisedAnsatz = Ansatz(θEnv)
    for i in range(Ne):
        q0i = qLen-2-i
        q1i = qLen-1-i
        qubits = (Q[q0i], Q[q1i])
        builder = ParameterisedAnsatz.decomposeJaqal(builder, qubits)

    ParameterisedAnsatz = Ansatz(θ)
    for i in range(N):
        q0i = qLen-2-i-Ne-offset
        q1i = qLen-1-i-Ne-offset
        qubits = [Q[q0i], Q[q1i]]
        builder = ParameterisedAnsatz.decomposeJaqal(builder, qubits)
    return builder, Q


def generateMPSCircuitInverseJaqal(θ, N, builder=None, θEnv=None, Ne=0, Q=None,
                            Ansatz=StateAnsatzXZ, offset=0):
    """Generate MPS Circuit Ansatz with environment

    Parameters
    ----------
    θ: State parameterisations
    N: Number of sites
    Theta: Env parameterisation
    Ne: Number of environments
    Q: List of qubits to apply
    Ansatz: State Ansatz class to use
    offset: offset from environment side
    """
    if builder is not None:
        assert Q is not None, 'Need to pass qubits'

    if builder is None:
        builder = circuitbuilder.CircuitBuilder(native_gates=std)
        Q = builder.register('q', N + Ne + 1)
        builder.gate('prepare_all')

    qLen = len(Q)
    ParameterisedAnsatz = Ansatz(θ)
    for i in reversed(range(N)):
        q0i = qLen-2-i-Ne-offset
        q1i = qLen-1-i-Ne-offset
        qubits = [Q[q0i], Q[q1i]]
        builder = ParameterisedAnsatz.decomposeJaqalInverse(builder, qubits)

    ParameterisedAnsatz = Ansatz(θEnv)
    for i in reversed(range(Ne)):
        q0i = qLen-2-i
        q1i = qLen-1-i
        qubits = (Q[q0i], Q[q1i])
        builder = ParameterisedAnsatz.decomposeJaqalInverse(builder, qubits)

    return builder, Q


def generateOverlapCircuit(θA, θB, N, Ne=0, ψA=None, ψB=None, Q=None,
                           Ansatz=StateAnsatzXZ, offset=0):
    """
    Generate cirq overlap circuits with environments.
    """
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θB

    circuitA = generateMPSCircuit(θA, N, ψA, Ne, Q, Ansatz, offset)
    circuitB = generateMPSCircuit(θB, N, ψB, Ne, Q, Ansatz, offset)

    circuit = cirq.Circuit()
    circuit.append(circuitA)
    circuit.append(cirq.inverse(circuitB))
    return circuit

def generateOverlapCircuitJaqal(θA, θB, N, builder=None, Ne=0, ψA=None, ψB=None, Q=None,
                           Ansatz=StateAnsatzXZ, offset=0):
    """
    Generate jaqal circuit builder representing an overlap circuit.
    """
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θB

    if builder is not None:
        assert Q is not None, 'Need to pass qubits'

    if builder is None:
        builder = circuitbuilder.CircuitBuilder(native_gates=std)
        Q = builder.register('q', N + Ne + 1)
        builder.gate('prepare_all')

    builder, Q = generateMPSCircuitJaqal(θA, N, builder=builder, Q=Q,
                                         Ne=Ne, θEnv=ψA, Ansatz=Ansatz,
                                         offset=offset)
    builder, Q = generateMPSCircuitInverseJaqal(θB, N, builder=builder, Q=Q,
                                                Ne=Ne, θEnv=ψB, Ansatz=Ansatz,
                                                offset=offset)
    return builder, Q


class TimeEvolutionCircuitGenerator():
    def __init__(self, U, qubits=None, Ansatz=StateAnsatzXZ):
        self.Ansatz = Ansatz

        # Generate the U gate ops
        self._refQubits = cirq.NamedQubit.range(2, prefix='_ref')
        self.Uops = two_qubit_matrix_to_ion_operations(
            self._refQubits[0], self._refQubits[1],
            U
        )
        self.U = U
        self.qubits = qubits

    def _prepareULayer(self, N, qubits):
        ops = []
        for i in range(1, N*2 + 1, 2):
            qubit_map = {
                self._refQubits[0]: qubits[i],
                self._refQubits[1]: qubits[i+1]
            }

            ops.extend([op.transform_qubits(qubit_map) for op in self.Uops])
        return ops

    def _prepareJaqalULayer(self, N, builder, Q):
        qubits = cirq.LineQubit.range(2*N + 1)
        ops = []
        for i in range(1, N*2 + 1, 2):
            qubit_map = {
                self._refQubits[0]: qubits[i],
                self._refQubits[1]: qubits[i+1]
            }

            ops.extend([op.transform_qubits(qubit_map) for op in self.Uops])

        cirqCircuit = cirq.Circuit()
        cirqCircuit.append(ops)

        buildStatements = buildStatementsFromCirq(cirqCircuit)

        for statement in buildStatements:
            qubits = [Q[i] for i in statement['qubits']]
            builder.gate(statement['gateName'], *qubits, *statement['params'])
        return builder

    def generate(self, θA, θB, N, Ne=0):
        '''
        Generate a Cirq time evolution circuit

        Parameters
        ----------
        θA : np.array
            Params representing initial MPS state
        θB : np.array
            Params representing target MPS state
        N : int
            Number of time evolution unit cells. (Will have 2*N MPS sites)
        Ne : int
            Number of MPS sites in environment
        '''
        Q = self.qubits
        if Q is None:
            Q = cirq.LineQubit.range(N*2 + Ne + 1)

        ψA = θA
        circuitA = generateMPSCircuit(θA, N*2, ψA, Ne, Q, self.Ansatz)
        circuitB = generateMPSCircuit(θB, N*2, ψA, Ne, Q, self.Ansatz)
        ULayer = self._prepareULayer(N, qubits=Q)

        circuit = cirq.Circuit()
        circuit.append(circuitA)
        circuit.append(ULayer)
        circuit.append(cirq.inverse(circuitB))
        return circuit

    def generateJaqal(self, θA, θB, N, Ne=0):
        '''
        Generate a Jaqal time evolution circuit

        Parameters
        ----------
        θA : np.array
            Params representing initial MPS state
        θB : np.array
            Params representing target MPS state
        N : int
            Number of time evolution unit cells. (Will have 2*N MPS sites)
        Ne : int
            Number of MPS sites in environment
        '''
        builder = circuitbuilder.CircuitBuilder(native_gates=std)
        Q = builder.register('q', N*2 + Ne + 1)
        builder.gate('prepare_all')
        ψA = θA

        builder, Q = generateMPSCircuitJaqal(θA, N*2, builder, ψA, Ne, Q,
                                             self.Ansatz)
        builder = self._prepareJaqalULayer(N, builder, Q)
        builder, Q = generateMPSCircuitInverseJaqal(θB, N*2, builder=builder,
                                                    θEnv=ψA, Ne=Ne,
                                                    Q=Q, Ansatz=self.Ansatz)
        builder.gate('measure_all')
        return builder.build()