
import warnings
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import NormalDistribution
import qiskit
from typing import cast, Tuple, Dict, Callable
from qiskit_ibm_runtime import SamplerV2 as Sampler

def iae_construct_circuit(
    estimation_problem: EstimationProblem, k: int = 0, measurement: bool = False
) -> QuantumCircuit:
    r"""Construct the circuit :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

    The A operator is the unitary specifying the QAE problem and Q the associated Grover
    operator.

    Args:
        estimation_problem: The estimation problem for which to construct the QAE circuit.
        k: The power of the Q operator.
        measurement: Boolean flag to indicate if measurements should be included in the
            circuits.

    Returns:
        The circuit implementing :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
    """
    num_qubits = max(
        estimation_problem.state_preparation.num_qubits,
        estimation_problem.grover_operator.num_qubits,
    )
    circuit = QuantumCircuit(num_qubits, name="circuit")

    # add classical register if needed
    if measurement:
        c = ClassicalRegister(len(estimation_problem.objective_qubits))
        circuit.add_register(c)

    # add A operator
    circuit.compose(estimation_problem.state_preparation, inplace=True)

    # add Q^k
    if k != 0:
        circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

        # add optional measurement
    if measurement:
        # real hardware can currently not handle operations after measurements, which might
        # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
        circuit.barrier()
        circuit.measure(estimation_problem.objective_qubits, c[:])

    return circuit

def good_state_probability(
    problem: EstimationProblem,
    counts_dict: Dict[str, int],
) -> Tuple[int, float]:
    """Get the probability to measure '1' in the last qubit.

    Args:
        problem: The estimation problem, used to obtain the number of objective qubits and
            the ``is_good_state`` function.
        counts_dict: A counts-dictionary (with one measured qubit only!)

    Returns:
        #one-counts, #one-counts/#all-counts
    """
    one_counts = 0
    for state, counts in counts_dict.items():
        if problem.is_good_state(state):
            one_counts += counts

    return int(one_counts), one_counts / sum(counts_dict.values())


def chernoff_confint(
    value: float, shots: int, max_rounds: int, alpha: float
) -> Tuple[float, float]:
    """Compute the Chernoff confidence interval for `shots` i.i.d. Bernoulli trials.

    The confidence interval is

        [value - eps, value + eps], where eps = sqrt(3 * log(2 * max_rounds/ alpha) / shots)

    but at most [0, 1].

    Args:
        value: The current estimate.
        shots: The number of shots.
        max_rounds: The maximum number of rounds, used to compute epsilon_a.
        alpha: The confidence level, used to compute epsilon_a.

    Returns:
        The Chernoff confidence interval.
    """
    eps = np.sqrt(3 * np.log(2 * max_rounds / alpha) / shots)
    lower = np.maximum(0, value - eps)
    upper = np.minimum(1, value + eps)
    return lower, upper


def clopper_pearson_confint(counts: int, shots: int, alpha: float) -> Tuple[float, float]:
    from scipy.stats import beta
    """Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.

    Args:
        counts: The number of positive counts.
        shots: The number of shots.
        alpha: The confidence level for the confidence interval.

    Returns:
        The Clopper-Pearson confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    if counts != 0:
        lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

    # if counts == shots, the beta quantile returns nan
    if counts != shots:
        upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

    return lower, upper

def find_next_k(
    k: int,
    upper_half_circle: bool,
    theta_interval: Tuple[float, float],
    min_ratio: float = 2.0,
) -> Tuple[int, bool]:
    """Find the largest integer k_next, such that the interval (4 * k_next + 2)*theta_interval
    lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

    Args:
        k: The current power of the Q operator.
        upper_half_circle: Boolean flag of whether theta_interval lies in the
            upper half-circle [0, pi] or in the lower one [pi, 2pi].
        theta_interval: The current confidence interval for the angle theta,
            i.e. (theta_lower, theta_upper).
        min_ratio: Minimal ratio K/K_next allowed in the algorithm.

    Returns:
        The next power k, and boolean flag for the extrapolated interval.

    Raises:
        AlgorithmError: if min_ratio is smaller or equal to 1
    """

    # initialize variables
    theta_l, theta_u = theta_interval
    old_scaling = 4 * k + 2  # current scaling factor, called K := (4k + 2)

    # the largest feasible scaling factor K cannot be larger than K_max,
    # which is bounded by the length of the current confidence interval
    max_scaling = int(1 / (2 * (theta_u - theta_l)))
    scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 * k_max + 2

    # find the largest feasible scaling factor K_next, and thus k_next
    while scaling >= min_ratio * old_scaling:
        theta_min = scaling * theta_l - int(scaling * theta_l)
        theta_max = scaling * theta_u - int(scaling * theta_u)

        if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
            # the extrapolated theta interval is in the upper half-circle
            upper_half_circle = True
            return int((scaling - 2) / 4), upper_half_circle

        elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
            # the extrapolated theta interval is in the upper half-circle
            upper_half_circle = False
            return int((scaling - 2) / 4), upper_half_circle

        scaling -= 4

    # if we do not find a feasible k, return the old one
    return int(k), upper_half_circle



def iterative_amplitude_estimation(problem, backend, shots, epsilon=0.01, alpha=0.05):
    # initialize memory variables
    min_ratio = 2

    # shot_count = int(np.ceil(32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon)))))

    shot_count = shots

    powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
    ratios = []  # list of multiplication factors (called 'q' in paper)
    theta_intervals = [[0, 1 / 4]]  # a priori knowledge of theta / 2 / pi
    a_intervals = [[0.0, 1.0]]  # a priori knowledge of the confidence interval of the estimate
    num_oracle_queries = 0
    oracle_queries_complexity = 0
    num_one_shots = []

    # maximum number of rounds
    max_rounds = (
        int(np.log(min_ratio * np.pi / 8 / epsilon) / np.log(min_ratio)) + 1
    )
    upper_half_circle = True  # initially theta is in the upper half-circle

    num_iterations = 0  # keep track of the number of iterations
    # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
    while theta_intervals[-1][1] - theta_intervals[-1][0] > epsilon / np.pi:
        num_iterations += 1

        # get the next k
        k, upper_half_circle = find_next_k(
            powers[-1],
            upper_half_circle,
            theta_intervals[-1],  # type: ignore
            min_ratio=min_ratio,
        )

        # store the variables
        powers.append(k)
        ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))
                
        iae_circuit = iae_construct_circuit(problem, k=k, measurement=True)

        transpiled_circuits = qiskit.transpile(
            iae_circuit, backend
        )

        sampler = Sampler(mode=backend)
        # Run the circuit
        job = sampler.run([transpiled_circuits], shots=shot_count)
        result = job.result()

        pub_res = result[0]
        
        databin = pub_res.data
        _, bitarray = next(iter(databin.__dict__.items()))
        counts = bitarray.get_counts()

        shots = bitarray.num_shots
        
        # calculate the probability of measuring '1', 'prob' is a_i in the paper
        one_counts, prob = good_state_probability(problem, counts)

        num_one_shots.append(one_counts)

        # track number of Q-oracle calls
        num_oracle_queries += shots * k
        
        oracle_queries_complexity += k * 2

        # if on the previous iterations we have K_{i-1} == K_i, we sum these samples up
        j = 1  # number of times we stayed fixed at the same K
        round_shots = shots
        round_one_counts = one_counts
        if num_iterations > 1:
            while (
                powers[num_iterations - j] == powers[num_iterations] and num_iterations >= j + 1
            ):
                j = j + 1
                round_shots += shots
                round_one_counts += num_one_shots[-j]

        # compute a_min_i, a_max_i
        # if confint_method == "chernoff":
        #     a_i_min, a_i_max = chernoff_confint(prob, round_shots, max_rounds, alpha)
        # else:  # 'beta'
        
        a_i_min, a_i_max = clopper_pearson_confint(
            round_one_counts, round_shots, alpha / max_rounds
        )

        # compute theta_min_i, theta_max_i
        if upper_half_circle:
            theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
            theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
        else:
            theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

        # compute theta_u, theta_l of this iteration
        scaling = 4 * k + 2  # current K_i factor
        theta_u = (int(scaling * theta_intervals[-1][1]) + theta_max_i) / scaling
        theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
        theta_intervals.append([theta_l, theta_u])

        # compute a_u_i, a_l_i
        a_u = np.sin(2 * np.pi * theta_u) ** 2
        a_l = np.sin(2 * np.pi * theta_l) ** 2
        a_u = cast(float, a_u)
        a_l = cast(float, a_l)
        a_intervals.append([a_l, a_u])
        

        # get the latest confidence interval for the estimate of a
    confidence_interval = cast(Tuple[float, float], a_intervals[-1])
    # the final estimate is the mean of the confidence interval
    estimation = np.mean(confidence_interval)

    estimation_processed = problem.post_processing(
        estimation  # type: ignore[arg-type,assignment]
    )
    confidence_interval_processed = tuple(
        problem.post_processing(bound) for bound in confidence_interval
    )
    
    return (
        estimation,
        estimation_processed,
        oracle_queries_complexity,
        num_oracle_queries,
        confidence_interval_processed,
    )
