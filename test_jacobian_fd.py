import json
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable

jax.config.update("jax_enable_x64", True)

from vehicle_model_jax import evalf, evalf_np, compute_jacobian_jax, get_default_params
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

EPS = 1e-3


@dataclass
class TestResult:
    name: str
    passed: bool
    max_error: float
    avg_error: float
    n_trials: int = 1
    comparison_method: str = "finite_difference"  # or "analytic" or "regression"


def compute_jacobian_error(J_jax: jnp.ndarray, J_reference: jnp.ndarray) -> Tuple[float, float]:
    errors = np.abs(J_jax - J_reference).flatten()
    return float(np.max(errors)), float(np.mean(errors))


def check_jacobians(J_jax: jnp.ndarray, J_reference: jnp.ndarray, eps: float = EPS) -> Tuple[bool, float, float]:
    max_error, avg_error = compute_jacobian_error(J_jax, J_reference)
    passed = max_error < eps
    return passed, max_error, avg_error


def compute_analytic_jacobian(x0: jnp.ndarray, p_tuple: tuple, u: jnp.ndarray) -> jnp.ndarray:
    def f_wrapper(x):
        return evalf(x, p_tuple, u)
    
    jacobian_fn = jax.jacfwd(f_wrapper)
    return jacobian_fn(x0)


def test_manual_regression(p_tuple: tuple, eps: float = EPS) -> TestResult:
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        data = json.load(f)
    
    all_errors = []
    max_error = 0.0
    
    for test_case in data:
        x0 = jnp.array(test_case["x"])
        p_case = tuple(test_case["p"])
        u = jnp.array(test_case["u"])
        J_true = jnp.array(test_case["J"])
        
        J = compute_jacobian_jax(x0, p_case, u)
        errors = np.abs(J_true - J).flatten()
        all_errors.extend(errors)
        max_error = max(max_error, np.max(errors))
    
    avg_error = float(np.mean(all_errors))
    passed = max_error < eps
    
    return TestResult("Manual Regression", passed, max_error, avg_error, len(data), "regression")


def test_steady_state(p_tuple: tuple, eps: float = EPS, use_analytic: bool = False) -> TestResult:
    x0 = jnp.zeros(10)
    u = jnp.array([0.0, 0.0])
    
    J_jax = compute_jacobian_jax(x0, p_tuple, u)
    
    if use_analytic:
        J_reference = compute_analytic_jacobian(x0, p_tuple, u)
        method = "analytic"
    else:
        J_reference, _ = eval_Jf_FiniteDifference(evalf_np, x0, p_tuple, u)
        method = "finite_difference"
    
    passed, max_error, avg_error = check_jacobians(J_jax, J_reference, eps)
    return TestResult("Steady State", passed, max_error, avg_error, 1, method)


def run_trial_based_test(
    test_name: str,
    state_generator: Callable[[int], Tuple[jnp.ndarray, jnp.ndarray]],
    p_tuple: tuple,
    n_trials: int = 1000,
    eps: float = EPS,
    use_analytic: bool = False
) -> TestResult:
    all_errors = []
    
    for trial in range(n_trials):
        x0, u = state_generator(trial)
        
        J_jax = compute_jacobian_jax(x0, p_tuple, u)
        
        if use_analytic:
            J_reference = compute_analytic_jacobian(x0, p_tuple, u)
        else:
            J_reference, _ = eval_Jf_FiniteDifference(evalf_np, x0, p_tuple, u)
        
        _, _, avg_error = check_jacobians(J_jax, J_reference, eps)
        all_errors.append(avg_error)
    
    max_error = max(all_errors)
    avg_error = float(np.mean(all_errors))
    passed = max_error < eps
    method = "analytic" if use_analytic else "finite_difference"
    
    return TestResult(test_name, passed, max_error, avg_error, n_trials, method)


def test_constant_speed(p_tuple: tuple, n_trials: int = 1000, eps: float = EPS, use_analytic: bool = False) -> TestResult:
    def state_gen(trial):
        v0 = np.random.rand() * 10.0
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, 0.0])
        return x0, u
    
    return run_trial_based_test("Constant Speed", state_gen, p_tuple, n_trials, eps, use_analytic)


def test_acceleration(p_tuple: tuple, n_trials: int = 1000, eps: float = EPS, use_analytic: bool = False) -> TestResult:
    def state_gen(trial):
        v0 = np.random.rand() * 10.0
        x0 = jnp.zeros(10)
        u = jnp.array([v0, 0.0])
        return x0, u
    
    return run_trial_based_test("Acceleration", state_gen, p_tuple, n_trials, eps, use_analytic)


def test_spin(p_tuple: tuple, n_trials: int = 1000, eps: float = EPS, use_analytic: bool = False) -> TestResult:
    def state_gen(trial):
        w0 = np.random.rand() * np.pi / 2
        x0 = jnp.zeros(10)
        u = jnp.array([0.0, w0])
        return x0, u
    
    return run_trial_based_test("Spinning", state_gen, p_tuple, n_trials, eps, use_analytic)


def test_level_turn(p_tuple: tuple, n_trials: int = 1000, eps: float = EPS, use_analytic: bool = False) -> TestResult:
    def state_gen(trial):
        v0 = np.random.rand() * 10.0
        w0 = np.random.rand() * np.pi / 2
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, w0])
        return x0, u
    
    return run_trial_based_test("Level Turn", state_gen, p_tuple, n_trials, eps, use_analytic)


def print_test_result(result: TestResult):
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    trials_info = f" ({result.n_trials} trials)" if result.n_trials > 1 else ""
    method_info = f" [{result.comparison_method}]"
    
    print(f"{status}: {result.name}{trials_info}{method_info}")
    print(f"  Max Error: {result.max_error:.2e}")
    print(f"  Avg Error: {result.avg_error:.2e}")
    print()


def main():
    # Load default parameters
    p = get_default_params()
    p_tuple = tuple(p.values())
    
    # Configuration: set to True to use analytic Jacobian instead of finite difference
    USE_ANALYTIC = False
    
    # Run all tests
    tests = [
        test_manual_regression(p_tuple),
        test_steady_state(p_tuple, use_analytic=USE_ANALYTIC),
        test_constant_speed(p_tuple, use_analytic=USE_ANALYTIC),
        test_acceleration(p_tuple, use_analytic=USE_ANALYTIC),
        test_spin(p_tuple, use_analytic=USE_ANALYTIC),
        test_level_turn(p_tuple, use_analytic=USE_ANALYTIC),
    ]
    
    # Print results
    print("=" * 60)
    print("JACOBIAN TEST RESULTS")
    print("=" * 60)
    print()
    
    for result in tests:
        print_test_result(result)
    
    # Print Summary
    all_passed = all(test.passed for test in tests)
    total_trials = sum(test.n_trials for test in tests)
    overall_avg_error = np.mean([test.avg_error for test in tests])
    
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Total Trials: {total_trials}")
    print(f"Overall Average Error: {overall_avg_error:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()