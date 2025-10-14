import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

import jax.numpy as jnp

import vehicle_model_jax as vm

# parameter keys (must match order expected by build_p_tuple / vehicle_model_jax.evalf)
KEYS = [
	'm', 'I', 'c_d', 'c_r',
	'R_s', 'L_ds', 'L_qs', 'lambda_f', 'p_pairs',
	'G', 'r_w', 'eta_g', 'k',
	'K_p_i', 'K_i_i', 'K_p_v', 'K_i_v', 'K_p_theta', 'K_d_theta',
	'v_w', 'psi', 'c_wx', 'c_wy'
]


def compute_jacobian(x, p_tuple, u):
	J = vm.compute_jacobian_jax(jnp.array(x, dtype=jnp.float64), p_tuple, jnp.array(u, dtype=jnp.float64))
	return np.array(J, dtype=np.float64)


def build_p_tuple(p_dict):
	"""Return parameter tuple in the order expected by `evalf` in `vehicle_model_jax`."""
	# ensure k is present
	if p_dict.get('k', None) is None:
		p_dict['k'] = p_dict['eta_g'] * p_dict['G'] / p_dict['r_w']
	return tuple([jnp.array(p_dict[k], dtype=jnp.float64) for k in KEYS])


def get_param_sweep_values(base_p, param_name, n=40):
	"""Return an array of parameter values to sweep for a given parameter name.
	Uses simple heuristics per-parameter to produce physically-meaningful ranges.
	"""
	base_val = base_p.get(param_name, None)
	# per-parameter bespoke ranges
	if param_name == 'psi':
		return np.linspace(-np.pi, np.pi, n)
	if param_name == 'v_w':
		# typical wind speeds 0..20 m/s
		return np.linspace(0.0, 20.0, n)
	if param_name in ('c_wx', 'c_wy'):
		return np.linspace(0.0, 2.0, n)
	if param_name == 'p_pairs':
		# small integers; return integer-like values across 1..8
		vals = np.linspace(1, 8, min(n, 8))
		return vals.astype(float)

	# default heuristics
	if base_val is None:
		return np.linspace(-1.0, 1.0, n)
	try:
		b = float(base_val)
	except Exception:
		# fallback
		return np.linspace(-1.0, 1.0, n)

	if b == 0.0:
		# small symmetric range
		return np.linspace(-1.0, 1.0, n)
	if b > 0:
		# sweep multiplicatively across several orders of magnitude
		return b * np.logspace(-2, 2, n)
	# negative base: sweep multiplicatively keeping sign
	return b * np.logspace(-2, 2, n)


def sample_operating_points(n_samples, seed=0):
	"""Produce a set of operating points x (numpy arrays) for the state vector used by vm."""
	rng = np.random.default_rng(seed)
	N = vm.N_STATES
	xs = []
	for _ in range(n_samples):
		i_ds_r = rng.uniform(-100.0, 100.0)
		i_qs_r = rng.uniform(-100.0, 100.0)
		I_err_ds = rng.uniform(-1.0, 1.0)
		I_err_qs = rng.uniform(-1.0, 1.0)
		theta = rng.uniform(-np.pi, np.pi)
		v = rng.uniform(0.0, 30.0)
		omega = rng.uniform(-50.0, 50.0)
		I_err_v = rng.uniform(-1.0, 1.0)
		x = np.array([i_ds_r, i_qs_r, I_err_ds, I_err_qs, theta, v, omega, I_err_v], dtype=np.float64)
		xs.append(x)
	return xs


def compute_condition_numbers(xs, p_tuple, max_samples=None):
	"""Compute Jacobian condition numbers for a list of operating points xs.
	Returns list of cond numbers and any diagnostics about NaNs/infs encountered.
	"""
	conds = []
	bad = 0
	total = 0
	for i, x in enumerate(xs):
		if max_samples and i >= max_samples:
			break
		# build u such that references equal the current states (steady operating point)
		v = x[5]
		theta = x[4]
		u = jnp.array([v, theta], dtype=jnp.float64)
		x_jnp = jnp.array(x, dtype=jnp.float64)
		try:
			J = vm.compute_jacobian_jax(x_jnp, p_tuple, u)
			J_np = np.array(J, dtype=np.float64)
			if np.any(np.isnan(J_np)) or np.any(np.isinf(J_np)):
				bad += 1
				conds.append(np.nan)
			else:
				c = np.linalg.cond(J_np, 2)
				conds.append(c)
		except Exception as e:
			# record as bad
			bad += 1
			conds.append(np.nan)
		total += 1
	return np.array(conds), {'bad': bad, 'total': total}


def parameter_sweep_param(base_p, x0, param_name, values):
	"""Generic parameter sweep.
	param_name: string name of parameter in p_dict to set. Special-case 'delta_L'
	    to set L_ds = L_qs + value while keeping L_qs fixed.
	values: iterable of parameter values to try.
	Returns an np.array of condition numbers (or nan where evaluation failed).
	"""
	results = []
	for val in values:
		p = base_p.copy()
		if param_name == 'delta_L':
			# set L_ds relative to L_qs
			p['L_qs'] = base_p['L_qs']
			p['L_ds'] = p['L_qs'] + float(val)
		else:
			p[param_name] = float(val)
		# recompute dependent parameters if any
		p['k'] = p['eta_g'] * p['G'] / p['r_w']
		p_tuple = build_p_tuple(p)
		v = x0[5]
		theta = x0[4]
		u = jnp.array([v, theta], dtype=jnp.float64)
		x_jnp = jnp.array(x0, dtype=jnp.float64)
		try:
			J = vm.compute_jacobian_jax(x_jnp, p_tuple, u)
			J_np = np.array(J, dtype=np.float64)
			if np.any(np.isnan(J_np)) or np.any(np.isinf(J_np)):
				results.append(np.nan)
			else:
				results.append(np.linalg.cond(J_np, 2))
		except Exception:
			results.append(np.nan)
	return np.array(results)

def main():
	out_dir = Path('conditioning_outputs')
	out_dir.mkdir(exist_ok=True)
	start_time = time.time()

	base_p = vm.get_default_params()
	p_tuple = build_p_tuple(base_p)

	# 1) sample many operating points and compute condition numbers
	print('Sampling operating points...')
	xs = sample_operating_points(500, seed=500)
	conds, diag = compute_condition_numbers(xs, p_tuple)
	valid = conds[~np.isnan(conds)]
	print(f'Computed {diag["total"]} Jacobians, {diag["bad"]} had NaN/inf or errors')

	# statistics
	stats = {
		'count': int(np.sum(~np.isnan(conds))),
		'min': float(np.nanmin(conds)),
		'q25': float(np.nanpercentile(conds, 25)),
		'median': float(np.nanmedian(conds)),
		'mean': float(np.nanmean(conds)),
		'q75': float(np.nanpercentile(conds, 75)),
		'max': float(np.nanmax(conds))
	}

	# save histogram
	plt.figure()
	plt.hist(valid, bins=60)
	plt.xlabel('Condition number (2-norm)')
	plt.ylabel('Frequency')
	plt.title('Histogram of Jacobian condition numbers across operating points')
	plt.savefig(out_dir / 'conds_histogram.png')
	plt.close()

	# 2) sweep every parameter in KEYS (except 'k' which is derived, and 'delta_L' is special-case)
	print('Running full parameter sweeps for all parameters...')
	x0 = np.array([0.0, 0.0, 0.0, 0.0, 4.0, 25.0, 0.0, 0.0], dtype=np.float64)
	param_summaries = {}
	for param in KEYS:
		if param == 'k':
			continue  # derived parameter, skip
		# get sweep values for this parameter
		vals = get_param_sweep_values(base_p, param, n=40)
		# use parameter_sweep_param to evaluate condition numbers for these values
		conds_param = parameter_sweep_param(base_p, x0, param, vals)
		param_summaries[param] = {
			'vals': vals,
			'conds': conds_param,
			'max': float(np.nanmax(conds_param)) if np.any(~np.isnan(conds_param)) else np.nan,
			'mean': float(np.nanmean(conds_param)) if np.any(~np.isnan(conds_param)) else np.nan,
		}
		# plot and save
		plt.figure()
		if np.all(np.isfinite(vals)) and np.all(vals > 0):
			plt.semilogx(vals, conds_param, '.-')
			xlabel = param
		else:
			plt.plot(vals, conds_param, '.-')
			xlabel = param
		plt.xlabel(xlabel)
		plt.ylabel('Condition number')
		plt.title(f'Condition number vs {param}')
		plt.grid(True)
		plt.savefig(out_dir / f'conds_vs_{param}.png')
		plt.close()
		# (no CSV output by request) plots are saved above

	# save a small summary file
	with open(out_dir / 'param_sweep_summary.txt', 'w') as sf:
		for param, info in param_summaries.items():
			sf.write(f"{param}: max={info['max']}, mean={info['mean']}\n")
			
if __name__ == '__main__':
	main()
	


