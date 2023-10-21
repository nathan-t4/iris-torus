import numpy as np
import alphashape
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection

def SeparatingHyperplanes(C, d, O):
	C_inv = np.linalg.inv(C)
	C_inv2 = C_inv @ C_inv.T
	O_excluded = []
	O_remaining = O
	ais = []
	bis = []
	while len(O_remaining) > 0:
		xs = []
		dists = []
		for o in O_remaining:
			x, dist = ClosestPointOnObstacle(C, C_inv, d, o)
			xs.append(x)
			dists.append(dist)
		best_idx = np.argmin(dists)
		x_star = xs[best_idx]
		ai, bi = TangentPlane(C, C_inv2, d, x_star)
		ais.append(ai)
		bis.append(bi)
		idx_list = []
		for i, li in enumerate(O_remaining):
			redundant = [np.dot(ai.flatten(), xj) >= bi for xj in li]
			if i == best_idx or np.all(redundant):
				idx_list.append(i)
		for i in reversed(idx_list):
			O_excluded.append(O_remaining[i])
			O_remaining.pop(i)
	A = np.array(ais).T[0]
	b = np.array(bis).reshape(-1,1)
	return (A, b)

def ClosestPointOnObstacle(C, C_inv, d, o):
	v_tildes = C_inv @ (o - d).T
	n = 2
	m = len(o)
	x_tilde = cp.Variable(n)
	w = cp.Variable(m)
	prob = cp.Problem(cp.Minimize(cp.sum_squares(x_tilde)), [
		v_tildes @ w == x_tilde,
		w @ np.ones(m) == 1,
		w >= 0
	])
	prob.solve()
	x_tilde_star = x_tilde.value
	dist = np.sqrt(prob.value) - 1
	x_star = C @ x_tilde_star + d
	return x_star, dist

def TangentPlane(C, C_inv2, d, x_star):
	a = 2 * C_inv2 @ (x_star - d).reshape(-1, 1)
	b = np.dot(a.flatten(), x_star)
	return a, b

def InscribedEllipsoid(A, b):
	n = 2
	C = cp.Variable((n,n), symmetric=True)
	d = cp.Variable(n)
	constraints = [C >> 0]
	constraints += [
		cp.atoms.norm2(ai.T @ C) + (ai.T @ d) <= bi for ai, bi in zip(A.T, b)
	]
	prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
	prob.solve()
	return C.value, d.value