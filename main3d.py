import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.transforms import Affine2D
from matplotlib.collections import PolyCollection
import alphashape
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection

from iris import SeparatingHyperplanes, ClosestPointOnObstacle, TangentPlane, InscribedEllipsoid

# np.random.seed(0)

seed_point = None
As = []
bs = []
Cs = []
ds = []
tolerance = 0.00001
limits = [[0, 1], [0, 1], [0, 1]]
max_iters = 10

cvx_range = 0.25

regions = []
current_region = []

always_contain_seed_point = True

save_images = False
fig_count = 0

def gen_obstacles():
	n_points = 80
	alpha = 10.

	points = np.random.random(size=(n_points, 3))
	gen = alphashape.alphasimplices(points)

	# print(*zip(*gen.vertices))

	tris = []
	for simplex, r in gen:
		print(simplex, r, 1/alpha)
		if r < 1/alpha:
			tris.append(points[simplex])

	for i in range(len(tris)):
		tris.append(tris[i] + np.array([1, 0, 0]))
		tris.append(tris[i] + np.array([-1, 0, 0]))
		tris.append(tris[i] + np.array([0, 1, 0]))
		tris.append(tris[i] + np.array([0, -1, 0]))
		tris.append(tris[i] + np.array([0, 0, 1]))
		tris.append(tris[i] + np.array([0, 0, -1]))
		tris.append(tris[i] + np.array([1, 1, 1]))
		tris.append(tris[i] + np.array([1, -1, 1]))
		tris.append(tris[i] + np.array([-1, 1, 1]))
		tris.append(tris[i] + np.array([-1, -1, 1]))
		tris.append(tris[i] + np.array([1, 1, -1]))
		tris.append(tris[i] + np.array([1, -1, -1]))
		tris.append(tris[i] + np.array([-1, 1, -1]))
		tris.append(tris[i] + np.array([-1, -1, -1]))

	return tris

def gen_boundaries(seed_point):
	global tris
	tris = orig_tris.copy()

	left = np.array([[0, -0.5], [0, 0.5], [-0.5, 0]]) + seed_point + np.array([-1 * cvx_range, 0])
	right = np.array([[0, -0.5], [0, 0.5], [0.5, 0]]) + seed_point + np.array([cvx_range, 0])
	bottom = np.array([[-0.5, 0], [0.5, 0], [0, -0.5]]) + seed_point + np.array([0, -1 * cvx_range])
	top = np.array([[-0.5, 0], [0.5, 0], [0, 0.5]]) + seed_point + np.array([0, cvx_range])

	tris.append(left)
	tris.append(right)
	tris.append(top)
	tris.append(bottom)

def draw_ellipse(C, d):
	ts = np.linspace(0, 2*np.pi)
	points = np.array([np.cos(ts), np.sin(ts)])
	points = C @ points + d.reshape(-1,1)
	ax.plot(*(points), color="blue")
	arrs = [
		np.array([1, 0]),
		np.array([-1, 0]),
		np.array([0, -1]),
		np.array([0, 1]),
		np.array([1, 1]),
		np.array([1, -1]),
		np.array([-1, 1]),
		np.array([-1, -1])
	]
	for arr in arrs:
		ax.plot(*(points + arr.reshape(-1, 1)), color="blue")

def draw_intersection(A, b, d):
	global current_region
	ineq = np.hstack((A.T, -b))
	hs = HalfspaceIntersection(ineq, d, incremental=False)
	points = hs.intersections
	centered_points = points - d
	thetas = np.arctan2(centered_points[:,1], centered_points[:,0])
	idxs = np.argsort(thetas)
	current_region = points[idxs]
	arrs = [
		np.array([0, 0]),
		np.array([1, 0]),
		np.array([-1, 0]),
		np.array([0, -1]),
		np.array([0, 1]),
		np.array([1, 1]),
		np.array([1, -1]),
		np.array([-1, 1]),
		np.array([-1, -1])
	]
	for arr in arrs:
		ax.add_patch(Polygon(current_region + arr, color="blue", alpha=0.25))

def draw():
	global seed_point, As, bs, Cs, ds, regions
	ax.cla()
	ax.set_xlim(limits[0])
	ax.set_ylim(limits[1])
	ax.set_zlim(limits[2])
	# ax.set_aspect("equal")
	for tri in orig_tris:
		# ax.add_patch(Polygon(tri, color="red"))
		ax.plot_trisurf(*tri)

	# # for i in range(-1, -5, -1):
	# # 	ax.add_patch(Polygon(tris[i], color="red", alpha=0.25))
	# if not (seed_point is None):
	# 	ax.scatter([seed_point[0]], [seed_point[1]])
	# if len(Cs) > 0:
	# 	C = Cs[-1]
	# 	d = ds[-1]
	# 	draw_ellipse(C, d)
	# if len(As) > 0:
	# 	A = As[-1]
	# 	b = bs[-1]
	# 	for i in range(len(b)):
	# 		w = A[:,i]
	# 		intercept = b[i]
	# 		xx = np.linspace(*limits[0])
	# 		yy = (-w[0] / w[1]) * xx + (intercept / w[1])
	# 		# ax.plot(xx, yy, color="blue")
	# 	draw_intersection(A, b, ds[-1])
	# for idx, region in enumerate(regions):
	# 	color = plt.get_cmap("Set3")(float(idx) / 12.)
	# 	arrs = [
	# 		np.array([0, 0]),
	# 		np.array([1, 0]),
	# 		np.array([-1, 0]),
	# 		np.array([0, -1]),
	# 		np.array([0, 1]),
	# 		np.array([1, 1]),
	# 		np.array([1, -1]),
	# 		np.array([-1, 1]),
	# 		np.array([-1, -1])
	# 	]
	# 	for arr in arrs:
	# 		temp = region + arr
	# 		plt.plot(temp[:,0], temp[:,1], color=color, alpha=0.75)
	# 		plt.plot(temp[[0,-1],0], temp[[0,-1],1], color=color, alpha=0.75)
	# 		ax.add_patch(Polygon(temp, color=color, alpha=0.75))
	# plt.draw()

	# if save_images:
	# 	global fig_count
	# 	plt.savefig("img_%03d.png" % fig_count)
	# 	fig_count += 1


	# n = 2
	# C = cp.Variable((n,n), symmetric=True)
	# d = cp.Variable(n)
	# constraints = [C >> 0]
	# constraints += [
	# 	cp.atoms.norm2(ai.T @ C) + (ai.T @ d) <= bi for ai, bi in zip(A.T, b)
	# ]
	# prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
	# prob.solve()
	# return C.value, d.value

def optim():
	global As, bs, Cs, ds, seed_point, regions, current_region, tris
	As = []
	bs = []
	Cs = []
	ds = []
	
	C0 = np.eye(2) * 0.01
	Cs.append(C0)
	ds.append(seed_point.copy())
	O = tris

	draw()
	plt.pause(0.001)

	iters = 0

	while True:
		print("Iteration %d" % iters)

		A, b = SeparatingHyperplanes(Cs[-1], ds[-1], O.copy())
		if always_contain_seed_point and np.any(A.T @ seed_point >= b.flatten()):
			print("Terminating early to keep seed point in region.")
			break

		As.append(A)
		bs.append(b)

		draw()
		plt.pause(0.001)

		C, d = InscribedEllipsoid(As[-1], bs[-1])
		Cs.append(C)
		ds.append(d)

		draw()
		plt.pause(0.001)

		iters += 1

		if (np.linalg.det(Cs[-1]) - np.linalg.det(Cs[-2])) / np.linalg.det(Cs[-2]) < tolerance:
			break

		if iters > max_iters:
			break

	print("Done")
	As = []
	bs = []
	Cs = []
	ds = []
	tris = orig_tris.copy()
	seed_point = None
	regions.append(current_region)
	draw()
	plt.pause(0.001)

def onmousepress(event):
	global A, b, C, d
	global seed_point, tris, orig_tris
	if seed_point is None:
		if event.inaxes:
			seed_point = np.array([event.xdata, event.ydata])
			gen_boundaries(seed_point)
			optim()
			draw()

orig_tris = gen_obstacles()
tris = orig_tris.copy()

print(tris)

ax = plt.figure().add_subplot(projection="3d")
# fig.canvas.mpl_connect("button_press_event", onmousepress)
draw()
plt.show()