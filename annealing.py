
# from scipy.optimize import dual_annealing
import dual_annealing
import time
import tipe
import numpy as np
from tqdm import tqdm
import os



N_THREAD = 2
NB_PARTICLE = 150
NB_ITER = 80
MAX_ITER = 500


track = tipe.Track("Data/Track_data/cbg.geojson")
pbar = tqdm(total=NB_ITER*NB_PARTICLE*track.nb_sectors, position=3, desc="Threads")

date = time.strftime("%d_%m_%y_%H:%M",time.localtime())
path = f"Data/annealing_output/{date}"

d = []
w = 0.8

a = []

def func(W):
	l = []
	W = list(W)
	c1 = round(W[0],2)
	c2 = round(W[1],2)
	W = [c1,c2]
	# print(f"Starting with c1={c1}, c2={c2} at {time.ctime()}")
	cur_param.set_description_str(f"Current parameters : c1={c1}, c2={c2} at {time.ctime()}")
 
	file = open(f"{path}/data_{id}.txt", "r")
	d = list(eval(file.read()))
	file.close()
	for c1_,c2_,lmin_,lmean_,lstd_ in d:
		if c1_ == c1 and c2_ == c2:
			# print(f"{c1},{c2} already computed")
			res.set_description_str(f"c1={c1}, c2={c2} already computed : {lmin_},{lmean_},{lstd_}")
			a.append(1)
			dual_annealing.pbar.refresh()
			return lmin_
	for i in range(N_THREAD):
		gs, gs_e = tipe.pso_(tipe.fitFunc, track.nb_sectors, track.boundaries, NB_PARTICLE, NB_ITER, w, c1, c2, track,pbar)
		pbar.reset()
		l.append(gs_e)


	
	l = np.array(l)
	lmin = l.min()
	lmean = l.mean()
	lstd = l.std()
	
	d.append([c1, c2, lmin, lmean, lstd])
	# print(f"c1={c1},c2={c2} : {lmin},{lmean}")
	res.set_description_str(f"c1={c1}, c2={c2} : {lmin},{lmean},{lstd}")
	
	file = open(f"{path}/data_{id}.txt", "w")
	# print(str(d))
	file.write(str(d))
	file.close()
	a.append(0)
	dual_annealing.pbar.refresh()
 
	return lmin



def main(id=id):
	try:
		os.mkdir(path)
	except:
		pass
	
	file = open(f"{path}/data_{id}.txt","w")
	file.write("[]")
	file.close()
 
 
	d = []
	
 
	dual_annealing.pbar = tqdm(total=MAX_ITER, position=0, desc="Iterations")
	dual_annealing.pbar.write(f"""Starting calculation with :
       \033[96mNB_PARTICLE\033[00m=\033[92m{NB_PARTICLE}\033[00m,
       \033[96mNB_ITER\033[00m=\033[92m{NB_ITER}\033[00m,
       \033[96mN_THREAD\033[00m=\033[92m{N_THREAD}\033[00m,
       \033[96mMAX_ITER\033[00m=\033[92m{MAX_ITER}\033[00m""")
	r_min, r_max = -10., 10.
	bounds = [[r_min, r_max], [r_min, r_max]]
	result = dual_annealing.dual_annealing(func, bounds,maxiter=MAX_ITER)
	
	
	# summarize the result
	# print('Status : %s' % result['message'])
	# print('Total Evaluations: %d' % result['nfev'])
	
	# evaluate solution
	solution = result['x']
	solution = solution.tolist()
	evaluation = func(solution)
	# print('Solution: f(%s) = %.5f' % (solution, evaluation))
 
	cur_param.write(f"Status : {result['message']}")
	dual_annealing.pbar.close()
	pbar.close()
	cur_param.set_description_str(f"Total Evaluations: {result['nfev']}")
	res.set_description_str(f"Solution found : f({solution}) = {evaluation}")
	cur_param.close()
	res.close()	
	
	file = open(f"{path}/data_{id}.txt","r")
	data = file.read()
	file.close()

	file = open(f"{path}/data{id}.txt","w")
	file.write(data)
	file.close()
	
	file = open(f"{path}/info{id}.txt","w")
	file.write(str((w,r_min,r_max,NB_PARTICLE,NB_ITER,MAX_ITER))+"\n")
	file.write(str([solution,evaluation,result['message'],result['nfev']])+"\n")
	file.write(str(a)+"\n")
	file.write(str(dual_annealing.r)+"\n")
	file.close()
	print(f"\nSaving OK")	
	file = open(f"{path}/data_{id}.txt","w")
	file.close()
	os.remove(f"{path}/data_{id}.txt")

 
 


if __name__ == "__main__":
	cur_param = tqdm(total=0, position=1, bar_format='{desc}')
	res = tqdm(total=0, position=2, bar_format='{desc}')
	
	main(id)
	
	
