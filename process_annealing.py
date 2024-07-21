import dual_annealing
import time
import tipe
import numpy as np
from tqdm import tqdm
import os
import sys


N_PROCESS = 8
NB_PARTICLE = 50
NB_ITER = 200

if not os.path.exists("Data/Track_data"):
    os.mkdir("Data/Track_data")
if not os.path.exists("Data/Track_data/cbg.geojson"):
    os.system("wget -O Data/Track_data/cbg.geojson https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/fr-1969.geojson")
    
track = tipe.Track("Data/Track_data/cbg.geojson")
pbar = tqdm(total=NB_ITER*NB_PARTICLE*track.nb_sectors, position=3, desc="Threads")
if not os.path.exists("Data/annealing_output"):
	os.mkdir("Data/annealing_output")
 
MAX_ITER = N_PROCESS*NB_PARTICLE*NB_ITER*track.nb_sectors

date = time.strftime("%d_%m_%y_%H:%",time.localtime())
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
	for i in range(N_PROCESS):
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
       \033[96mN_PROCESS\033[00m=\033[92m{N_PROCESS}\033[00m,
       \033[96mMAX_ITER\033[00m=\033[92m{MAX_ITER}\033[00m""")
	r_min, r_max = -10., 10.
	# bounds = [[r_min, r_max], [r_min, r_max]]
	bounds = [(-10,40),(-10,10)]
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
	file.write(str((w,bounds,NB_PARTICLE,NB_ITER,MAX_ITER))+"\n")
	file.write(str([solution,evaluation,result['message'],result['nfev']])+"\n")
	file.write(str(a)+"\n")
	file.write(str(dual_annealing.r)+"\n")
	file.close()
	print(f"\nSaving OK")	
	file = open(f"{path}/data_{id}.txt","w")
	file.close()
	os.remove(f"{path}/data_{id}.txt")

 
 


if __name__ == "__main__":
	if len(sys.argv) > 1:
		id = int(sys.argv[1])
	else:
		id = 0
	if os.path.exists(f"Data/annealing_output/{date}/data_{id}.txt") or os.path.exists(f"Data/annealing_output/{date}/data{id}.txt"):
		print(f"data_{id}.txt already exists")
		id_list = []
		for file_name in os.listdir(f"Data/annealing_output/{date}"):
			if file_name.startswith("data_") and file_name.endswith(".txt"):
				id_list.append(int(file_name.split("_")[1].split(".")[0]))
			elif file_name.startswith("data") and file_name.endswith(".txt"):
				id_list.append(int(file_name[4:].split(".")[0]))
		if id_list:
			biggest_id = max(id_list)
		else:
			biggest_id = 0
		print(f"Starting with id={biggest_id+1}")
		id = biggest_id+1
	
	cur_param = tqdm(total=0, position=1, bar_format='{desc}')
	res = tqdm(total=0, position=2, bar_format='{desc}')
	
	main(id)
	
	
