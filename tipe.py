import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import LinearRing, get_coordinates, shortest_line, Point
import math
from scipy import interpolate
import time 
from pyproj import Proj
from tqdm import tqdm
import os


# Constants
WIDTH = 12
# A = 6378249.2 # m
# B = 6356515 # m
A = 6378137 # m
F= 1/298.257223563



class Track:
	def __init__(self, filename):

		self.nb_sectors = 70
		self.sectors = []

		self.filename = filename
		self.trackData = gpd.read_file(filename)
		self.trackname = self.trackData['Name'][0]

		self.coords = None
		self.pos_x, self.pos_y = self.getCoords() # useless
		
		# self.trackWidth = self.trackData['Width'][0] if self.trackData['Width'][0] else WIDTH
		self.centerLine = None
		self.innerLimit = None
		self.outerLimit = None

		self.s_points = None
		self.inside_points = None 
		self.outside_points = None

		self.boundaries = []
  
		self.test = []
		self.test2 = True


		self.generateTrackLimits()

		# self.defineSector(cleaner(self.pos_x,self.pos_y,sorted(flatten(sectors(self.pos_x,self.pos_y)[0]))))
		self.defineSector()
		# self.defineSector(sorted(flatten(sectors(self.pos_x,self.pos_y)[0])))


		self.getBoundaries()

		self.racingLine = None
		self.rl2 = None

		
		
	def getCoords(self):
		self.coords = interpolator(self.trackData['geometry'][0],1000).flatten()
		x,y = [self.coords[i] for i in range(len(self.coords)) if i % 2 == 0], [self.coords[i] for i in range(len(self.coords)) if i % 2 == 1]
		p = Proj(proj='utm',zone=31,ellps='WGS84', preserve_units=False)
		x,y = p(x,y)
		self.coords = np.array([x,y]).T.flatten()
		return x,y



	def plotTrack(self, centerLine=False, innerLimit=True, outerLimit=True, sectors=False,racingLine =False,nb=False, points=False):
		
		plt.figure(figsize=(11.2,6.3))

		if centerLine:
			plt.plot(*self.centerLine.xy, color='black', linewidth=2)
		if innerLimit:
			# print(self.innerLimit.xy)
			plt.plot(*self.innerLimit.xy, color='red', linewidth=1)
		if outerLimit:
			# print(self.outerLimit.xy)
			plt.plot(*self.outerLimit.xy, color='blue', linewidth=1)
		if racingLine:
			plt.plot(*self.racingLine, color='black', linewidth=2)
		if sectors:
			for i in range(self.nb_sectors):
				plt.plot([self.outside_points[i][0],self.inside_points[i][0]],[self.outside_points[i][1],self.inside_points[i][1]])
		if nb:
			for j in range(len(self.rl2[0])):
				if j == 0:
					plt.text(self.rl2[0][j]-15,self.rl2[1][j]-34, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				elif len(self.rl2[0])-j in [4,5,15,23,39,40,41,45,46]:
					plt.text(self.rl2[0][j]+5,self.rl2[1][j]-18, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				elif len(self.rl2[0])-j-1 in [32,33,35]:
					plt.text(self.rl2[0][j]+20,self.rl2[1][j]+10, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				else:	
					plt.text(self.rl2[0][j]+12,self.rl2[1][j]+12, len(self.rl2[0])-j-1, fontsize=12)
		if points:
			plt.plot(self.rl2[0],self.rl2[1], "go", markersize=4)

		plt.gca().set_aspect('equal')
		plt.title(self.trackname)
		plt.legend(['Center Line', 'Inner Limit', 'Outer Limit'])
		plt.show()
  
  
	def saveTrack(self, centerLine=False, innerLimit=True, outerLimit=True, sectors=False,racingLine =False,nb=False, points=False, transparent=False, path="Data/plot_output/"):  
		try:
			os.mkdir(path)
		except:
			pass  

		path_ = f"{path}{self.trackname}{'_racingline'if racingLine else ''}{'_centerline'if centerLine else ''}{'_nb' if nb else ''}{'_t' if transparent else ''}{'_s' if sectors else ''}.png"
		print(f"saving {path_} ... at {time.strftime('%H:%M:%S')}")
  
		plt.figure(figsize=(11.2,6.3))

		if centerLine:
			plt.plot(*self.centerLine.xy, color='black', linewidth=2)
		if innerLimit:
			# print(self.innerLimit.xy)
			plt.plot(*self.innerLimit.xy, color='red', linewidth=1)
		if outerLimit:
			# print(self.outerLimit.xy)
			plt.plot(*self.outerLimit.xy, color='blue', linewidth=1)
		if racingLine:
			plt.plot(*self.racingLine, color='black', linewidth=2)
		if sectors:
			for i in range(self.nb_sectors):
				plt.plot([self.outside_points[i][0],self.inside_points[i][0]],[self.outside_points[i][1],self.inside_points[i][1]],color='black')
		if nb:
			for j in range(len(self.rl2[0])):
				if j == 0:
					plt.text(self.rl2[0][j]-15,self.rl2[1][j]-34, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				elif len(self.rl2[0])-j in [4,5,15,23,39,40,41,45,46]:
					plt.text(self.rl2[0][j]+5,self.rl2[1][j]-18, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				elif len(self.rl2[0])-j-1 in [32,33,35]:
					plt.text(self.rl2[0][j]+20,self.rl2[1][j]+10, len(self.rl2[0])-j-1, fontsize=12, horizontalalignment='left', verticalalignment='top')
				else:	
					plt.text(self.rl2[0][j]+12,self.rl2[1][j]+12, len(self.rl2[0])-j-1, fontsize=12)
		if points:
			plt.plot(self.rl2[0],self.rl2[1], "go", markersize=4)
	
		plt.gca().set_aspect('equal')
		plt.title(self.trackname)
		plt.legend(['Center Line', 'Inner Limit', 'Racing Line'])
		plt.savefig(path_,dpi=600,transparent=transparent)
		plt.close('all')
		plt.clf()

	def generateTrackLimits(self):
		self.centerLine = LinearRing(self.coords.reshape(-1, 2))
		# print(self.coords.reshape(-1,2))
		self.innerLimit = self.centerLine.buffer(WIDTH/2, join_style=2, mitre_limit=1).interiors[0]
		self.outerLimit = self.centerLine.buffer(WIDTH/2, join_style=2, mitre_limit=1).exterior
		
	def defineSector(self):
		'''Defines sectors' search space
		
		Parameters
		----------
		center_line : LineString
			Center line of the track
		inside_line : LineString
			Inside line of the track
		outside_line : LineString
			Outside line of the track
		n_secotrs : int
			Number of sectors

		Returns
		-------
		inside_points : list
			List coordinates corresponding to the internal point of each sector segment
		outside_points : list
			List coordinates corresponding to the external point of each sector segment
		'''
		center_line = self.centerLine
		inside_line = self.innerLimit
		outside_line = self.outerLimit

		sect = [0, 35, 69, 81, 95, 114, 128, 140, 158, 172, 183, 196, 207, 246, 262, 283, 293, 310, 326, 338, 355, 390, 435, 463, 479, 494, 506, 520, 575, 620, 680, 701, 716, 733, 745, 762, 780, 789, 797, 812, 830, 848, 867, 882, 899, 914, 930, 960, 0]
		
  		# sect = [0, 35, 70, 82, 95, 114, 125, 140, 155, 170, 185, 195, 207, 245, 260, 284, 292, 310, 326, 338, 355, 400, 454, 479, 494, 506, 515, 575, 620, 690, 701, 716, 737, 748, 762, 780, 790, 797, 805, 820, 840, 866, 886, 898, 914, 930, 960, 0]
		self.nb_sectors = len(sect)
		n_sectors = self.nb_sectors
  
  
		distances = np.linspace(0, center_line.length, n_sectors)

		center_points = [self.pos_x[sect[i]] for i in range(len(sect))],[self.pos_y[sect[i]] for i in range(len(sect))]
		center_points = np.array(center_points).T
  
		distances = np.linspace(0,inside_line.length, 1000)
		inside_border = [inside_line.interpolate(distance) for distance in distances]
		inside_border = np.array([[e.x, e.y] for e in inside_border])
		inside_points = np.array([get_closest_points([center_points[i][0],center_points[i][1]], inside_border) for i in range(len(center_points))]) 

		distances = np.linspace(0,outside_line.length, 1000)
		outside_border = [outside_line.interpolate(distance) for distance in distances]
		outside_border = np.array([[e.x, e.y] for e in outside_border])
		# outside_points = np.array([get_closest_points([center_points[i][0],center_points[i][1]], outside_border) for i in range(len(center_points))])
		outside_points = np.array([get_closest_points([inside_points[i][0],inside_points[i][1]], outside_border) for i in range(len(center_points))])

		self.inside_points = inside_points
		self.outside_points = outside_points



	def getBoundaries(self):
		"""
		Calculates the boundaries of each sector by computing the Euclidean distance between the inside and outside points.
		Stores the boundaries in the 'boundaries' attribute of the object.
		"""
   		# self.boundaries = [np.linalg.norm(np.array(self.inside_points)[:, i] - np.array(self.outside_points)[:, i]) 
		# 					for i in range(self.nb_sectors)]
		for i in range(self.nb_sectors):
			self.boundaries.append(np.linalg.norm(self.inside_points[i]-self.outside_points[i]))
		# print(self.boundaries)
		self.boundaries = [12]*self.nb_sectors 
  



	def getRacingLine(self, sectors):
		"""
		Calculates the racing line for a given set of sectors.

		Args:
			sectors (list): A list of sectors.

		Returns:
			list: A list of points representing the racing line.
		"""
		rl = []
		for i in range(len(sectors)):
			x1, y1 = self.inside_points[i][0], self.inside_points[i][1]
			x2, y2 = self.outside_points[i][0], self.outside_points[i][1]
			m = (y2-y1)/(x2-x1)

			a = math.cos(math.atan(m)) # angle with x axis
			b = math.sin(math.atan(m)) # angle with x axis

			xp = x1 - sectors[i]*a
			yp = y1 - sectors[i]*b
   
			if xp < min([x1, x2]) or xp > max([x1,x2]):
				xp = x1 + sectors[i]*a
				yp = y1 + sectors[i]*b

			rl.append([xp, yp])
			if self.test2:
				self.test.append([xp,yp])
	
		return rl
		
  	
def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b
   
   
  
def get_closest_points(point, array):
	"""
	Returns the closest point(s) in an array to a given point.

	Args:
	point (tuple): A tuple containing the x and y coordinates of the point.
	array (list): A list of tuples, each containing the x and y coordinates of a point.

	Returns:
	list: A list containing the x and y coordinates of the closest point(s) in the array to the given point.
	"""
	result = []
	distance = 1000
	for i in range(len(array)):
		temp = math.sqrt((point[0]-array[i][0])**2+(point[1]-array[i][1])**2)
		if temp<distance:
			distance = temp
			result = [array[i][0], array[i][1]]
	return result


def interpolator(linring, nb):
	"""
	Interpolates a linear ring to obtain a specified number of points.

	Args:
		linring (shapely.geometry.LinearRing): The linear ring to interpolate.
		nb (int): The number of points to interpolate.

	Returns:
		numpy.ndarray: An array of interpolated points.
	"""
	length = linring.length
	step = length / nb
	points = []
	for i in range(nb):
		points.append(linring.interpolate(i*step).coords[0])
	return np.array(points)

def flatten(l):
	result = []
	for el in l:
		result.extend(el)
	return result	
		
def dist(x1,y1,x2,y2):
	return math.sqrt((x1-x2)**2+(y1-y2)**2)
		
		


""" --------------Particle Swarm Optimization algorithm implementation. -----------------------"""


class Particle:
	def __init__(self, dim, boundaries):
		
		self.position = []
		self.velocity = []
		self.b_position =  []
		
		
		for i in range(dim):
			# np.append(self.position,np.random.uniform(0, boundaries[i]))
			self.position.append(np.random.uniform(0, boundaries[i]))
			# np.append(self.velocity,np.random.uniform(-boundaries[i], boundaries[i]))
			self.velocity.append(np.random.uniform(-boundaries[i], boundaries[i]))

		self.position = np.array(self.position)
		self.velocity = np.array(self.velocity)
   
		self.b_position = self.position
		
	def setPosition(self,value):
		self.position = value
		
	def setVelocity(self,value):
		self.velocity = value
		
	def setBestPosition(self,value):
		self.b_position = value
		
def pso_(fitFunc,dim,boundaries, nbParticle, nbIter, w, a, b, track,pbar):
	
	"""
	Particle Swarm Optimization algorithm implementation.
	
	Parameters:
	fitFunc (function): The fitness function to optimize.
	dim (int): The number of dimensions of the search space.
	boundaries (list): A list of tuples representing the boundaries of the search space.
	nbParticle (int): The number of particles in the swarm.
	nbIter (int): The number of iterations to run the algorithm.
	w (float): The inertia weight.
	a (float): The cognitive parameter.
	b (float): The social parameter.
	track (Track): The track object to plot the particles' positions.
	
	Returns:
	tuple: A tuple containing the global best position, the global best fitness value, 
		   the history of global best positions, and the history of global best fitness values.
	"""
	
	# track.plotTrack(centerLine=True,innerLimit=True,outerLimit=True,sectors=False,racingLine=False)
 
	particles = []
	gs = []
	gs_e = []
	# gs_h = []
	# gs_he = []
	
	for i in range(nbParticle):
		particles.append(Particle(dim,boundaries))
	
 
	gs = particles[0].position
	gs_e = fitFunc(gs,track)
	for p in particles:		
		p_eval = fitFunc(p.b_position,track)
		if p_eval < gs_e:
			gs = p.b_position
			gs_e = fitFunc(gs,track)

	# gs_h.append(gs)
	# gs_he.append(gs_e)
	for t in range(nbIter):
		for p in particles:
			e1 = np.random.uniform(0,1)
			e2 = np.random.uniform(0,1)
			newPosition = []
			newVelocity = []
			for d in range(dim):				
	
				newVelocity.append(w * p.velocity[d] + a * e1 * ( p.b_position[d] - p.position[d] ) + b * e2 * ( gs[d] - p.position[d]))
				
				# Check the particles will stay in between the boundaries before computing the positions
				if newVelocity[d] < -boundaries[d]:
					newVelocity[d] = -boundaries[d]
				elif newVelocity[d] > boundaries[d]:
					newVelocity[d] = boundaries[d]
				
				newPosition.append(p.position[d] + newVelocity[d])
			
				# Check if the particle is still in the boundaries
				if newPosition[d] < 0.0 :
					newPosition[d] = 0.0
				elif  newPosition[d] > boundaries[d]:
					newPosition[d] = boundaries[d]

				pbar.update(1)
	
			p.setVelocity(newVelocity)
			p.setPosition(newPosition)
		
			pEval = fitFunc(p.position,track)
			if pEval < fitFunc(p.b_position,track):
				p.setBestPosition(p.position)
				if pEval < gs_e:
					gs = p.position
					gs_e = pEval
	
		# gs_h.append(gs)
		# gs_he.append(gs_e)
		
	return gs, gs_e


def pso_2(fitFunc,dim,boundaries, nbParticle, nbIter, w, a, b, track):
	
	"""
	Particle Swarm Optimization algorithm implementation.
	
	Parameters:
	fitFunc (function): The fitness function to optimize.
	dim (int): The number of dimensions of the search space.
	boundaries (list): A list of tuples representing the boundaries of the search space.
	nbParticle (int): The number of particles in the swarm.
	nbIter (int): The number of iterations to run the algorithm.
	w (float): The inertia weight.
	a (float): The cognitive parameter.
	b (float): The social parameter.
	track (Track): The track object to plot the particles' positions.
	
	Returns:
	tuple: A tuple containing the global best position, the global best fitness value, 
		   the history of global best positions, and the history of global best fitness values.
	"""
	
	# track.plotTrack(centerLine=True,innerLimit=True,outerLimit=True,sectors=False,racingLine=False)
 
	particles = []
	gs = []
	gs_e = []
	# gs_h = []
	# gs_he = []
	
	for i in range(nbParticle):
		particles.append(Particle(dim,boundaries))
	
 
	gs = particles[0].position
	gs_e = fitFunc(gs,track)
	for p in particles:		
		p_eval = fitFunc(p.b_position,track)
		if p_eval < gs_e:
			gs = p.b_position
			gs_e = fitFunc(gs,track)

	# gs_h.append(gs)
	# gs_he.append(gs_e)
	for t in range(nbIter):
		for p in particles:
			e1 = np.random.uniform(0,1)
			e2 = np.random.uniform(0,1)
			newPosition = []
			newVelocity = []
			for d in range(dim):				
	  
  
				newVelocity.append(w * p.velocity[d] + a * e1 * ( p.b_position[d] - p.position[d] ) + b * e2 * ( gs[d] - p.position[d]))
				
				# Check the particles will stay in between the boundaries before computing the positions
				if newVelocity[d] < -boundaries[d]:
					newVelocity[d] = -boundaries[d]
				elif newVelocity[d] > boundaries[d]:
					newVelocity[d] = boundaries[d]
				
				newPosition.append(p.position[d] + newVelocity[d])
			
				# Check if the particle is still in the boundaries
				if newPosition[d] < 0.0 :
					newPosition[d] = 0.0
				elif  newPosition[d] > boundaries[d]:
					newPosition[d] = boundaries[d]

	
			p.setVelocity(newVelocity)
			p.setPosition(newPosition)
		
			pEval = fitFunc(p.position,track)
			if pEval < fitFunc(p.b_position,track):
				p.setBestPosition(p.position)
				if pEval < gs_e:
					gs = p.position
					gs_e = pEval
	
		# gs_h.append(gs)
		# gs_he.append(gs_e)
		
	return gs, gs_e



def pso(fitFunc,dim,boundaries, nbParticle, nbIter, w, a, b, track):
	
	"""
	Particle Swarm Optimization algorithm implementation.
	
	Parameters:
	fitFunc (function): The fitness function to optimize.
	dim (int): The number of dimensions of the search space.
	boundaries (list): A list of tuples representing the boundaries of the search space.
	nbParticle (int): The number of particles in the swarm.
	nbIter (int): The number of iterations to run the algorithm.
	w (float): The inertia weight.
	a (float): The cognitive parameter.
	b (float): The social parameter.
	track (Track): The track object to plot the particles' positions.
	
	Returns:
	tuple: A tuple containing the global best position, the global best fitness value, 
		   the history of global best positions, and the history of global best fitness values.
	"""
	
	# track.plotTrack(centerLine=True,innerLimit=True,outerLimit=True,sectors=False,racingLine=False)
 
	particles = []
	gs = []
	gs_e = []
	# gs_h = []
	# gs_he = []
	
	for i in range(nbParticle):
		particles.append(Particle(dim,boundaries))
	
 
	gs = particles[0].position
	gs_e = fitFunc(gs,track)
	for p in particles:		
		p_eval = fitFunc(p.b_position,track)
		if p_eval < gs_e:
			gs = p.b_position
			gs_e = fitFunc(gs,track)

	# gs_h.append(gs)
	# gs_he.append(gs_e)
	with tqdm(total=nbIter*len(particles)*dim) as pbar:
		for t in range(nbIter):
			# for i in range(nbParticle):
			for p in particles:
				e1 = np.random.uniform(0,1)
				e2 = np.random.uniform(0,1)
				newPosition = []
				newVelocity = []
				for d in range(dim):				
		
					newVelocity.append(w * p.velocity[d] + a * e1 * ( p.b_position[d] - p.position[d] ) + b * e2 * ( gs[d] - p.position[d]))
					# p.setVelocity(p.velocity + a*e1*(gs -  p.position) + b*e2*(p.b_position - p.position))
					
					# Check the particles will stay in between the boundaries before computing the positions
					if newVelocity[d] < -boundaries[d]:
						newVelocity[d] = -boundaries[d]
					elif newVelocity[d] > boundaries[d]:
						newVelocity[d] = boundaries[d]
					
					newPosition.append(p.position[d] + newVelocity[d])
					# p.setPosition(p.position + p.velocity)
				
					# Check if the particle is still in the boundaries
					if newPosition[d] < 0.0 :
						newPosition[d] = 0.0
					elif  newPosition[d] > boundaries[d]:
						newPosition[d] = boundaries[d]

					pbar.update(1)
		
				p.setVelocity(newVelocity)
				p.setPosition(newPosition)					
						
				pEval = fitFunc(p.position,track)
				if pEval < fitFunc(p.b_position,track):
					p.setBestPosition(p.position)
					if pEval < gs_e:
						gs = p.position
						gs_e = pEval
		
			# gs_h.append(gs)
			# gs_he.append(gs_e)
		pbar.close()
		
	return gs, gs_e



def fitFunc(sectors,track):
	"""
	Calculates the lap time for a given set of sectors.

	Returns:
		float: The lap time for the given sectors.
	"""
 
	# if track.test2:
		# print("sectors : ")
		# print(sectors)
		# print(len(sectors))
	a = getLapTime(track.getRacingLine(sectors))[0]
	track.test2 = False
	return a
	
def getLapTime(racingLine):
	"""
	Computes the lap time and the (x, y) coordinates of the racing line.
	
	Parameters:
	racingLine (list): A list of (x, y) coordinates of the racing line.
	
	Returns:
	tuple: A tuple containing the lap time and the (x, y) coordinates of the racing line.
	"""
	rl = np.array(racingLine)
	
	
	# Find the spline
	tck, _ = interpolate.splprep([rl[:,0], rl[:,1]],s=0, k=3)
	# Evaluate the spline
	x, y = interpolate.splev(np.linspace(0, 1, 2000), tck)
	
	# Compute the derivative
	dx, dy = np.gradient(x), np.gradient(y)
	d2x, d2y = np.gradient(np.array(dx)), np.gradient(np.array(dy))
	
	k = np.abs(dx*d2y - dy*d2x) / np.power(dx*dx + dy*dy, 1.5)
	r = 1/k
	
	# µ = 0.7
	µ = 1.78
	
	# computing the max speed
	v = np.sqrt(µ * r * 9.81)
	v = np.clip(v,None, 95.66)
	# print(v)
	
	# computing the lap time	
	lapTime = np.sum(np.sqrt(np.power(np.diff(x), 2) + np.power(np.diff(y), 2)) / v[:-1])
	 
	return lapTime, (x, y)
	

def main():
	nbpart = 10
	nbiter = 10
	path = f"Data/plot_output/{time.strftime('%d-%m-%Y')}/%s/"
	path = next_path(path)
	w = 0.8
	a = 1
	b = 0.2
	print(f"""
- nbIteration : {nbiter}
- nbParticule : {nbpart}
- w : {w}
- c1 : {a}
- c2 : {b}
       """)
	gs ,gs_e = pso(fitFunc,track.nb_sectors,track.boundaries,nbpart, nbiter,w,a,b,track)
	rl = track.getRacingLine(gs)
	gs_e = getLapTime(rl)[0]
	track.racingLine = getLapTime(rl)[1]
	track.rl2 = [[rl[i][0] for i in range(len(rl))],[rl[i][1] for i in range(len(rl))]]
	# print(rl)
	print(f"""
- Global best position : {gs}
- Lap time : {gs_e}
- Parameters : {w}, {a}, {b}
       """)
 
 
	gs = [6.0 for _ in range(len(gs))]
	rl = track.getRacingLine(gs)
	gs_e = getLapTime(rl)[0]
	print(f"""
- Global best position : {gs}
- Lap time : {gs_e}
- Parameters : {w}, {a}, {b}
       """)

	path="Img/"
	track.plotTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True)
	# track.saveTrack(transparent=False, path=path, sectors=True)
	# track.saveTrack(transparent=True, path=path, sectors=True)
	# track.saveTrack(centerLine=False, transparent=True, path=path)
	# track.saveTrack(centerLine=False, transparent=False, path=path)
 
	
 
	track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=False, path=path)
	# track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=True, path=path)
	# track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=False,points=True,transparent=False, path=path)
	
	with open(f"{path}data.txt","w") as file:
		file.write(str([gs_e,gs,rl]))
		file.close()


def main2():
	track = Track("Data/Track_data/cbg.geojson")
	nbpart = 50
	nbiter = 50
	print(f"nbiter : {nbiter}, nbpart : {nbpart}")
	path = f"Data/plot_output/{time.strftime('%d-%m-%Y')}/"
	try:
		os.mkdir(path)
	except:
		pass
	n = 100
	with tqdm(total=n,position=0) as pbar:
		with tqdm(total=0, position=1, bar_format='{desc}') as pbar2:
 
			for i in range(n):
				# print(f"i : {i}")
				pbar.set_description(f" n°{i} ")
				path_ = next_path(f"{path}%s/")
				try:
					os.mkdir(path_)
				except:
					pass
				pbar2.set_description(f"Calcul de la meilleure trajectoire de {path_}...")
				gs ,gs_e = pso_2(fitFunc,track.nb_sectors,track.boundaries,nbpart, nbiter,0.8,1,0.20,track)
				rl = track.getRacingLine(gs)
				track.racingLine = getLapTime(rl)[1]
				track.rl2 = [[rl[i][0] for i in range(len(rl))],[rl[i][1] for i in range(len(rl))]]
				pbar2.set_description(f"Sauvegarde de {path_}data.txt ...")
				with open(f"{path_}data.txt","w") as file:
					file.write(str([gs_e,gs,rl]))
					file.close()
				pbar2.set_description(f"Sauvegarde des plots de {path_} ...")
				# track.plotTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True)
				track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=False, path=path_)
				# track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=True, path=path_)
				track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=False,points=True,transparent=False, path=path_)
				pbar2.set_description(f"Sauvegarde des plots n°{i} ... OK")
				pbar.update(1)
    
	
def openData():
	file = open("Data/Results/2/2/Circuit Paul Ricard.txt","r")
	n,gs_e,gs,rl = list(eval(file.read())[7])
	print(n,gs_e)
	track.racingLine = getLapTime(rl)[1]
	track.rl2 = [[rl[i][0] for i in range(len(rl))],[rl[i][1] for i in range(len(rl))]]
	track.plotTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True)
	track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=False)
	track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=True,points=True,transparent=True)
	track.saveTrack(centerLine=False,sectors=False,racingLine=True,nb=False,points=True,transparent=False)
	file.close()
	
    


if __name__ == "__main__":    
	track = Track("Data/Track_data/cbg.geojson")
	main()
	# main2()