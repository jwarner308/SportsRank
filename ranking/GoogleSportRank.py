import numpy as np
import math
import optparse
import sys

class EigenvectorSolver:

	def __init__(self,tol_=0.001,maxIter_=10):
		self.tol = tol_
		self.maxIter = maxIter_

	def computeLeftEigenvector(self, A):
		#A is a numpy array, calculate and return dominant left eigenvector
		if(not A.ndim == 2):
			print "ERROR - number of dimensions of A is not 2!"
		
		Asize = A.shape
		if(not Asize[0]==Asize[1]):
			print "ERROR - matrix is not square!"

		#Initial guess vector:
		b_prev = np.ones(Asize[0]) * 1.0/Asize[0]

		#Iterate until convergence of max iterations:
	#	print "Computing eigenvector..."
		numIter = 1;
		normChange = 100;
		while( (numIter < self.maxIter) and (normChange > self.tol)):
			b_new = np.dot(b_prev, A)
			#Compute norm of difference in eigenvector:
			diff = b_new - b_prev
			normChange = np.linalg.norm(diff)
			#Prepare for next iteration:
			b_prev = b_new
			numIter +=1
			#print numIter, normChange

		return b_new

class GoogleRanker:
	
	def __init__(self, dataFile_, outFile_='ranks.txt', alpha_=0.85, 
			 			tol_=1e-4, maxIter_=30,useHomeFactor_=False, 
						homeFactor_ = None, numGames_=1e5):
		#dataFile_ - name of input file; outFile_ - name of output ranking file
		#alpha_ - free parameter in Google ranking; #tol/maxIter - stopping for EV solver
		#numGames - # of games to use for ranking out of score file

		#Google ranker variables:
		self.dataFile	   = dataFile_;
		self.outFile  	   = outFile_;
		self.alpha    	   = alpha_
		self.useHomeFactor = useHomeFactor_
		self.numGames      = numGames_
		self.homeFactor    = homeFactor_

		#For increasing significance of last games:
		self.num_wks = 6
		self.factor  = 1.75

		#Initialize eigenvector solver:
		self.EVsolver = EigenvectorSolver(tol_, maxIter_)

	def computeHomeCourtFactor(self):
	
		#Compute the average margin of victory of the home team over away

		if(not self.homeFactor==None):
			return self.homeFactor
		else:
			diffSum = 0.0
			gameCount = 0.0;
			for gameData in self.scoreData:
				gameCount+=1.0
				scoreDiff = gameData[3]-gameData[1]
				diffSum += scoreDiff
		
			avgDiff  = diffSum / gameCount
			print "Computed home factor = " , avgDiff
			return avgDiff

	def readScoreDataFile(self):
		#Open score data file for reading:
		fileIn = open(self.dataFile,'r')
		rawScoreData = fileIn.readlines()
		#Extract CSV formatted data into list of lists & build map from team Name to ID:
		self.scoreData = []; self.teamList = []; self.teamToIDMap = {}
		teamCount = 0 
		for count,line in enumerate(rawScoreData):

			#Only use # specified games for ranking:
			if(count>=self.numGames):
				break;

			#convention -  game: [ team_away, score_away, team_home, score_home]
			game = line.split(',')   
			#Add teams to data structures if they haven't been encountered:
			for i in [0,2]:
				if(not game[i] in self.teamToIDMap):
					teamCount += 1
					self.teamToIDMap[game[i]] = teamCount
					self.teamList.append(game[i])
			#Add team IDs &  score info to data structure:
			self.scoreData.append([self.teamToIDMap[game[0]], int(game[1]), 
					self.teamToIDMap[game[2]], int(game[3])])

		#Store number of teams considered:
		self.numTeams = teamCount
		print "Number of teams considered: ",self.numTeams
		#Close score data file
		fileIn.close()

	def initializeRanker(self):
		
		#Read score data from input file:
		self.readScoreDataFile()
		
		#Compute avg. margin of victory for home team & factor into game outcomes 
		if(self.useHomeFactor):
			self.homeFactor = self.computeHomeCourtFactor()
		else:
			self.homeFactor = 0
		
	def getPointsLostBySums(self):

		#Sum the total points lost by through season for each team
		ptsLostBySums = np.zeros(self.numTeams)
		for i,gameData in enumerate(self.scoreData):
			scoreDiff = gameData[1]-gameData[3] + self.homeFactor
			# neg. diff. means the away team lost  & pos. means home team lost
			if(scoreDiff < 0):
				loseID = gameData[0]
			else:
				loseID = gameData[2]

			if(i > (len(self.scoreData) - self.num_wks*16)):
				factor = self.factor
			else:
				factor = 1
			#Sum the points lost by for losing team:
 
			ptsLostBySums[ loseID - 1 ] += factor*math.fabs(scoreDiff)
		
		return ptsLostBySums

	def computeGoogleSportMatrix(self):

		ptsLostBySums = self.getPointsLostBySums()
		
		adjacencyMatrix = self.generateAdjacencyMatrix(ptsLostBySums)
		# need to check for undefeated team / zero row
		Emat = np.ones( (self.numTeams, self.numTeams) )*(1.0/self.numTeams)

		#Form google score matrix as linear combo. of adjacency & Emat:
		self.googleMatrix = self.alpha*adjacencyMatrix + (1.0-self.alpha)*Emat

	def generateAdjacencyMatrix(self, ptsLostBySums):
		
		adjMat = np.zeros( (self.numTeams, self.numTeams) )
		
		#build adjacency matrix from score data
		for i, gameData in enumerate(self.scoreData): #[A_ID A_score H_ID H_score]
			scoreDiff = gameData[1] - gameData[3] + self.homeFactor
			if(scoreDiff < 0):  #away team loses
				loseID = gameData[0]-1
				winID  = gameData[2]-1
			else:
				loseID = gameData[2]-1
				winID  = gameData[0]-1
			
			if(i > (len(self.scoreData) - self.num_wks*16)):
	#			print " i = ", i , " factorz"
				factor = self.factor
			else:
				factor = 1

			adjMat[loseID][winID] += factor*float(math.fabs(scoreDiff))/float(ptsLostBySums[loseID])

		#Account for undefeated teams/ zero rows 
		for i in range(0,self.numTeams):
			if ptsLostBySums[i]==0:
				#print "Undefeated team found: ", self.teamList[i]
				for j in range(0,self.numTeams):
					adjMat[i][j] = 1.0/self.numTeams


		return adjMat

	def writeRankedTeamsToFile(self, rankVec):

		teamOrder = rankVec.argsort()
		rankedIndices = teamOrder.argsort()
		
		orderedTeams = np.zeros( self.numTeams, dtype=int )

		for i in range(0, self.numTeams):
			ind = rankedIndices[i]
			orderedTeams[ind] = i

		fileOut = open(self.outFile,'w')
		#For printing actual rank magnitudes to file
		fileOut2 = open("rankMagnitudes.txt",'w');
		rankVec.sort()

		for i in range(0, self.numTeams):
			print >> fileOut, i+1, ",",  self.teamList[orderedTeams[self.numTeams-i-1]]
			print >> fileOut2, rankVec[self.numTeams-i-1], "\t",  self.teamList[orderedTeams[self.numTeams-i-1]]

		fileOut.close()

	def getTeamToStrengthMap(self, rankVec):

		team2Strength = {}
		for i in range(0,self.numTeams):
			team2Strength[self.teamList[i]] = rankVec[i]
		return team2Strength


	def computeGoogleSportRank(self):
		
		#Initialize - read score data file, set/allocate  member variables
		self.initializeRanker()
		#Generate the google matrix based on team interactions:
		self.computeGoogleSportMatrix()
		#Compute rank vector (left eigenvector of the google matrix):
		rankVector = self.EVsolver.computeLeftEigenvector(self.googleMatrix)*100
		#Order teams based on rank vector & print to file:
		self.writeRankedTeamsToFile(rankVector)
		rankMap = self.getTeamToStrengthMap(rankVector)
		return rankMap


#=================================================================
#Driver:

    #Driver:
if __name__ == "__main__":

    #Set up command line arguments:
	parser = optparse.OptionParser()
	parser.add_option("-d", "--dataFile", type="string", dest="dataFile")
	parser.add_option("-t", "--tol", type="float", dest="tol")
	parser.add_option("-n", "--numIters", type="int", dest="iters")
	parser.add_option("-a", "--alpha", type="float", dest="alpha")
	parser.add_option("-u", "--useHomeFactor", action="store_true", dest="homeFlag",default=False)
	parser.add_option("-f", "--homeFactor",type="float",dest="homeFactor")
	parser.add_option("-g", "--numGames", type="int",dest="numGames")
	parser.add_option("-o", "--outfile", type="string", dest="outFile")
    #Default values if not specified:
	parser.set_defaults(tol=1e-8,iters=30,alpha=0.5,homeFactor=None,outFile="ranks.txt",numGames=1e5)
	opts, args = parser.parse_args()

	if(opts.dataFile==None):
		print("ERROR: must specify a score data file at the command line with -d flag!")
		sys.exit()

    #Initialize ranker & generate rankings:
	g = GoogleRanker(opts.dataFile, opts.outFile, opts.alpha, opts.tol, opts.iters, opts.homeFlag, opts.homeFactor, opts.numGames)
	g.computeGoogleSportRank()

