#Author: Gabriel Mesquita Nespoli
#Student of Masters in Data Science at Università Sapienza di Roma

import csv
import numpy as np
import math
import gradientdescent-lib
import tkinter

def normalize(wines):
	print(wines)
	meanWinterRain = np.average(wines[:,2])
	meanAGST = np.average(wines[:,3])
	meanHarvestRain = np.average(wines[:,4])
	meanAge = np.average(wines[:,5])
	meanFrancePop = np.average(wines[:,6])

	varWinterRain = math.sqrt(sum([(w - meanWinterRain)**2 for w in wines[:,2]])/nLines)
	varAGST = math.sqrt(sum([(a - meanAGST)**2 for a in wines[:,3]])/nLines)
	varHarvestRain = math.sqrt(sum([(h - meanHarvestRain)**2 for h in wines[:,4]])/nLines)
	varAge = math.sqrt(sum([(age - meanAge)**2 for age in wines[:,5]])/nLines)
	varFrancePop = math.sqrt(sum([(f - meanFrancePop)**2 for f in wines[:,6]])/nLines)

	for l in range(0,nLines):
		for c in range(0,nColumns):
			if c == 2:
				wines[l][c] = (wines[l][c] - meanWinterRain)/varWinterRain
			if c == 3:
				wines[l][c] = (wines[l][c] - meanAGST)/varAGST
			if c == 4:
				wines[l][c] = (wines[l][c] - meanHarvestRain)/varHarvestRain
			if c == 5:
				wines[l][c] = (wines[l][c] - meanAge)/varAge
			if c == 6:
				wines[l][c] = (wines[l][c] - meanFrancePop)/varFrancePop
	
f = open('./wine.csv')
winesList = []
nLines = 0
nColumns = 0
next(f)
for line in csv.reader(f):
	winesList = winesList + line
	nLines += 1
	nColumns = len(line)
wines = np.array(winesList, dtype='f')
wines = wines.reshape((nLines,nColumns))
normalize(wines)

#AGPS
x = np.c_[ np.ones(nLines), wines[:,3]]
y = wines[:,1]
theta = mylib.descent(y, x)
print(theta)
print("R2 = " + str(mylib.r2(y,theta,x)))
mylib.plotGradientDescent(x,y,theta)

#4 VARIABLES
x = np.c_[ np.ones(nLines), wines[:,2:6]]
y = wines[:,1]
theta = mylib.descent(y, x)
print(theta)
print("R2 = " + str(mylib.r2(y,theta,x)))
