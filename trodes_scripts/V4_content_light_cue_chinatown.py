import math
import struct
import re
import time
import random
import numpy as np
import pyaudio
import wave
# V8pre_forage
# visits to incorrect wells cause 5s lockout
# exception is repeat visit to prior well (is ok, no lockout)
# can go to any outer well, any number of times
# lockout from getting rip/wait wells wrong is also 5s


# decide what type of up trigger was just recieved; act accordingly
# only for home and outer, center well defined in statescript
def pokeIn(dio):
	global homeWell
	global centerWell
	global outerWells
	global currWell
 
	currWell = int(dio[1])
	if currWell == homeWell: 
		doHome()

	for num in range(len(outerWells)):
		if currWell == outerWells[num]:
			doOuter(num)

# decide what type of down trigger was just recieved; act accordingly
def pokeOut(dio):
	global homeWell
	global centerWell
	global outerWells
	global currWell
	global lastWell
 
	currWell = int(dio[1])
	if currWell == homeWell: 
		endHome()
	if currWell == centerWell: 
		endWait()    
	for num in range(len(outerWells)):
		if currWell == outerWells[num]:
			endOuter()
	lastWell = currWell

#home poke: decide trial type and upcoming wait length; turn on lights accordingly
def doHome():
	global trialtype  # 0 go to home,1 go to center, 2 go to outer, 3 lockout
	global homePump
	global lastWell
	global currWell
	global goalWell
	global outerWells

	if trialtype == 0:	
		trialtype = 1
		print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")  
		#delaytime = chooseDelay()
		# set goal well to 1 arm for this trial - get line from old script
		goalWell = np.random.choice(outerWells,1,replace=False)
		print("SCQTMESSAGE: homeCount = homeCount + 1;\n") # update homecount in SC
		print("SCQTMESSAGE: rewardWell = "+str(homePump)+";\n")
		print("SCQTMESSAGE: trigger(1);\n")   # deliver reward
	#check for home poke out of sequence, start lockout 1
	elif trialtype > 0 and trialtype < 3 and lastWell != currWell:
		lockout([0,1])

# def chooseDelay():
#     global trialtype
#     global centercount
#     global waitdist
#     global startwaitdist

#     #print(centercount)

#     if centercount<3:  #first 3 trials of of each type should be short
#         return startwaitdist[centercount]

#     else:
#         if centercount<=10:  #trials 4-10 of each type will be avg of startwaitdist and normal waitdist
#             return int(round(np.mean([int(np.random.choice(startwaitdist,1)), int(np.random.choice(waitdist,1))])))

#         else: # all trial 10 and later
#             return int(np.random.choice(waitdist,1))


def endHome():
	global trialtype
	global lastWell
	global homeWell
	global centerWell

	if trialtype == 1:
		print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")   # turn on light in center well
		print("SCQTMESSAGE: trigger(3);\n")
		print("SCQTMESSAGE: dio = "+str(homeWell)+";\n")
		print("SCQTMESSAGE: trigger(4);\n")
		print("SCQTMESSAGE: trigger(5);\n")   # display stats

#function: add time to wait dist. 
def addtime(newtime):
	global count
	global waitdist

	count+=1
	# % means mod (reaminder) function
	waitdist[count%8] = int(newtime[1])  #new time 1 will be timediff, not timestamp
	print(waitdist)

def beep():
	global centerPump
	global centerWell
	global trialtype
	global currWell
	global centercount

	centercount+=1
	trialtype = 2                   # ready for outer visit
	print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
	#deliver reward
	if currWell == centerWell:
		print("SCQTMESSAGE: rewardWell = "+str(centerPump)+";\n")
	print("SCQTMESSAGE: trigger(1);\n")
	print("SCQTMESSAGE: centerCount = centerCount + 1;\n") # update centercount in SC

def endWait():
	global trialtype
	global goalWell
	global currWell
	global outerWells

	if trialtype == 2:   # wait complete

		# check for required number of visits to each outer arm here
		# if number of visits > 3 then set goal arm from spykshrk output to statescript

		print("SCQTMESSAGE: dio = "+str(currWell)+";\n")     # turn off rip light
		print("SCQTMESSAGE: trigger(4);\n")

		# turn on light for goalWell only
		print("SCQTMESSAGE: dio = "+str(goalWell[0])+";\n")
		print("SCQTMESSAGE: trigger(3);\n")

		print("SCQTMESSAGE: trigger(5);\n")   # display stats
		print("SCQTMESSAGE: disp('CURRENTGOAL IS "+str(goalWell)+"');\n") 


def doOuter(val):
	global outerPumps
	global trialtype
	global allGoal
	global goalWell 
	global currWell
	global lastWell
	global homeWell
	global waslock

	if trialtype == 2:
		trialtype = 0      # outer satisfied, head home next
		print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
		if currWell in goalWell :  # repeated; reward
			print("SCQTMESSAGE: rewardWell = "+str(outerPumps[val])+";\n")
			print("SCQTMESSAGE: trigger(2);\n")   # deliver reward
			allGoal+=1
			# create and add to counter for each of the 4 outer arms
			if arm1:
				arm1Reward+=1
			elif arm2:
				arm2Reward+=1
			elif arm3:
				arm3Reward+=1
			elif arm4:
				arm4Reward+=1
				
			print("SCQTMESSAGE: goalTotal = "+str(allGoal)+";\n") # update goaltotal in SC

		else:   # wrong well; add to forage record if newly visited
			print("SCQTMESSAGE: otherCount = otherCount + 1;\n") # update othercount in SC

	elif trialtype < 2 and waslock<1:
		lockout([0,1])

def endOuter():
	global trialtype
	global outerWells
	global homeWell
	global lastWell
	global currWell

	if trialtype == 0 and lastWell != currWell:  # outer satisfied
		for num in range(len(outerWells)):			# turn off outer lights
			print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
			print("SCQTMESSAGE: trigger(4);\n")
		print("SCQTMESSAGE: dio = "+str(homeWell)+";\n")   # turn homewell on
		print("SCQTMESSAGE: trigger(3);\n")
		print("SCQTMESSAGE: trigger(5);\n")   # display stats

def lockout(val):   # turn off all lights for certain amount of time
	global centerWell
	global outerWells
	global lastWell
	global trialtype
	global waslock

	print("lockout val "+str(val)+"\n")
	locktype = int(val[1])
	trialtype = 3
	print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
	print("SCQTMESSAGE: trigger(6);\n")  # start lockout timer in SCQTMESSAGE
	#turn off all lights
	print("SCQTMESSAGE: dio = "+str(homeWell)+";\n") # turn off home well
	print("SCQTMESSAGE: trigger(4);\n")
	print("SCQTMESSAGE: dio = "+str(centerWell)+";\n")  #turn off all center and outer well lights
	print("SCQTMESSAGE: trigger(4);\n")
	for num in range(len(outerWells)):
		print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
		print("SCQTMESSAGE: trigger(4);\n")
	waslock=1
	print("SCQTMESSAGE: waslock = "+str(waslock)+";\n") # turn off home well
	if locktype == 1:
		print("SCQTMESSAGE: locktype1 = locktype1 + 1;\n") # type 1 = wrong well order
	if locktype == 2:
		print("SCQTMESSAGE: locktype2 = locktype2 + 1;\n") # type 2 = impatience at center well

def lockend():
	global trialtype
	global homeWell

	trialtype = 0
	print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
	print("SCQTMESSAGE: dio = "+str(homeWell)+";\n") # turn on home well
	print("SCQTMESSAGE: trigger(3);\n")

def updateWaslock(val):
	global waslock

	waslock = int(val[1])
	print("SCQTMESSAGE: waslock = "+str(waslock)+";\n")

# Function: generate cowbell sound
def generate_beep():

	File='Beep.wav'
	#File='noise.wav'
	spf = wave.open(File, 'rb')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	p = pyaudio.PyAudio()
	stream = p.open(format =
				p.get_format_from_width(spf.getsampwidth()),
				channels = 1,
				rate = spf.getframerate(),
				output = True)
	#play 
	data = struct.pack("%dh"%(len(signal)), *list(signal))    
	stream.write(data)
	stream.close()
	p.terminate()

def makewhitenoise():  #play white noise for duration of lockout
	global locksoundlength

	soundlength = int(44100*locksoundlength/1000)
	p = pyaudio.PyAudio()
	stream = p.open(format = 8, channels = 1, rate = 44100, output = True)
	whitenoise = np.random.randint(700,size = soundlength)
	data = struct.pack("%dh"%(len(whitenoise)), *list(whitenoise))    
	stream.write(data)
	stream.close()
	p.terminate()


# This is the custom callback function. When events occur, addScQtEvent will
# call this function. This function MUST BE NAMED 'callback'!!!!
def callback(line):

	global waslock 

	if line.find("UP") >= 0: #input triggered
		pokeIn(re.findall(r'\d+',line))
	if line.find("DOWN") >= 0: #input triggered
		pokeOut(re.findall(r'\d+',line))
	# add ripwait to holding vector
	if line.find("riptime") >=0:
		addtime(re.findall(r'\d+',line))
	if line.find("BEEP 1") >= 0: # make a beep and deliver reward
		beep()
	if line.find("BEEP 2") >= 0: # make a beep and deliver reward
		generate_beep()
	if line.find("LOCKOUT") >= 0: # lockout procedure
		lockout(re.findall(r'\d+',line))
	if line.find("LOCKEND") >= 0: # reset trialtype to 0
		lockend()
	if line.find("WHITENOISE") >= 0: # make noise during lockout
		makewhitenoise()
	if line.find("waslock") >= 0:  #update waslock value
		updateWaslock(re.findall(r'\d+',line))


# all variables are initialized - this applies to first trial or first contigency, etc
# define wells
homeWell = 1
centerWell = 2
outerWells = [10,11,12,13]

# define pumps
homePump = 25
centerPump = 26
outerPumps = [19, 20, 21, 22]

#global variables 
lastWell = -1
currWell = -1
trialtype = 0


allGoal = 0

# choose one well at random of the 4
goalWell = 0

waitdist = [0, 0, 0, 0, 0, 0, 0, 0]
count = 0

locksoundlength = 1000
print(goalWell)
waslock=0
centercount = 0
