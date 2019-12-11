import math
import struct
import re
import time
import random
import numpy as np
import pyaudio
import wave
from statistics import mean

# V8pre_forage
# visits to incorrect wells cause 5s lockout
# exception is repeat visit to prior well (is ok, no lockout)
# can go to any outer well, any number of times
# lockout from getting rip/wait wells wrong is also 5s


# decide what type of up trigger was just recieved; act accordingly
def pokeIn(dio):
    global homeWell
    global waitWells
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
    global waitWells
    global outerWells
    global currWell
    global lastWell
 
    currWell = int(dio[1])
    if currWell == homeWell: 
        endHome()
    for num in range(len(waitWells)):
        if currWell == waitWells[num]:
            endWait()
    for num in range(len(outerWells)):
        if currWell == outerWells[num]:
            endOuter()
    lastWell = currWell

#home poke: decide trial type and upcoming wait length; turn on lights accordingly
def doHome():
    global trialtype  # 0 home,1 waitR, 2 waitL, 3 lockout
    global homePump
    global lastWell
    global currWell
    global waitdist
    global homeCount

    if trialtype == 0:	
        opts = [1, 2]        
        randnum = np.random.randint(0,2)  #-1
        trialtype = opts[randnum] 
        #trialtype = np.random.randint(1,3)  #set upcoming trialtype to 1 or 2
        print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")  
        delaytime = chooseDelay()
        print("SCQTMESSAGE: waittime = "+str(delaytime)+";\n")
        print("SCQTMESSAGE: rewardWell = "+str(homePump)+";\n")
        print("SCQTMESSAGE: trigger(1);\n")   # deliver reward
        homeCount+=1
    elif trialtype > 0 and trialtype < 4 and lastWell != currWell:
        lockout([2,2])

# this will only apply to wait trials 
def chooseDelay():
    global trialtype
    global waitCount
    global waitdist
    global startwaitdist

    if waitCount<3:  #first 3 trials of of each type should be short
        return startwaitdist[waitCount]

    else:
        if waitCount<=5:  #trials 4-10 of each type will be avg of startwaitdist and normal waitdist
            return round(mean([int(np.random.choice(startwaitdist,1)), int(np.random.choice(waitdist,1))]))

        else: 
            return int(np.random.choice(waitdist,1))

def endHome():
    global trialtype
    global lastWell
    global homeWell
    global waitWells

    if trialtype == 1 or trialtype == 2:
        print("SCQTMESSAGE: dio = "+str(waitWells[trialtype-1])+";\n")   # turn on the correct wait well 
        print("SCQTMESSAGE: trigger(3);\n")
        print("SCQTMESSAGE: dio = "+str(homeWell)+";\n")
        print("SCQTMESSAGE: trigger(4);\n")
        printStats()

#function: add time to wait dist. 
def addtime(newtime):
    global count
    global waitdist

    if int(newtime[1]) > 2000:  #only add times > 2 sec (otherwise, likely just false pos time)
        count+=1
        waitdist[count%8] = int(newtime[1])  #new time 1 will be timediff, not timestamp
        print(waitdist)
        #return waitdist, count

def decreasewaitdist(num):   #each time waitdelay is called, reduce a random wait by ripwin (1000)
    global waitdist
    randnum = np.random.randint(0,8)
    if waitdist[randnum]>int(num[1])*1000:
        waitdist[randnum] = waitdist[randnum]-int(num[1])*1000  


def click():
    global waitPumps
    global waitWells
    global trialtype
    global currWell
    global waitCount
    
    #generate_click()
    trialtype = 3  # ready for outer visit
    print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
    #deliver reward
    for num in range(len(waitWells)):
        if currWell == waitWells[num]:
            print("SCQTMESSAGE: rewardWell = "+str(waitPumps[num])+";\n")
    print("SCQTMESSAGE: trigger(1);\n")
    waitCount+=1
    
def beep():
    global waitPumps
    global waitWells
    global trialtype
    global currWell
    global ripCount
   
    #generate_beep()
    trialtype = 3                   # ready for outer visit
    print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
    #deliver reward
    for num in range(len(waitWells)):
        if currWell == waitWells[num]:
            print("SCQTMESSAGE: rewardWell = "+str(waitPumps[num])+";\n")
    print("SCQTMESSAGE: trigger(1);\n")
    ripCount+=1

def endWait():
    global trialtype
    global goalWell
    global currWell
    global outerWells

    if trialtype == 3:   # wait complete
        print("SCQTMESSAGE: dio = "+str(currWell)+";\n")     # turn off rip light
        print("SCQTMESSAGE: trigger(4);\n")
        for num in range(len(outerWells)):          # turn on outer lights
            print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
            print("SCQTMESSAGE: trigger(3);\n")
        print("SCQTMESSAGE: disp('CURRENTGOAL IS "+str(goalWell)+"');\n") 
        printStats()

def doOuter(val):
    global outerPumps
    global trialtype
    global allGoal
    global thisGoal
    global goalWell 
    global oldGoals 
    global outerReps
    global currWell
    global lastWell
    global homeWell
    global otherCount
    global waslock
   
    if trialtype == 3:
        trialtype = 0      # outer satisfied, head home next
        print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
        if currWell in goalWell :  # repeated; reward
            print("SCQTMESSAGE: rewardWell = "+str(outerPumps[val])+";\n")
            print("SCQTMESSAGE: trigger(2);\n")   # deliver reward
            allGoal+=1
            thisGoal+=1

            if thisGoal >= outerReps:  #maxed repeats reached, time to switch
                print("time to choose new goals!")
                chooseGoal()   # this version currently only allows there to be one goal
                thisGoal = 0
                print("SCQTMESSAGE: goalCount = "+str(thisGoal)+";\n") # update goalcount in SC

            if numgoals == 1 and thisGoal == 1:  # set the goal as the one he found (part of forage assist)
                goalWell = [currWell]

        else:   # wrong well; add to forage record if newly visited
            otherCount+= 1

    elif trialtype < 3 and waslock<1:
        lockout([2,2])

def chooseGoal():    #to assist forage, choose 2 goals; first encountered will turn into only goal
    global goalWell
    global oldGoals
    global numgoals
    global outerWells
    global forageNum
    global outerReps

    oldGoals.append(goalWell)
    print(oldGoals)
    if len(oldGoals)>7:   # if all outers have been goal, reset
        print("resetting oldgoals")
        oldGoals = [None]
    
    if len(oldGoals)> (8-forageNum):   # make sure that forageassist doesnt run out of new arms
        forageNum = 1

    goalWell = np.random.choice(outerWells,forageNum,replace=False) 
    while any(i in goalWell for i in oldGoals):  #if any common elements 
        goalWell = np.random.choice(outerWells,forageNum,replace=False) 
        print(goalWell)

    outerReps = np.random.randint(4,8) 
    #print("SCQTMESSAGE: disp('outerreps = "+str(outerReps)+"');\n") 
    print("new goal is "+str(goalWell)+"\n")

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
        printStats()   # display stats

def lockout(val):   # turn off all lights for certain amount of time
    global waitWells
    global outerWells
    global lastWell
    global trialtype
    global locktype1
    global locktype2
    global locktype3
    global waslock

    locktype = int(val[1])
    trialtype = 4
    print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
    print("SCQTMESSAGE: trigger(6);\n")  # start lockout timer in SCQTMESSAGE
    #turn off all lights
    print("SCQTMESSAGE: dio = "+str(homeWell)+";\n") # turn off home well
    print("SCQTMESSAGE: trigger(4);\n")
    for num in range(len(waitWells)):   #turn off all wait and outer well lights 
        print("SCQTMESSAGE: dio = "+str(waitWells[num])+";\n")
        print("SCQTMESSAGE: trigger(4);\n")
    for num in range(len(outerWells)):
        print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
        print("SCQTMESSAGE: trigger(4);\n")
    waslock = 1
    print("SCQTMESSAGE: waslock = "+str(waslock)+";\n") # turn off home well
    if locktype == 1:
        locktype1 += 1 # type 1 = wrong rip/wait well
    if locktype == 2:
        locktype2+=1 # type 2 = order error
    if locktype == 3:
        locktype3+=1  # type3 = didn't wait long enough


def lockend():
    global trialtype
    global homeWell
    global waitWells
    global outerWells

    # make sure all other wells are off
    for num in range(len(waitWells)):   #turn off all wait and outer well lights 
        print("SCQTMESSAGE: dio = "+str(waitWells[num])+";\n")
        print("SCQTMESSAGE: trigger(4);\n")
    for num in range(len(outerWells)):
        print("SCQTMESSAGE: dio = "+str(outerWells[num])+";\n")
        print("SCQTMESSAGE: trigger(4);\n")
    trialtype = 0
    print("SCQTMESSAGE: trialtype = "+str(trialtype)+";\n")
    print("SCQTMESSAGE: dio = "+str(homeWell)+";\n") # turn on home well
    print("SCQTMESSAGE: trigger(3);\n")


def updateWaslock(val):
    global waslock

    waslock = int(val[1])
    print("SCQTMESSAGE: waslock = "+str(waslock)+";\n")

# Function: generate click sound
def generate_click():

    File='ZippoClick_short.wav'
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

# Function: generate cowbell sound
def generate_beep():

    File='Beep.wav'
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

def printStats():
    global homeCount
    global ripCount
    global waitCount
    global locktype1
    global locktype2
    global locktype3
    global thisGoal
    global allGoal
    global otherCount

    print("SCQTMESSAGE: disp('homeCount = "+str(homeCount)+"');\n") 
    print("SCQTMESSAGE: disp('waitCount = "+str(waitCount)+"');\n") 
    print("SCQTMESSAGE: disp('ripCount = "+str(ripCount)+"');\n") 
    print("SCQTMESSAGE: disp('locktype1 = "+str(locktype1)+"');\n") 
    print("SCQTMESSAGE: disp('locktype2 = "+str(locktype2)+"');\n") 
    print("SCQTMESSAGE: disp('locktype3 = "+str(locktype3)+"');\n") 
    print("SCQTMESSAGE: disp('goalCount = "+str(thisGoal)+"');\n")
    print("SCQTMESSAGE: disp('goalTotal = "+str(allGoal)+"');\n") 
    print("SCQTMESSAGE: disp('otherCount = "+str(otherCount)+"');\n")  


# This is the custom callback function. When events occur, addScQtEvent will
# call this function. This function MUST BE NAMED 'callback'!!!!
def callback(line):
    global waslock 

    if line.find("UP") >= 0: #input triggered
        pokeIn(re.findall(r'\d+',line))
    if line.find("DOWN") >= 0: #input triggered
        pokeOut(re.findall(r'\d+',line))
    if line.find("trigger function 22") >= 0:  # ripple has been detected
        print("SCQTMESSAGE: trigger(7);\n" )
    if line.find("riptime") >=0:  # add ripwait to holding vector
       addtime(re.findall(r'\d+',line))
    if line.find("BEEP 1") >= 0: # make a beep and deliver reward
        beep()
    if line.find("BEEP 2") >= 0: # make a beep and deliver reward
       generate_beep()
    if line.find("CLICK 1") >= 0: # make a click and deliver reward
        click()
    if line.find("CLICK 2") >= 0: # make a click and deliver reward
        generate_click()
    if line.find("LOCKOUT") >= 0: # lockout procedure
        lockout(re.findall(r'\d+',line))
    if line.find("LOCKEND") >= 0: # reset trialtype to 0
        lockend()
    if line.find("WHITENOISE") >= 0: # make noise during lockout
        makewhitenoise()
    if line.find("waitdelay") >= 0: # reset trialtype to 0
        decreasewaitdist(re.findall(r'\d+',line))
    if line.find("waslock") >= 0:  #update waslock value
        updateWaslock(re.findall(r'\d+',line))


# define wells
homeWell = 1
waitWells = [28, 29]
outerWells = [9, 10, 11, 12, 13, 14, 15, 16]
# define pumps
homePump = 25
waitPumps = [26, 27]
outerPumps = [17, 18, 19, 20, 21, 22, 23, 24]

#global variables 
lastWell = -1
currWell = -1
trialtype = 0
outerReps = np.random.randint(4,8)
#outerReps = 15
print("SCQTMESSAGE: disp('outerreps = "+str(outerReps)+"');\n") 
allGoal = 0
thisGoal = 0
oldGoals = []   #initialize empty
numgoals = 1  # unnecessary because innately coded elsewhere (set to 1 after first encounter of goal)
forageNum = 1
goalWell = np.random.choice(outerWells,forageNum,replace=False)   #initialize to 4 wells

waitdist = [14067, 20082, 4058, 3928, 4243, 12039, 8122, 23012]
startwaitdist = [2500,2500,3000]
count = 0 
locksoundlength = 1000
print(goalWell)
waslock = 0

# counter variables; moved over from SC
homeCount=0
waitCount = 0
ripCount = 0
locktype1 = 0
locktype2 = 0
locktype3 = 0
otherCount = 0 



