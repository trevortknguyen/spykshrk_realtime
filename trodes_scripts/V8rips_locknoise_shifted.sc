% PROGRAM NAME: 	V8pre_goal_nowait	

% constants
int deliverPeriodBox= 150   	% how long to deliver the reward at home/rip/wait
int deliverPeriodOuter= 450   	% how long to deliver the reward at outer wells
int lockoutPeriod= 30000 	% length of lockout, 20 sec
int proxTime = 250  		% amount of time allowed to be away from nose poke (originally 200)
int ripwin = 1000		% ms after which ripstate is off

% variables
int rewardWell = 0
int currWell = 0
int lastWell = 0
int dio = 0
int waittime = 0
int proximity = 0
int trialtype = -1
int ripwait = 0			% state of waiting for rip
int riptime = 0			% track lenth of time intil rip 
int ripstart = 0		% starttime of ripwait
int ripstate = 0                % state of having detected a rip
int waslock = 0

portout[1] = 1
portout[28] = 0
portout[29] = 0

portout[9] = 0
portout[10] = 0
portout[11] = 0
portout[12] = 0
portout[13] = 0
portout[14] = 0
portout[15] = 0
portout[16] = 0
;

% function to deliver reward to box wells
function 1
	portout[rewardWell]=1 % reward
	do in deliverPeriodBox 
		portout[rewardWell]=0 % reset reward
	end	
end;

% function to deliver reward to outer wells
function 2
	disp('outer reward')	
	portout[rewardWell]=1 % reward
	do in deliverPeriodOuter 
		portout[rewardWell]=0 % reset reward
	end	
end;

function 3  % Function to turn on output
	portout[dio]=1
end;

function 4  % function to turn off output	
	portout[dio]=0	
end;

function 6 % end lockout and reactivate home
	disp('WHITENOISE')	
	do in lockoutPeriod
		disp('LOCKEND')
	end
end;

% function to make click if rip detected
function 7
	if ripwait == 1 do
		disp('BEEP 1')
		disp('BEEP 2')
		riptime = clock() - ripstart
		disp(riptime)
		ripwait = 0
		trialtype = 3	
		proximity = 0 	
	else do
		ripstate = 1
		do in ripwin
			ripstate = 0
		end
	end	
end;

% CALLBACKS -- EVENT-DRIVEN TRIGGERS
callback portin[1] up
	if trialtype != 4 do 
		currWell = 1
		disp('UP 1')
		waslock = 0
		disp(waslock)
	end
end;

callback portin[1] down
	if trialtype != 4 do
		lastWell = 1
		disp('DOWN 1')
	end
end;

callback portin[28] up % Rip well
	currWell = 28 % well currently active
	disp('UP 28')
	if (trialtype == 1 && lastWell==1) do
		proximity = 1
		disp('start rip wait')
		ripstart = clock()
		ripwait = 1
	end 
	if (trialtype==1 && lastWell==28 && proximity >0) do
		proximity=proximity+1
	end			
	if (trialtype != 4 && trialtype != 1 && currWell != lastWell && waslock != 1) do 
		if trialtype == 2 do
			disp('LOCKOUT 1')
		else do
			disp('LOCKOUT 2')
		end

	end
end;

callback portin[28] down
	lastWell=28 % well left, now last well
	disp('DOWN 28')
	if (trialtype == 0 && proximity < 1) do  %criteria satisfied so turn light off
		portout[28] = 0
	end
	if proximity>0 do
		do in proxTime	
			proximity=proximity-1	
			if (proximity <1 && trialtype <3) do
				disp('LOCKOUT 3')
				ripwait = 0
			end	
		end
	end
end;

callback portin[29] up % wait well
	currWell = 29 % well currently active
	disp('UP 29')
	if trialtype == 2 do
		if lastWell != currWell do
			proximity = 1
			do in waittime
				if (proximity > 0 && ripstate == 0 && trialtype == 2) do
					proximity = 0
					trialtype = 3
					disp('CLICK 1')
					disp('CLICK 2')
				else do in ripwin
					if (proximity > 0 && ripstate == 0 && trialtype == 2) do
						proximity = 0
						trialtype = 3
						disp('CLICK 1')
						disp('CLICK 2')
						disp('waitdelay 1')
					else do in ripwin  % 2 delays max, regardless of ripstate
						if (proximity > 0 && trialtype == 2) do    
							proximity = 0
							trialtype = 3
							disp('CLICK 1')
							disp('CLICK 2')
							disp('waitdelay 2')
						end
					end	
				end
			end
		else do 
			proximity=proximity+1			
		end
	else do
		if (trialtype != 4 && currWell != lastWell && waslock != 1) do
			if trialtype == 1 do
				disp('LOCKOUT 1')
			else do
				disp('LOCKOUT 2')
			end			
		end
	end
end;

callback portin[29] down
	lastWell=29 % well left, now last well
	disp('DOWN 29')
	if proximity>0 do
		do in proxTime	
			proximity=proximity-1	
			if (proximity <1 && trialtype <3) do
				disp('LOCKOUT 3')
			end	
		end
	end
end;



callback portin[9] up
	currWell = 9
	if currWell != lastWell do
		disp('UP 9')
	end
end;

callback portin[9] down
	lastWell = 9
	disp('DOWN 9')
end;

callback portin[10] up
	currWell = 10
	if currWell != lastWell do
		disp('UP 10')
	end
end;

callback portin[10] down
	lastWell = 10
	disp('DOWN 10')
end;

callback portin[11] up
	currWell = 11
	if currWell != lastWell do
		disp('UP 11')
	end
end;

callback portin[11] down
	lastWell = 11
	disp('DOWN 11')
end;

callback portin[12] up
	currWell = 12
	if currWell != lastWell do
		disp('UP 12')
	end
end;

callback portin[12] down
	lastWell = 12

	disp('DOWN 12')
end;

callback portin[13] up
	currWell = 13
	if currWell != lastWell do
		disp('UP 13')
	end
end;

callback portin[13] down
	lastWell = 13
	disp('DOWN 13')
end;

callback portin[14] up
	currWell = 14
	if currWell != lastWell do
		disp('UP 14')
	end
end;

callback portin[14] down
	lastWell = 14
	disp('DOWN 14')
end;

callback portin[15] up
	currWell = 15
	if currWell != lastWell do
		disp('UP 15')
	end
end;

callback portin[15] down
	lastWell = 15
	disp('DOWN 15')
end;

callback portin[16] up
	currWell = 16
	if currWell != lastWell do
		disp('UP 16')
	end
end;

callback portin[16] down
	lastWell = 16
	disp('DOWN 16')
end;
