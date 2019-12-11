% PROGRAM NAME: 	Load milk
% AUTHOR: 		AKG 
% DESCRIPTION:	

int deliverPeriod= 200   	% how long to deliver the reward
int rewardWell=0;

portout[1]=0
portout[2]=0

portout[8]=0
portout[9]=0
portout[10]=0
portout[11]=0
portout[12]=0
portout[13]=0
portout[14]=0
portout[15]=0;




function 1
	portout[rewardWell]=1 % reward
	do in deliverPeriod 
		portout[rewardWell]=0 % reset reward
	end	
end;

callback portin[1] up   % home	
	rewardWell = 25
	trigger(1)
end;
					

callback portin[2] up   % center	
	rewardWell = 26
	trigger(1)
end;

callback portin[8] up   % arm1	
	rewardWell = 17
	trigger(1)
end;

callback portin[9] up   % arm2	
	rewardWell = 18
	trigger(1)
end;

callback portin[10] up   % arm3	
	rewardWell = 19
	trigger(1)
end;

callback portin[11] up   % arm4	
	rewardWell = 20
	trigger(1)
end;

callback portin[12] up   % arm5
	rewardWell = 21
	trigger(1)
end;

callback portin[13] up   % arm6	
	rewardWell = 22
	trigger(1)
end;

callback portin[14] up   % arm7
	rewardWell = 23
	trigger(1)
end;

callback portin[15] up   % arm8
	rewardWell = 24
	trigger(1)
end;
