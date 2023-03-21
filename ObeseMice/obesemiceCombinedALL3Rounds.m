% EFatMice Round1 and 2 combined

%% LOADING files - both from tracking video and from photometry (Written by S?ren)
%Round1
% Trial 1: %Fiber 1: Mouse 250.3 NAc dlight - DRUG b2   Fiber 3:Mouse 250.4 NAc dlight - DRUG c3
% Trial 2: %Fiber 1: Mouse 250.5 NAc dlight - DRUG a1   Fiber 3:Mouse 250.7 NAc dlight - DRUG a1
% Trial 3: %Fiber 1: Mouse 253.4 NAc dlight - DRUG c3   Fiber 3:Mouse 253.6 NAc dlight - DRUG b2
% Trial 4: %Fiber 1: Mouse 254.2 NAc dlight - DRUG b2   Fiber 3:Mouse 254.5 NAc dlight - DRUG a1
% Trial 5: %Fiber 1: Mouse 254.8 NAc dlight - DRUG c3
%Round2
% Trial 6: %Fiber 1: Mouse 250.3 NAc - DRUG c4  Fiber 3:Mouse 250.4 NAc - DRUG b4
% Trial 7: %Fiber 1: Mouse 250.5 NAc - DRUG a4  Fiber 3:Mouse 250.7 NAc - DRUG a5
% Trial 8: %Fiber 1: Mouse 253.4 NAc - DRUG b5  Fiber 3:Mouse 253.6 NAc - DRUG c5
% Trial 9: %Fiber 1: Mouse 254.2 NAc - DRUG c6  Fiber 3:Mouse 254.5 NAc - DRUG b6
% Trial 10: %Fiber 1: Mouse 254.8 NAc - DRUG a6

%Round 3  OBS NEW DRUG CODES - BUT HERE THEY ARE CHANGED TO THE CORRESPONDING
%OLD DRUG CODES
% Trial 11 %Fiber 1: Mouse 250.5 NAc - DRUG b7  Fiber 3:Mouse 250.4 NAc - DRUG a7
% Trial 12: %Fiber 1: Mouse 254.2 NAc - DRUG c7  Fiber 3:Mouse 254.5 NAc - DRUG c8
% Trial 13: %Fiber 1: Mouse 253.4 NAc - DRUG a7


drugA=[3 4 8 12 13 18];
drugB=[1 6 7 11 14 17] ;
drugC=[2 5 9 10 15 16];

NoOfRounds=13;
trackingfile = cell(1,NoOfRounds);
%Round1
trackingfile{1} = "BaslerTrackVidDoc2020-07-15T11_29_00.csv"; %Fiber 1: Mouse 5224  Fiber 3:Mouse 5222
trackingfile{2} = "BaslerTrackVidDoc2020-07-15T15_01_23.csv"; %Fiber 1: Mouse 5226 Fiber 3:Mouse 5228
trackingfile{3} = "BaslerTrackVidDoc2020-07-16T08_49_15.csv"; %Fiber 1: Mouse 5218 Fiber 3:Mouse 5212
trackingfile{4} = "BaslerTrackVidDoc2020-07-16T12_01_59.csv"; %Fiber 1: Mouse 5216 Fiber 3:Mouse 5214
trackingfile{5} = "BaslerTrackVidDoc2020-07-16T15_24_57.csv"; %Fiber 1: Mouse 5216 Fiber 3:Mouse 5214
%Round2
trackingfile{6} = "BaslerTrackVidDoc2020-08-18T14_31_04.csv"; %
trackingfile{7} = "BaslerTrackVidDoc2020-08-17T09_29_41.csv"; %
trackingfile{8} = "BaslerTrackVidDoc2020-08-17T12_32_26.csv"; %
trackingfile{9} = "BaslerTrackVidDoc2020-08-17T15_32_05.csv"; %
trackingfile{10} = "BaslerTrackVidDoc2020-08-18T09_12_49.csv"; %
%Round 3
trackingfile{11} = "BaslerTrackVidDoc2020-09-15T11_09_31.csv"; %
trackingfile{12} = "BaslerTrackVidDoc2020-09-16T11_22_45.csv"; %
trackingfile{13} = "BaslerTrackVidDoc2020-09-16T14_38_44.csv"; %


TrackingData = cell(1,NoOfRounds);
%Round1
TrackingData{1} = load_trackingfile(trackingfile{1});
TrackingData{2} = load_trackingfile(trackingfile{2}); 
TrackingData{3} = load_trackingfile(trackingfile{3}); 
TrackingData{4} = load_trackingfile(trackingfile{4}); 
TrackingData{5} = load_trackingfile(trackingfile{5}); 
%Round2
TrackingData{6} = load_trackingfile(trackingfile{6});
 TrackingData{7} = load_trackingfile(trackingfile{7}); 
TrackingData{8} = load_trackingfile(trackingfile{8}); 
TrackingData{9} = load_trackingfile(trackingfile{9}); 
TrackingData{10} = load_trackingfile(trackingfile{10}); 
%Round3
TrackingData{11} = load_trackingfile(trackingfile{11});
TrackingData{12} = load_trackingfile(trackingfile{12}); 
TrackingData{13} = load_trackingfile(trackingfile{13}); 


file_name = cell(1,NoOfRounds);
%Round1
file_name{1} = "aFlyCap2020-07-15T11_29_00.csv"; 
file_name{2} = "aFlyCap2020-07-15T15_01_23.csv"; 
file_name{3} = "aFlyCap2020-07-16T08_49_15.csv"; 
file_name{4} = "aFlyCap2020-07-16T12_01_59.csv"; 
file_name{5} = "aFlyCap2020-07-16T15_24_57.csv"; 
%Round2
file_name{6} = "aFlyCap2020-08-18T14_31_04.csv"; 
file_name{7} ="aFlyCap2020-08-17T09_29_41.csv";  
file_name{8} = "aFlyCap2020-08-17T12_32_26.csv"; 
file_name{9} = "aFlyCap2020-08-17T15_32_05.csv"; 
file_name{10} = "aFlyCap2020-08-18T09_12_49.csv";
%ROund3
file_name{11} = "aFlyCap2020-09-15T11_09_31.csv"; 
file_name{12} = "aFlyCap2020-09-16T11_22_45.csv";  
file_name{13} = "aFlyCap2020-09-16T14_38_44.csv"; 


sig_data = cell(1,NoOfRounds);
ref_data = cell(1,NoOfRounds);
%Round1
[ref_data{1}, sig_data{1}] = load_File(file_name{1},TrackingData{1});
[ref_data{2}, sig_data{2}] = load_File(file_name{2},TrackingData{2}); 
[ref_data{3}, sig_data{3}] = load_File(file_name{3},TrackingData{3}); 
[ref_data{4}, sig_data{4}] = load_File(file_name{4},TrackingData{4}); 
[ref_data{5}, sig_data{5}] = load_File(file_name{5},TrackingData{5}); 
%Round2
[ref_data{6}, sig_data{6}] = load_File(file_name{6},TrackingData{6});
 [ref_data{7}, sig_data{7}] = load_File(file_name{7},TrackingData{7}); 
[ref_data{8}, sig_data{8}] = load_File(file_name{8},TrackingData{8}); 
[ref_data{9}, sig_data{9}] = load_File(file_name{9},TrackingData{9}); 
[ref_data{10}, sig_data{10}] = load_File(file_name{10},TrackingData{10}); 
% 
[ref_data{11}, sig_data{11}] = load_File(file_name{11},TrackingData{11});
[ref_data{12}, sig_data{12}] = load_File(file_name{12},TrackingData{12}); 
[ref_data{13}, sig_data{13}] = load_File(file_name{13},TrackingData{13}); 



% %% Straight plot
% figure()
% hold on
%  plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber1)
%  plot(ref_data{1}.SecFromZero,ref_data{1}.Fiber1)
% % plot(sig_data{1}.SecFromZero,ref_data{1}.Fiber3)
% %plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber3)
% % plot(ref_data{2}.SecFromZero,ref_data{2}.Fiber1)
% plot(TrackingData{1}.SecFromZero, TrackingData{1}.Dist1)
% % plot(TrackingData{1}.SecFromZero, TrackingData{1}.Dist2)
% %xlim([70 9350])
% hold off
% 
% figure()
% hold on
% % plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber1)
% % % plot(sig_data{1}.SecFromZero,ref_data{1}.Fiber1)
% plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber3)
% plot(sig_data{1}.SecFromZero,ref_data{1}.Fiber3)
% % plot(sig_data{2}.SecFromZero,sig_data{2}.Fiber1)
% % plot(ref_data{2}.SecFromZero,ref_data{2}.Fiber1)
%  %plot(TrackingData{1}.SecFromZero, TrackingData{1}.Dist1)
% plot(TrackingData{1}.SecFromZero, TrackingData{1}.Dist2)
% %xlim([70 9350])
% hold off
% 
% for i=1:NoOfRounds
% figure(i)
% hold on
% plot(sig_data{i}.SecFromZero,sig_data{i}.Fiber1, 'green')
% plot(sig_data{i}.SecFromZero,ref_data{i}.Fiber1, 'blue')
% % % plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber3)
% % % plot(sig_data{1}.SecFromZero,ref_data{1}.Fiber3)
% % plot(sig_data{2}.SecFromZero,sig_data{2}.Fiber1)
% % plot(ref_data{2}.SecFromZero,ref_data{2}.Fiber1)
%  plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist1)
% %plot(TrackingData{1}.SecFromZero, TrackingData{1}.Dist2)
% xlim([70 9350])
% hold off
% end
% 
% for i=1:NoOfRounds
% figure(i+100)
% hold on
% % plot(sig_data{i}.SecFromZero,sig_data{i}.Fiber1, 'green')
% % plot(sig_data{i}.SecFromZero,ref_data{i}.Fiber1, 'blue')
% plot(sig_data{i}.SecFromZero,sig_data{i}.Fiber3)
% plot(sig_data{i}.SecFromZero,ref_data{i}.Fiber3)
% % plot(sig_data{2}.SecFromZero,sig_data{2}.Fiber1)
% % plot(ref_data{2}.SecFromZero,ref_data{2}.Fiber1)
% %  plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist1)
% plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist2)
% xlim([70 9350])
% hold off
% end
%% Arena1 tracking timepoints
time = cell(1,NoOfRounds); %Don't know if this is used

FPMCam=40*60;%frames/sec*sec/min  Frames pr minute for Basler Tracking cam - depends on what it is set to

% Define interval wherein trial is expected to start:
searchStartArena1=FPMCam*([1, 4]);
% Define interval wherein first injection is expected
search2Arena1=FPMCam*([62, 68]);
% Define interval wherein second injection is expected
search3Arena1=FPMCam*([92, 102]);
% Define timepoint where experiment is expected to end
searchEndArena1=FPMCam*153;

%Search for StartIndex
for i = 1:NoOfRounds 
    for z = searchStartArena1(1):searchStartArena1(2)
        if mean(TrackingData{i}.Dist1(z-100:z))==0
           StartIndex_Arena1(i)=z;
        end
    end
end

%Search for StartIndex
for i = 1:NoOfRounds 
    for z = search2Arena1(1):search2Arena1(2)
        if mean(TrackingData{i}.Dist1(z-100:z))==0
           Inject_1_Index_Arena1(i)=z;
        end
    end
end

for i = 1:NoOfRounds 
    for z = search3Arena1(1):search3Arena1(2)
        if mean(TrackingData{i}.Dist1(z-50:z))==0
           Inject_2_Index_Arena1(i)=z;
        end
    end
end

%%% Search for very last tracking point and use that for finding actual
%%% last point where mouse was in the arena - only necessary for arena1
%%% because it detects when I remove mouse 2.
for i = 1:NoOfRounds 
    for z = searchEndArena1:length(TrackingData{i}.Dist1)
        if mean(TrackingData{i}.Dist1(z-10:z))>0
           EndIndex_Arena1(i)=z-10;
        end
    end
    ActualEndIndex_Arena1(i)=find(TrackingData{i}.Dist1(searchEndArena1:EndIndex_Arena1(i))==0, 1, 'first')+searchEndArena1;  %DO THIS
end
%% Arena1 sig timepoints
for i = 1:NoOfRounds
t0_synch =sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero( StartIndex_Arena1(i));
 sig_t0_Arena1(i)  =find(t0_synch==min(t0_synch(t0_synch>0)));
%  
   tInj1_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(Inject_1_Index_Arena1(i));
   sig_tInj1_Arena1(i)=find(tInj1_synch==min(tInj1_synch(tInj1_synch>0)));
% %   
   tInj2_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(Inject_2_Index_Arena1(i));    
   sig_tInj2_Arena1(i)=find(tInj2_synch==min(tInj2_synch(tInj2_synch>0)));
%     
 tEnd_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(ActualEndIndex_Arena1(i));
 sig_tEnd_Arena1(i)=find(tEnd_synch==min(tEnd_synch(tEnd_synch>0)));  
end
%% Plotting Arena1

% for i=1:NoOfRounds
% figure(i+100)
% plot(sig_data{i}.SecFromZero(sig_t0_Arena1(i):sig_tEnd_Arena1(i)),sig_data{i}.Fiber1(sig_t0_Arena1(i):sig_tEnd_Arena1(i)),'green')
% hold on
% xline(sig_data{i}.SecFromZero(sig_t0_Arena1(i)))
% xline(sig_data{i}.SecFromZero(sig_tInj1_Arena1(i)))
% xline(sig_data{i}.SecFromZero(sig_tInj2_Arena1(i)))
% xline(sig_data{i}.SecFromZero(sig_tEnd_Arena1(i)))
% plot(ref_data{i}.SecFromZero(sig_t0_Arena1(i):sig_tEnd_Arena1(i)),ref_data{i}.Fiber1(sig_t0_Arena1(i):sig_tEnd_Arena1(i)),'blue')
% % plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist1)
% %plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist2)
%  hold off
% end
%% Straight plot Arena2
% 
% for i=[1:4 6:9]
% figure()
% hold on
% % plot(sig_data{1}.SecFromZero,sig_data{1}.Fiber1)
% plot(sig_data{i}.SecFromZero,ref_data{i}.Fiber1)
% plot(sig_data{i}.SecFromZero,sig_data{i}.Fiber3)
% % % plot(sig_data{1}.SecFromZero,ref_data{1}.Fiber3)
% % plot(sig_data{2}.SecFromZero,sig_data{2}.Fiber1)
% % plot(ref_data{2}.SecFromZero,ref_data{2}.Fiber1)
% % plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist1)
% %plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist2)
% %xlim([70 9350])
% hold off
% end
%% Arena2 tracking timepoints
Arena2delay=2; %in minutes

% Define interval wherein trial is expected to start:
searchStartArena2=FPMCam*(([1, 4])+Arena2delay);
% Define interval wherein first injection is expected
search2Arena2=FPMCam*(([62, 68])+Arena2delay);
% Define interval wherein second injection is expected
search3Arena2=FPMCam*(([92, 98])+Arena2delay);
% Define timepoint where experiment is expected to end
searchEndArena2=FPMCam*(153+Arena2delay);

%Search for StartIndex
for i = [1:4 6:9 11:12]
    for z = searchStartArena2(1):searchStartArena2(2)
        if mean(TrackingData{i}.Dist2(z-100:z))==0
           StartIndex_Arena2(i)=z;
        end
    end
end

%Search for StartIndex
for i = [1:4 6:9 11:12] 
    for z = search2Arena2(1):search2Arena2(2)
        if mean(TrackingData{i}.Dist2(z-100:z))==0
           Inject_1_Index_Arena2(i)=z;
        end
    end
end

for i = [1:4 6:9 11:12]
    for z = search3Arena2(1):search3Arena2(2)
            if mean(TrackingData{i}.Dist2(z-50:z))==0
               Inject_2_Index_Arena2(i)=z;
        end
    end
end

%%% Search for very last tracking point and use that for finding actual
%%% last point where mouse was in the arena
for i = [1:4 6:9 11:12]
    for z = searchEndArena2:length(TrackingData{i}.Dist2)
        if mean(TrackingData{i}.Dist2(z-10:z))>0
           EndIndex_Arena2(i)=z-10;
        end
    end
end
%% Arena2 sig timepoints
for i = [1:4 6:9 11:12]
t0_synch =sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(StartIndex_Arena2(i));
 sig_t0_Arena2(i)  =find(t0_synch==min(t0_synch(t0_synch>0)));
 
   tInj1_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(Inject_1_Index_Arena2(i));
  sig_tInj1_Arena2(i)=find(tInj1_synch==min(tInj1_synch(tInj1_synch>0)));
%   
   tInj2_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(Inject_2_Index_Arena2(i));    
   sig_tInj2_Arena2(i)=find(tInj2_synch==min(tInj2_synch(tInj2_synch>0)));
    
 tEnd_synch=sig_data{i}.SecFromZero-TrackingData{i}.SecFromZero(EndIndex_Arena2(i));
 sig_tEnd_Arena2(i)=find(tEnd_synch==min(tEnd_synch(tEnd_synch>0)));  
end
%% Plotting Arena2
% 
% for i=[1:4 6:9]
% figure(i+200)
% plot(sig_data{i}.SecFromZero(sig_t0_Arena2(i):sig_tEnd_Arena2(i)),sig_data{i}.Fiber3(sig_t0_Arena2(i):sig_tEnd_Arena2(i)),'green')
% hold on
% xline(sig_data{i}.SecFromZero(sig_t0_Arena2(i)))
% xline(sig_data{i}.SecFromZero(sig_tInj1_Arena2(i)))
% xline(sig_data{i}.SecFromZero(sig_tInj2_Arena2(i)))
% xline(sig_data{i}.SecFromZero(sig_tEnd_Arena2(i)))
% plot(ref_data{i}.SecFromZero(sig_t0_Arena2(i):sig_tEnd_Arena2(i)),ref_data{i}.Fiber3(sig_t0_Arena2(i):sig_tEnd_Arena2(i)),'blue')
% %  plot(TrackingData{i}.SecFromZero, TrackingData{i}.Dist2)
% hold off
% end
%% remove error in signal for mouse 253.6 - Round1
%Find datapoints of error in signal and call them removStart and removEnd
figure()
plot(sig_data{3}.Fiber3)
figure()
plot(ref_data{3}.Fiber3)
%

removStart.DataIndex=32756
removEnd.DataIndex=36322

%plot segment with signal with error
figure()
plot(sig_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex))

%data from removStart to removEnd is set to mean of
%removStart and removEnd !!! NOT NaN as detrending doesnt work with NaNs

sig_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex)= (sig_data{3}.Fiber3(removStart.DataIndex)+sig_data{3}.Fiber3(removEnd.DataIndex))/2  
ref_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex)= (ref_data{3}.Fiber3(removStart.DataIndex)+ref_data{3}.Fiber3(removEnd.DataIndex))/2

%plot segment with signal with error again - flat
figure()
plot(sig_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{3}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
%% remove error in signal for mouse 253.6 - ROund 2

figure()
plot(sig_data{8}.Fiber1)
figure()
plot(ref_data{8}.Fiber1)

figure()
plot(sig_data{8}.Fiber3)
figure()
plot(ref_data{8}.Fiber3)
%
%Find datapoints of error in signal and call them removStart and removEnd
% Check what the data indices are - and plot in after the equal sign
removStart.DataIndex=18751
removEnd.DataIndex=22427

%plot segment with signal with error
figure()
plot(sig_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex))

figure()
plot(sig_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex))

%data from removStart to removEnd is set to mean of
%removStart and removEnd !!! NOT NaN as detrending doesnt work with NaNs
sig_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex)= (sig_data{8}.Fiber1(removStart.DataIndex)+sig_data{8}.Fiber1(removEnd.DataIndex))/2  
ref_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex)= (ref_data{8}.Fiber1(removStart.DataIndex)+ref_data{8}.Fiber1(removEnd.DataIndex))/2
% 
% sig_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex)= (sig_data{8}.Fiber3(removStart.DataIndex)+sig_data{8}.Fiber3(removEnd.DataIndex))/2  
% ref_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex)= (ref_data{8}.Fiber3(removStart.DataIndex)+ref_data{8}.Fiber3(removEnd.DataIndex))/2

%plot segment with signal with error again - flat
figure()
plot(sig_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{8}.Fiber1(removStart.DataIndex:removEnd.DataIndex))

figure()
plot(sig_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
hold on
plot(ref_data{8}.Fiber3(removStart.DataIndex:removEnd.DataIndex))
%% Make common list
% Trial 1: %Fiber 1: Mouse 250.3 NAc dlight - DRUG b1   Fiber 3:Mouse 250.4 NAc dlight - DRUG c1
% Trial 2: %Fiber 1: Mouse 250.5 NAc dlight - DRUG a1   Fiber 3:Mouse 250.7 NAc dlight - DRUG a2
% Trial 3: %Fiber 1: Mouse 253.4 NAc dlight - DRUG c2   Fiber 3:Mouse 253.6 NAc dlight - DRUG b2
% Trial 4: %Fiber 1: Mouse 254.2 NAc dlight - DRUG b3   Fiber 3:Mouse 254.5 NAc dlight - DRUG a3
% Trial 5: %Fiber 1: Mouse 254.8 NAc dlight - DRUG c3
%Round2
% Trial 6: %Fiber 1: Mouse 250.3 NAc - DRUG c4  Fiber 3:Mouse 250.4 NAc - DRUG b4
% Trial 7: %Fiber 1: Mouse 250.5 NAc - DRUG a4  Fiber 3:Mouse 250.7 NAc - DRUG a5
% Trial 8: %Fiber 1: Mouse 253.4 NAc - DRUG b5  Fiber 3:Mouse 253.6 NAc - DRUG c5
% Trial 9: %Fiber 1: Mouse 254.2 NAc - DRUG c6  Fiber 3:Mouse 254.5 NAc - DRUG b6
% Trial 10: %Fiber 1: Mouse 254.8 NAc - DRUG a6
%Round 3  OBS NEW DRUG CODES - BUT HERE THEY ARE CHANGED TO THE CORRESPONDING
%OLD DRUG CODES
% Trial 11: %Fiber 1: Mouse 250.5 NAc - DRUG b7  Fiber 3:Mouse 250.4 NAc - DRUG a7
% Trial 12: %Fiber 1: Mouse 254.2 NAc - DRUG c7  Fiber 3:Mouse 254.5 NAc - DRUG c8
% Trial 13: %Fiber 1: Mouse 253.4 NAc - DRUG a8


drugA=[3 4 8 12 13 18 20 23];
drugB=[1 6 7 11 14 17 19] ;
drugC=[2 5 9 10 15 16 21 22];

sig_t0(1)=sig_t0_Arena1(1);
sig_t0(2)=sig_t0_Arena2(1);
sig_t0(3)=sig_t0_Arena1(2);
sig_t0(4)=sig_t0_Arena2(2);
sig_t0(5)=sig_t0_Arena1(3);
sig_t0(6)=sig_t0_Arena2(3);
sig_t0(7)=sig_t0_Arena1(4);
sig_t0(8)=sig_t0_Arena2(4);
sig_t0(9)=sig_t0_Arena1(5);
sig_t0(10)=sig_t0_Arena1(6);
sig_t0(11)=sig_t0_Arena2(6);
sig_t0(12)=sig_t0_Arena1(7);
sig_t0(13)=sig_t0_Arena2(7);
sig_t0(14)=sig_t0_Arena1(8);
sig_t0(15)=sig_t0_Arena2(8);
sig_t0(16)=sig_t0_Arena1(9);
sig_t0(17)=sig_t0_Arena2(9);
sig_t0(18)=sig_t0_Arena1(10);
sig_t0(19)=sig_t0_Arena1(11);
sig_t0(20)=sig_t0_Arena2(11);
sig_t0(21)=sig_t0_Arena1(12);
sig_t0(22)=sig_t0_Arena2(12);
sig_t0(23)=sig_t0_Arena1(13);

sig_tInj1(1)=sig_tInj1_Arena1(1);
sig_tInj1(2)=sig_tInj1_Arena2(1);
sig_tInj1(3)=sig_tInj1_Arena1(2);
sig_tInj1(4)=sig_tInj1_Arena2(2);
sig_tInj1(5)=sig_tInj1_Arena1(3);
sig_tInj1(6)=sig_tInj1_Arena2(3);
sig_tInj1(7)=sig_tInj1_Arena1(4);
sig_tInj1(8)=sig_tInj1_Arena2(4);
sig_tInj1(9)=sig_tInj1_Arena1(5);
sig_tInj1(10)=sig_tInj1_Arena1(6);
sig_tInj1(11)=sig_tInj1_Arena2(6);
sig_tInj1(12)=sig_tInj1_Arena1(7);
sig_tInj1(13)=sig_tInj1_Arena2(7);
sig_tInj1(14)=sig_tInj1_Arena1(8);
sig_tInj1(15)=sig_tInj1_Arena2(8);
sig_tInj1(16)=sig_tInj1_Arena1(9);
sig_tInj1(17)=sig_tInj1_Arena2(9);
sig_tInj1(18)=sig_tInj1_Arena1(10);
sig_tInj1(19)=sig_tInj1_Arena1(11);
sig_tInj1(20)=sig_tInj1_Arena2(11);
sig_tInj1(21)=sig_tInj1_Arena1(12);
sig_tInj1(22)=sig_tInj1_Arena2(12);
sig_tInj1(23)=sig_tInj1_Arena1(13);


sig_tInj2(1)=sig_tInj2_Arena1(1);
sig_tInj2(2)=sig_tInj2_Arena2(1);
sig_tInj2(3)=sig_tInj2_Arena1(2);
sig_tInj2(4)=sig_tInj2_Arena2(2);
sig_tInj2(5)=sig_tInj2_Arena1(3);
sig_tInj2(6)=sig_tInj2_Arena2(3);
sig_tInj2(7)=sig_tInj2_Arena1(4);
sig_tInj2(8)=sig_tInj2_Arena2(4);
sig_tInj2(9)=sig_tInj2_Arena1(5);
sig_tInj2(10)=sig_tInj2_Arena1(6);
sig_tInj2(11)=sig_tInj2_Arena2(6);
sig_tInj2(12)=sig_tInj2_Arena1(7);
sig_tInj2(13)=sig_tInj2_Arena2(7);
sig_tInj2(14)=sig_tInj2_Arena1(8);
sig_tInj2(15)=sig_tInj2_Arena2(8);
sig_tInj2(16)=sig_tInj2_Arena1(9);
sig_tInj2(17)=sig_tInj2_Arena2(9);
sig_tInj2(18)=sig_tInj2_Arena1(10);
sig_tInj2(19)=sig_tInj2_Arena1(11);
sig_tInj2(20)=sig_tInj2_Arena2(11);
sig_tInj2(21)=sig_tInj2_Arena1(12);
sig_tInj2(22)=sig_tInj2_Arena2(12);
sig_tInj2(23)=sig_tInj2_Arena1(13);


sig_tEnd(1)=sig_tEnd_Arena1(1);
sig_tEnd(2)=sig_tEnd_Arena2(1);
sig_tEnd(3)=sig_tEnd_Arena1(2);
sig_tEnd(4)=sig_tEnd_Arena2(2);
sig_tEnd(5)=sig_tEnd_Arena1(3);
sig_tEnd(6)=sig_tEnd_Arena2(3);
sig_tEnd(7)=sig_tEnd_Arena1(4);
sig_tEnd(8)=sig_tEnd_Arena2(4);
sig_tEnd(9)=sig_tEnd_Arena1(5);
sig_tEnd(10)=sig_tEnd_Arena1(6);
sig_tEnd(11)=sig_tEnd_Arena2(6);
sig_tEnd(12)=sig_tEnd_Arena1(7);
sig_tEnd(13)=sig_tEnd_Arena2(7);
sig_tEnd(14)=sig_tEnd_Arena1(8);
sig_tEnd(15)=sig_tEnd_Arena2(8);
sig_tEnd(16)=sig_tEnd_Arena1(9);
sig_tEnd(17)=sig_tEnd_Arena2(9);
sig_tEnd(18)=sig_tEnd_Arena1(10);
sig_tEnd(19)=sig_tEnd_Arena1(11);
sig_tEnd(20)=sig_tEnd_Arena2(11);
sig_tEnd(21)=sig_tEnd_Arena1(12);
sig_tEnd(22)=sig_tEnd_Arena2(12);
sig_tEnd(23)=sig_tEnd_Arena1(13);


noOfMice=23;
sig_data_Combined= cell(1,noOfMice);
sig_data_Combined{1}=table(sig_data{1}.SecFromZero,sig_data{1}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{2}=table(sig_data{1}.SecFromZero,sig_data{1}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{3}=table(sig_data{2}.SecFromZero,sig_data{2}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{4}=table(sig_data{2}.SecFromZero,sig_data{2}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{5}=table(sig_data{3}.SecFromZero,sig_data{3}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{6}=table(sig_data{3}.SecFromZero,sig_data{3}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{7}=table(sig_data{4}.SecFromZero,sig_data{4}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{8}=table(sig_data{4}.SecFromZero,sig_data{4}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{9}=table(sig_data{5}.SecFromZero,sig_data{5}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{10}=table(sig_data{6}.SecFromZero,sig_data{6}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{11}=table(sig_data{6}.SecFromZero,sig_data{6}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{12}=table(sig_data{7}.SecFromZero,sig_data{7}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{13}=table(sig_data{7}.SecFromZero,sig_data{7}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{14}=table(sig_data{8}.SecFromZero,sig_data{8}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{15}=table(sig_data{8}.SecFromZero,sig_data{8}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{16}=table(sig_data{9}.SecFromZero,sig_data{9}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{17}=table(sig_data{9}.SecFromZero,sig_data{9}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{18}=table(sig_data{10}.SecFromZero,sig_data{10}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{19}=table(sig_data{11}.SecFromZero,sig_data{11}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{20}=table(sig_data{11}.SecFromZero,sig_data{11}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{21}=table(sig_data{12}.SecFromZero,sig_data{12}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{22}=table(sig_data{12}.SecFromZero,sig_data{12}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
sig_data_Combined{23}=table(sig_data{13}.SecFromZero,sig_data{13}.Fiber1,'VariableNames',{'SecFromZero','NAc'});


ref_data_Combined= cell(1,noOfMice);
ref_data_Combined{1}=table(ref_data{1}.SecFromZero,ref_data{1}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{2}=table(ref_data{1}.SecFromZero,ref_data{1}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{3}=table(ref_data{2}.SecFromZero,ref_data{2}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{4}=table(ref_data{2}.SecFromZero,ref_data{2}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{5}=table(ref_data{3}.SecFromZero,ref_data{3}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{6}=table(ref_data{3}.SecFromZero,ref_data{3}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{7}=table(ref_data{4}.SecFromZero,ref_data{4}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{8}=table(ref_data{4}.SecFromZero,ref_data{4}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{9}=table(ref_data{5}.SecFromZero,ref_data{5}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{10}=table(ref_data{6}.SecFromZero,ref_data{6}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{11}=table(ref_data{6}.SecFromZero,ref_data{6}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{12}=table(ref_data{7}.SecFromZero,ref_data{7}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{13}=table(ref_data{7}.SecFromZero,ref_data{7}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{14}=table(ref_data{8}.SecFromZero,ref_data{8}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{15}=table(ref_data{8}.SecFromZero,ref_data{8}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{16}=table(ref_data{9}.SecFromZero,ref_data{9}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{17}=table(ref_data{9}.SecFromZero,ref_data{9}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{18}=table(ref_data{10}.SecFromZero,ref_data{10}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{19}=table(ref_data{11}.SecFromZero,ref_data{11}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{20}=table(ref_data{11}.SecFromZero,ref_data{11}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{21}=table(ref_data{12}.SecFromZero,ref_data{12}.Fiber1,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{22}=table(ref_data{12}.SecFromZero,ref_data{12}.Fiber3,'VariableNames',{'SecFromZero','NAc'});
ref_data_Combined{23}=table(ref_data{13}.SecFromZero,ref_data{13}.Fiber1,'VariableNames',{'SecFromZero','NAc'});


% for i =1:noOfMice
%     figure(i)
% plot(sig_data_Combined{i}.SecFromZero(sig_t0(i):sig_tEnd(i)),sig_data_Combined{i}.NAc(sig_t0(i):sig_tEnd(i)))
% hold on
% plot(ref_data_Combined{i}.SecFromZero(sig_t0(i):sig_tEnd(i)),ref_data_Combined{i}.NAc(sig_t0(i):sig_tEnd(i)))
% hold off
% end
%% dF/F conversion
%Specific range for technical background measurement
TechBackStart = 200;
TechBackEnd = 500;

%Specific range for biological background measurement

sig_dFF_NAc = cell(1,noOfMice);


for i = 1:noOfMice % Fiber1 dF/F
    BioBackStart(i) = sig_tInj1(i)-15000;
    BioBackEnd(i) = sig_tInj1(i)-5000;
    capSig = mean(sig_data_Combined{i}.NAc(TechBackStart:TechBackEnd));
    capRef = mean(ref_data_Combined{i}.NAc(TechBackStart:TechBackEnd));
    diffRefSig(i) = mean((sig_data_Combined{i}.NAc(BioBackStart(i):BioBackEnd(i))-capSig)-(ref_data_Combined{i}.NAc(BioBackStart(i):BioBackEnd(i))-capRef));
    cleanSig = sig_data_Combined{i}.NAc-capSig-diffRefSig(i);
    cleanRef = ref_data_Combined{i}.NAc-capRef;
%     figure(i+1000)
%     plot(cleanSig(sig_t0_Arena1(i):sig_tEnd_Arena1(i)))
%     hold on
%     plot(cleanRef(sig_t0_Arena1(i):sig_tEnd_Arena1(i)))
%     hold off


    sig_dFF_NAc{i} = (cleanSig(sig_t0(i):sig_tEnd(i))-movmean(cleanRef(sig_t0(i):sig_tEnd(i)),2))./movmean(cleanRef(sig_t0(i):sig_tEnd(i)),2);
    trend = sig_dFF_NAc{i}(1:sig_tInj2(i)-sig_t0(i))-detrend(sig_dFF_NAc{i}(1:sig_tInj2(i)-sig_t0(i)));%.*100;
    x_val = (trend(end)-trend(1))/length(sig_dFF_NAc{i}(1:sig_tInj2(i)-sig_t0(i))); %ax+b = y
    full_trend = linspace(trend(1),(x_val*length(sig_dFF_NAc{i}))+trend(1),length(sig_dFF_NAc{i}));
    sig_clean{i} = (sig_dFF_NAc{i} - full_trend.').*100;
% %     % Make sigClean not equal to zero
%     if min(sig_clean{i})<0;
%      sig_clean{i} = sig_clean{i}-min(sig_clean{i});
%     end
%     figure(1111+i)
%    plot(sig_clean{i})
end



figure
for i = drugA(1:end)
    plot(sig_clean{i}+i*20)
    hold on
end

figure
for i = drugB(1:end)
    plot(sig_clean{i}+i*20)
    hold on
end


figure
for i = drugC(1:end)
    plot(sig_clean{i}+i*20)
    hold on
end


figure
for i = 1:noOfMice
    plot(sig_clean{i}+i*20)
    hold on
end
%% Baseline

meanAsaline=[sig_clean{drugA(1)}(sig_tInj1(drugA(1)):sig_tInj1(drugA(1))+20*60*20),...
sig_clean{drugA(2)}(sig_tInj1(drugA(2)):sig_tInj1(drugA(2))+20*60*20),...
sig_clean{drugA(3)}(sig_tInj1(drugA(3)):sig_tInj1(drugA(3))+20*60*20),...
sig_clean{drugA(4)}(sig_tInj1(drugA(4)):sig_tInj1(drugA(4))+20*60*20),...
sig_clean{drugA(5)}(sig_tInj1(drugA(5)):sig_tInj1(drugA(5))+20*60*20),...
sig_clean{drugA(6)}(sig_tInj1(drugA(6)):sig_tInj1(drugA(6))+20*60*20)];

meanBsaline=[sig_clean{drugB(1)}(sig_tInj1(drugB(1)):sig_tInj1(drugB(1))+20*60*20),...
sig_clean{drugB(2)}(sig_tInj1(drugB(2)):sig_tInj1(drugB(2))+20*60*20),...
sig_clean{drugB(3)}(sig_tInj1(drugB(3)):sig_tInj1(drugB(3))+20*60*20),...
sig_clean{drugB(4)}(sig_tInj1(drugB(4)):sig_tInj1(drugB(4))+20*60*20),...
sig_clean{drugB(5)}(sig_tInj1(drugB(5)):sig_tInj1(drugB(5))+20*60*20),...
sig_clean{drugB(6)}(sig_tInj1(drugB(6)):sig_tInj1(drugB(6))+20*60*20)];

meanCsaline=[sig_clean{drugC(1)}(sig_tInj1(drugC(1)):sig_tInj1(drugC(1))+20*60*20),...
sig_clean{drugC(2)}(sig_tInj1(drugC(2)):sig_tInj1(drugC(2))+20*60*20) ,...
sig_clean{drugC(3)}(sig_tInj1(drugC(3)):sig_tInj1(drugC(3))+20*60*20) ,...
sig_clean{drugC(4)}(sig_tInj1(drugC(4)):sig_tInj1(drugC(4))+20*60*20) ,...
sig_clean{drugC(5)}(sig_tInj1(drugC(5)):sig_tInj1(drugC(5))+20*60*20) ,...
sig_clean{drugC(6)}(sig_tInj1(drugC(6)):sig_tInj1(drugC(6))+20*60*20)];


figure()
plot(meanAsaline)
figure()
plot(meanBsaline)
figure()
plot(meanCsaline)

%% Basal levels - AOC
basal_levels = cell(1,3);
for j = 1:3 
    for i = 1:noOfMice
        if j == 1
            range = 1:sig_tInj1(i)-200;
        elseif j == 2
            range = sig_tInj1(i)+200:sig_tInj2(i)-200;
        elseif j == 3
            range = sig_tInj2(i)+200:sig_tEnd(i)-sig_t0(i)-200;
        end
        
        basal_levels{j}{i} = basal_analysis(sig_clean{i}(range));
%        basal_levels_Fiber3{j}{i} = basal_analysis(sig_clean_Fiber3{i}(range));
    end
end
  
%%

Basal_lvl_A = cell(1,3);
Basal_lvl_B = cell(1,3);
Basal_lvl_C = cell(1,3);

pre_range = 30;
inj_range = 29;
coke_range = 50;

for j = 1:3
    
    if j == 1
        range = pre_range;
    elseif j == 2
        range = inj_range;
    elseif j == 3
        range = coke_range;
    end
    
    for i = drugA(1:end)
        Basal_lvl_A{j} = [Basal_lvl_A{j}; basal_levels{j}{i}(1:range)];
    end
    
    for i = drugB(1:end)
        Basal_lvl_B{j} = [Basal_lvl_B{j}; basal_levels{j}{i}(1:range)];
    end
    
    for i = drugC(1:end)
        Basal_lvl_C{j} = [Basal_lvl_C{j}; basal_levels{j}{i}(1:range)];
    end
   
end

CI95_Basal_lvl_A = cell(1,3);
CI95_Basal_lvl_B = cell(1,3);
CI95_Basal_lvl_C = cell(1,3);

for j = 1:3
    SEM_Basal_lvl_A = std(Basal_lvl_A{j}, [], 1)./ sqrt(size(Basal_lvl_A{j},1));    % Calculate Standard Error Of The Mean
    CI95_Basal_lvl_A{j} = bsxfun(@plus, mean(Basal_lvl_A{j},1), bsxfun(@times, [-1; 1], SEM_Basal_lvl_A));   % 95% Confidence Intervals
   %B 
    SEM_Basal_lvl_B = std(Basal_lvl_B{j}, [], 1)./ sqrt(size(Basal_lvl_B{j},1));    % Calculate Standard Error Of The Mean
    CI95_Basal_lvl_B{j} = bsxfun(@plus, mean(Basal_lvl_B{j},1), bsxfun(@times, [-1; 1], SEM_Basal_lvl_B));   % 95% Confidence Intervals
    %C
    SEM_Basal_lvl_C = std(Basal_lvl_C{j}, [], 1)./ sqrt(size(Basal_lvl_C{j},1));    % Calculate Standard Error Of The Mean
    CI95_Basal_lvl_C{j} = bsxfun(@plus, mean(Basal_lvl_C{j},1), bsxfun(@times, [-1; 1], SEM_Basal_lvl_C));   % 95% Confidence Intervals
    
end
% pre_range_plot = 1:pre_range;
% inj_range_plot = pre_range+2:pre_range+inj_range+1;
% coke_range_plot = pre_range+inj_range+3:pre_range+inj_range+coke_range+2;
pre_range_plot = -pre_range-inj_range-1:-inj_range-2;
inj_range_plot = -inj_range:-1;
coke_range_plot = 1:coke_range;


%% Area under the curve analysis

figure();
hold on
%A
plot(pre_range_plot,mean(Basal_lvl_A{1},1),'k','LineWidth',1.5)
fill([pre_range_plot, fliplr(pre_range_plot)],[CI95_Basal_lvl_A{1}(1,:), fliplr(CI95_Basal_lvl_A{1}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(inj_range_plot,mean(Basal_lvl_A{2},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([inj_range_plot, fliplr(inj_range_plot)],[CI95_Basal_lvl_A{2}(1,:), fliplr(CI95_Basal_lvl_A{2}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(coke_range_plot,mean(Basal_lvl_A{3},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([coke_range_plot, fliplr(coke_range_plot)],[CI95_Basal_lvl_A{3}(1,:), fliplr(CI95_Basal_lvl_A{3}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
xline(-inj_range-1,'Color',[0.4 0.4 0.4], 'LineStyle','--','LineWidth',1.5);
xline(0,'Color',[0.1 0.1 0.1], 'LineStyle','--','LineWidth',1.5);
title('Basal DA levels in NAc DRUG A (1 min. bins)')
ylabel('Basal level (Z-dF/F)')
xlabel('Time (min.)')
hold off

%B
figure();
hold on
plot(pre_range_plot,mean(Basal_lvl_B{1},1),'k','LineWidth',1.5)
fill([pre_range_plot, fliplr(pre_range_plot)],[CI95_Basal_lvl_B{1}(1,:), fliplr(CI95_Basal_lvl_B{1}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(inj_range_plot,mean(Basal_lvl_B{2},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([inj_range_plot, fliplr(inj_range_plot)],[CI95_Basal_lvl_B{2}(1,:), fliplr(CI95_Basal_lvl_B{2}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(coke_range_plot,mean(Basal_lvl_B{3},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([coke_range_plot, fliplr(coke_range_plot)],[CI95_Basal_lvl_B{3}(1,:), fliplr(CI95_Basal_lvl_B{3}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
xline(-inj_range-1,'Color',[0.4 0.4 0.4], 'LineStyle','--','LineWidth',1.5);
xline(0,'Color',[0.1 0.1 0.1], 'LineStyle','--','LineWidth',1.5);
title('Basal DA levels in NAc DRUG B(1 min. bins)')
ylabel('Basal level (Z-dF/F)')
xlabel('Time (min.)')
hold off

%C
figure();
hold on
plot(pre_range_plot,mean(Basal_lvl_C{1},1),'k','LineWidth',1.5)
fill([pre_range_plot, fliplr(pre_range_plot)],[CI95_Basal_lvl_C{1}(1,:), fliplr(CI95_Basal_lvl_C{1}(2,:))],...
   'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(inj_range_plot,mean(Basal_lvl_C{2},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([inj_range_plot, fliplr(inj_range_plot)],[CI95_Basal_lvl_C{2}(1,:), fliplr(CI95_Basal_lvl_C{2}(2,:))],...
 'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
plot(coke_range_plot,mean(Basal_lvl_C{3},1),'k','LineWidth',1.5,'HandleVisibility','off')
fill([coke_range_plot, fliplr(coke_range_plot)],[CI95_Basal_lvl_C{3}(1,:), fliplr(CI95_Basal_lvl_C{3}(2,:))],...
    'k','FaceAlpha', 0.2,'linestyle','none','HandleVisibility','off')
xline(-inj_range-1,'Color',[0.4 0.4 0.4], 'LineStyle','--','LineWidth',1.5);
xline(0,'Color',[0.1 0.1 0.1], 'LineStyle','--','LineWidth',1.5);
title('Basal DA levels in NAc DRUG C (1 min. bins)')
ylabel('Basal level (Z-dF/F)')
xlabel('Time (min.)')
hold off

%% WAVELET 
sig_LowFreq_NAc = cell(1,noOfMice);
sig_HighFreq_NAc = cell(1,noOfMice);
Residuals_NAc = cell(1,noOfMice);
Combined_NAc = cell(1,noOfMice);


for i = 1:noOfMice % NAc
    [sig_LowFreq_NAc{i}, sig_HighFreq_NAc{i}, Residuals_NAc{i}, Combined_NAc{i}] = Wavelet_Analysis(sig_clean{i},4);
end

%% Groups
basal_levelA = zeros(170621,1);
basal_levelB = zeros(170621,1);
basal_levelC = zeros(170621,1);
for i = drugA
   basal_levelA = [basal_levelA sig_clean{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620)-Combined_NAc{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620).'];
end

for i = drugB
   basal_levelB = [basal_levelB sig_clean{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620)-Combined_NAc{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620).'];
end

for i = drugC
   basal_levelC = [basal_levelC sig_clean{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620)-Combined_NAc{i}(sig_tInj2(i)-sig_t0(i)-100000:sig_tInj2(i)-sig_t0(i)+70620).'];
end
figure();
plot(mean(basal_levelA,2))
hold on
plot(mean(basal_levelB,2))
plot(mean(basal_levelC,2))
hold off

figure
hold on
stdshade(basal_levelA.',0.15,'b')
stdshade(basal_levelB.',0.15,'r')
stdshade(basal_levelC.',0.15,'g')

time = -83.3333:8.3333e-04:58.8500;
basal_levelA(:,1) = time;
basal_levelB(:,1) = time;
basal_levelC(:,1) = time;

%Old Export
% to_fig = [basal_levelA];
% csvwrite("Q:\Common\Lab member folder\Aske\Paper draft\Nicotine_Liraglutide\figure_data\Liraglutide.csv",to_fig)
% to_fig = [basal_levelB];
% csvwrite("Q:\Common\Lab member folder\Aske\Paper draft\Nicotine_Liraglutide\figure_data\Both.csv",to_fig)
% to_fig = [basal_levelC];
% csvwrite("Q:\Common\Lab member folder\Aske\Paper draft\Nicotine_Liraglutide\figure_data\Nicotine.csv",to_fig)


%New export - directly to working directory
to_fig = [basal_levelA];
csvwrite("Liraglutide.csv",to_fig)
to_fig = [basal_levelB];
csvwrite("Both.csv",to_fig)
to_fig = [basal_levelC];
csvwrite("Nicotine.csv",to_fig)


%% Functions

function [TrackingData] = load_trackingfile(trackingfile)
    
opts=detectImportOptions(trackingfile);
%Defines the row location of channel variable name
opts.VariableNamesLine = 1;
%Specifies that the data is comma seperated
opts.Delimiter =' '; %Specifies that the data is space seperated
%Read the table
TrackingData = readtable(trackingfile,opts, 'ReadVariableNames', true);
TrackingData.SecFromZero = TrackingData.BonsaiTrackingTimestamp - TrackingData.BonsaiTrackingTimestamp(1);
end

function [Reference, Signal, Time] = load_File(file_name,TrackingData)
    
opts=detectImportOptions(file_name);
%Defines the row location of channel variable name
opts.VariableNamesLine = 1;
%Specifies that the data is comma seperated
opts.Delimiter =' '; %Specifies that the data is space seperated
%Read the table
Data = readtable(file_name,opts, 'ReadVariableNames', true);

%SecFromZero is synched to the first datapoint of the faster basler camera
Data.SecFromZero = Data.BonsaiFlyTimestamp -TrackingData.BonsaiTrackingTimestamp(1);

    %Parse out with downsample to get 470 signal and 415 reference 
    Signal = downsample(Data, 2, 1);
    Reference = downsample(Data, 2);

    %Adjust length of variable to be of equal lengths
    if(length(Reference.SecFromZero)>length(Signal.SecFromZero)==1)
        Reference(end,:) = [];
    end

    if(length(Signal.SecFromZero)>length(Reference.SecFromZero)==1)
        Signal(end,:) = [];
    end
    
    Time = Signal.SecFromZero;
end


function [basal_levels] = basal_analysis(sig_data)
    
    block_size = 1200; % 1 min time bins
    blocks = floor(length(sig_data)/block_size);
    start_range = 1;
    end_range = block_size;
    
    basal_levels = [];

    for i = 1:blocks
        
        %AOC = trapz(sig_data(start_range:end_range)/length(sig_data(start_range:end_range)));
        basal_levels = [basal_levels, median(sig_data(start_range:end_range))];
        
        start_range = start_range + block_size;
        end_range = end_range + block_size;
    end
end


function [sig_LowFreq, sig_HighFreq, HighFreq_Residuals, combined] = Wavelet_Analysis(sig_clean,coord_level)
    %RETURNS LEVEL 4 AND LEVEL 1 CONTRIBUTION TO TOTAL SIGNAL BY WAVELET
    %ANAYLIS USING DB2 WAVELET AND COMPUTES THE FINAL RESIDUALS
    wavelet_type = 'sym4';
    LowPass_lvl = 10;
    LowPassDec = modwt(sig_clean,wavelet_type,LowPass_lvl); LowPassRec = modwtmra(LowPassDec,wavelet_type);
    sig_LowPass= LowPassRec(LowPass_lvl+1,:);
    LowPass_Residuals = sig_clean-sig_LowPass.';
    
    LowFreq_lvl = coord_level;
    LowFreqDec = modwt(LowPass_Residuals,wavelet_type,LowFreq_lvl); LowFreqRec = modwtmra(LowFreqDec,wavelet_type);
    sig_LowFreq= LowFreqRec(LowFreq_lvl+1,:);
    LowFreq_Residuals = LowPass_Residuals-sig_LowFreq.';
    
    HighFreq_lvl = 1;
    HighFreqDec = modwt(LowFreq_Residuals,wavelet_type,HighFreq_lvl); HighFreqRec = modwtmra(HighFreqDec,wavelet_type);
    sig_HighFreq = HighFreqRec(HighFreq_lvl+1,:); HighFreq_Residuals = LowFreq_Residuals-sig_HighFreq.';

    combined = sig_LowFreq + sig_HighFreq + HighFreq_Residuals.';
end

function [ry,test,f] = Fourier_Pass_Filter(z_dFF,timeAxis, cutoff, filter_type)
    %Fourier_PassFilter(z_dFF_Ctrl,Signal_Ctrl.SecFromZero,1025.9,0.25,1);
    
    samplingtime = max(timeAxis) - min(timeAxis);
    %Power spectrum
    fy=fft(z_dFF.'); %Make a check to see dimensions and possibly transpose
    py=fy .* conj(fy);
    
    %Filter
    plotrange = 2:length(fy)/2;
    f=((plotrange-1)./samplingtime);
    [val, idx] = min(abs(f-cutoff));
    filt1 = ones(1,idx);
    filt2 = zeros(1,length(plotrange)-idx+1);
    lefthand_filter = [filt1, filt2]; 
    if rem(size(timeAxis),2) == 1
        total_filter = [lefthand_filter, 0, flip(lefthand_filter)];
    else
        total_filter = [lefthand_filter, flip(lefthand_filter)];
    end
    if filter_type==1
        total_filter=1-total_filter; 
    end
    residual_filter = 1-total_filter;
    ffy=fy.*total_filter;
    ffy_res=fy.*residual_filter;
    ry=real(ifft(ffy));
    ry_res=real(ifft(ffy_res));
    
    test = real(py(plotrange))/sum(real(py(plotrange)))*100;
    
%     plot
%     figure
%     subplot(2,2,1)
%     plot(timeAxis,z_dFF,'k')
%     title('Input')
%     subplot(2,2,[3 4])
%     plot(f, real(py(plotrange))/sum(real(py(plotrange)))*100);%/sum(real(py(plotrange)))*100
%     xlim([0.001 10])%[0.001 10
%     hold on
%     plot(f,max(real(py(plotrange))).*total_filter(plotrange),'LineWidth',2);
%     title('Power Spectrum and filter')
%     legend('Power','Filter')
%     subplot(2,2,2)
%     plot(timeAxis,ry,'g')
%     hold on
%     plot(timeAxis,ry_res,'r')
%     title('Filtered Data')
%     legend('Filtered','Residual')
    
end

function [mean_amplitude, frequency, mean_width, mean_decay] = prominence_analysis(WL_series,filter)
    
    block_size = 1200; % 1 min time bins
    blocks = floor(length(WL_series)/block_size);
    start_range = 1;
    end_range = block_size;
    
    mean_amplitude = [];
    mean_width = [];
    mean_decay = [];
    frequency = [];

    
    for i = 1:blocks
        
        if filter == "NAc"
            [pks,locs,w,p] = findpeaks(WL_series(start_range:end_range),'MinPeakProminence',1);
        elseif filter == "DS"
            [pks,locs,w,p] = findpeaks(WL_series(start_range:end_range),'MinPeakProminence',3);
        elseif filter == "Asynch"
            [pks,locs,w,p] = findpeaks(WL_series(start_range:end_range));
        end
        mean_amplitude = [mean_amplitude, mean(p)];
        mean_width = [mean_width, mean(w)/20]; %Sampled at 20 Hz so converting to sec
        mean_decay = [mean_decay, (mean(p)/2)/(mean(w)/20)]; % Half prominence divded by width at half prominence for clearance rate
        frequency = [frequency, (length(p)/(block_size/20))];
        
        start_range = start_range + block_size;
        end_range = end_range + block_size;
    end

end

