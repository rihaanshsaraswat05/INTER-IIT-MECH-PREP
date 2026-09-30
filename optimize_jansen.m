clc;
clear;
close all;

%% ============================================================
%  JANSEN LINKAGE - TARGET SHAPE OPTIMIZATION
%
%  Objective:
%  1. Minimize complete foot-path error
%  2. Match the long flat stance phase
%  3. Match 200 mm peak lift
%  4. Match target stride
%  5. Keep dimensions practical
%
%  No Optimization Toolbox required.
%
%  INPUT:
%       target_foot_tip_path.csv
%
%  CSV format:
%       theta   x   y
%
%  OUTPUT:
%       optimized_jansen_dimensions.csv
%       optimized_jansen_path.csv
%% ============================================================


%% 1. LOAD TARGET

data = readmatrix('target_foot_tip_path.csv');

thetaTarget = data(:,1);
xTarget     = data(:,2);
yTarget     = data(:,3);

N = length(thetaTarget);

fprintf('Target points = %d\n',N);


%% ============================================================
% 2. STANDARD JANSEN DIMENSIONS
%
% a b c d e f g h i j k l m
%% ============================================================

L0 = [ ...
    38.0 ...
    41.5 ...
    39.3 ...
    40.1 ...
    55.8 ...
    39.4 ...
    36.7 ...
    65.7 ...
    49.0 ...
    50.0 ...
    61.9 ...
     7.8 ...
    15.0];


%% ============================================================
% 3. TARGET FEATURES
%% ============================================================

targetHeight = max(yTarget)-min(yTarget);

targetStride = max(xTarget)-min(xTarget);

fprintf('\nTarget height = %.3f mm\n',targetHeight);
fprintf('Target stride = %.3f mm\n',targetStride);


%% ============================================================
% 4. IDENTIFY FLAT STANCE REGION
%
% Points close to ground are treated as stance points.
%
% This is very important for your competition target.
%% ============================================================

stanceLimit = min(yTarget) + 0.08*targetHeight;

stanceIndex = find(yTarget <= stanceLimit);

fprintf('Stance points = %d\n',length(stanceIndex));


%% ============================================================
% 5. OPTIMIZATION VARIABLES
%
% x(1:13) = link-length multipliers
% x(14)   = crank phase
%
% We start near the standard Jansen dimensions.
%% ============================================================

nVar = 14;

LB = [ ...
    0.70*ones(1,13), ...
   -180];

UB = [ ...
    1.30*ones(1,13), ...
    180];


%% ============================================================
% 6. DIFFERENTIAL EVOLUTION
%% ============================================================

populationSize = 60;

generations = 500;

F = 0.65;

CR = 0.90;


population = zeros(populationSize,nVar);

for p = 1:populationSize

    population(p,:) = LB + ...
        rand(1,nVar).*(UB-LB);

end


% Put standard Jansen into population

population(1,1:13) = ones(1,13);

population(1,14) = 0;


%% ============================================================
% 7. INITIAL FITNESS
%% ============================================================

fitness = zeros(populationSize,1);

for p = 1:populationSize

    fitness(p) = objectiveFunction( ...
        population(p,:), ...
        L0, ...
        thetaTarget, ...
        xTarget, ...
        yTarget, ...
        stanceIndex, ...
        targetStride);

end


[bestFitness,bestID] = min(fitness);

bestX = population(bestID,:);


fprintf('\n====================================\n');
fprintf('STARTING OPTIMIZATION\n');
fprintf('====================================\n');

fprintf('Initial error = %.3f mm\n',bestFitness);


%% ============================================================
% 8. DIFFERENTIAL EVOLUTION LOOP
%% ============================================================

for gen = 1:generations

    for p = 1:populationSize

        % Random permutation

        r = randperm(populationSize);

        r(r == p) = [];

        r1 = r(1);
        r2 = r(2);
        r3 = r(3);


        %% Mutation

        mutant = population(r1,:) + ...
            F*(population(r2,:) - population(r3,:));


        %% Bound mutant

        mutant = max(mutant,LB);

        mutant = min(mutant,UB);


        %% Crossover

        trial = population(p,:);

        forced = randi(nVar);

        for j = 1:nVar

            if rand < CR || j == forced

                trial(j) = mutant(j);

            end

        end


        %% Trial fitness

        trialFitness = objectiveFunction( ...
            trial, ...
            L0, ...
            thetaTarget, ...
            xTarget, ...
            yTarget, ...
            stanceIndex, ...
            targetStride);


        %% Selection

        if trialFitness < fitness(p)

            population(p,:) = trial;

            fitness(p) = trialFitness;

        end

    end


    %% Best solution

    [generationBest,id] = min(fitness);


    if generationBest < bestFitness

        bestFitness = generationBest;

        bestX = population(id,:);

    end


    if mod(gen,10) == 0

        fprintf( ...
            'Generation %3d / %3d    Error = %.3f\n', ...
            gen, ...
            generations, ...
            bestFitness);

    end

end


%% ============================================================
% 9. LOCAL REFINEMENT
%
% Differential evolution finds the region.
% fminsearch performs final fine tuning.
%% ============================================================

fprintf('\nStarting local refinement...\n');


options = optimset( ...
    'MaxIter',5000, ...
    'MaxFunEvals',20000, ...
    'Display','iter');


bestX = fminsearch( ...
    @(x)objectiveFunction( ...
        x, ...
        L0, ...
        thetaTarget, ...
        xTarget, ...
        yTarget, ...
        stanceIndex, ...
        targetStride), ...
    bestX, ...
    options);


%% ============================================================
% 10. FINAL DIMENSIONS
%% ============================================================

ratios = bestX(1:13);

phase = bestX(14);

Lraw = L0 .* ratios;


%% ============================================================
% 11. CALCULATE RAW PATH
%% ============================================================

thetaFinal = thetaTarget + phase;

rawPath = jansenPath(Lraw,thetaFinal);


if any(isnan(rawPath(:)))

    error('Final mechanism contains invalid positions.');

end


xRaw = rawPath(:,1);

yRaw = rawPath(:,2);


%% ============================================================
% 12. SCALE TO EXACTLY 200 mm
%% ============================================================

rawHeight = max(yRaw)-min(yRaw);

scale = targetHeight/rawHeight;

Lfinal = Lraw*scale;

xFinal = xRaw*scale;

yFinal = yRaw*scale;


%% ============================================================
% 13. TRANSLATION
%% ============================================================

xFinal = xFinal - min(xFinal) + min(xTarget);

yFinal = yFinal - min(yFinal) + min(yTarget);


%% ============================================================
% 14. FINAL ERROR
%% ============================================================

dx = xFinal-xTarget;

dy = yFinal-yTarget;

pointError = sqrt(dx.^2+dy.^2);

RMS = sqrt(mean(pointError.^2));

finalHeight = max(yFinal)-min(yFinal);

finalStride = max(xFinal)-min(xFinal);


%% ============================================================
% 15. STANCE ERROR
%% ============================================================

stanceY = yFinal(stanceIndex);

targetStanceY = yTarget(stanceIndex);

stanceRMSE = sqrt( ...
    mean((stanceY-targetStanceY).^2));


%% ============================================================
% 16. PEAK ERROR
%% ============================================================

[maxTarget,peakTargetIndex] = max(yTarget);

[maxFinal,peakFinalIndex] = max(yFinal);

peakError = abs(maxFinal-maxTarget);


%% ============================================================
% 17. PRINT FINAL RESULT
%% ============================================================

fprintf('\n\n');
fprintf('============================================\n');
fprintf('FINAL OPTIMIZED JANSEN MECHANISM\n');
fprintf('============================================\n');

fprintf('a = %.3f mm\n',Lfinal(1));
fprintf('b = %.3f mm\n',Lfinal(2));
fprintf('c = %.3f mm\n',Lfinal(3));
fprintf('d = %.3f mm\n',Lfinal(4));
fprintf('e = %.3f mm\n',Lfinal(5));
fprintf('f = %.3f mm\n',Lfinal(6));
fprintf('g = %.3f mm\n',Lfinal(7));
fprintf('h = %.3f mm\n',Lfinal(8));
fprintf('i = %.3f mm\n',Lfinal(9));
fprintf('j = %.3f mm\n',Lfinal(10));
fprintf('k = %.3f mm\n',Lfinal(11));
fprintf('l = %.3f mm\n',Lfinal(12));
fprintf('m = %.3f mm\n',Lfinal(13));

fprintf('\nCrank phase = %.3f deg\n',phase);

fprintf('\n--------------------------------------------\n');

fprintf('Target height       = %.3f mm\n',targetHeight);

fprintf('Final height        = %.3f mm\n',finalHeight);

fprintf('Target stride       = %.3f mm\n',targetStride);

fprintf('Final stride        = %.3f mm\n',finalStride);

fprintf('Overall RMS error   = %.3f mm\n',RMS);

fprintf('Stance RMS error    = %.3f mm\n',stanceRMSE);

fprintf('Peak height error   = %.3f mm\n',peakError);

fprintf('--------------------------------------------\n');


%% ============================================================
% 18. TARGET VS OPTIMIZED PATH
%% ============================================================

figure;

plot(xTarget,yTarget,'k','LineWidth',2);

hold on;

plot(xFinal,yFinal,'--','LineWidth',2);

grid on;

axis equal;

xlabel('X (mm)');

ylabel('Y (mm)');

title('TARGET vs OPTIMIZED FOOT TRAJECTORY');

legend( ...
    'Target', ...
    'Optimized Jansen', ...
    'Location','best');


%% ============================================================
% 19. ERROR MAP
%% ============================================================

figure;

plot(thetaTarget,pointError,'LineWidth',1.5);

grid on;

xlabel('Crank angle (deg)');

ylabel('Position error (mm)');

title('Trajectory Error');


%% ============================================================
% 20. HEIGHT COMPARISON
%% ============================================================

figure;

plot(thetaTarget,yTarget,'w','LineWidth',2);

hold on;

plot(thetaTarget,yFinal,'--','LineWidth',2);

grid on;

xlabel('Crank angle (deg)');

ylabel('Foot height (mm)');

title('Foot Height vs Crank Angle');

legend( ...
    'Target', ...
    'Optimized', ...
    'Location','best');


%% ============================================================
% 21. STANCE ZOOM
%% ============================================================

figure;

plot(xTarget,yTarget,'w','LineWidth',2);

hold on;

plot(xFinal,yFinal,'--','LineWidth',2);

grid on;

xlim([min(xTarget)-20 max(xTarget)+20]);

ylim([min(yTarget)-10 50]);

xlabel('X (mm)');

ylabel('Y (mm)');

title('STANCE PHASE COMPARISON');

legend('Target','Optimized');


%% ============================================================
% 22. SAVE RESULTS
%% ============================================================

result = [ ...
    thetaTarget ...
    xTarget ...
    yTarget ...
    xFinal ...
    yFinal ...
    pointError];


writematrix( ...
    result, ...
    'optimized_jansen_path.csv');


writematrix( ...
    Lfinal, ...
    'optimized_jansen_dimensions.csv');


fprintf('\nFiles saved:\n');

fprintf('optimized_jansen_path.csv\n');

fprintf('optimized_jansen_dimensions.csv\n');


%% ============================================================
% OBJECTIVE FUNCTION
%% ============================================================

function errorValue = objectiveFunction( ...
    variable, ...
    L0, ...
    theta, ...
    xTarget, ...
    yTarget, ...
    stanceIndex, ...
    targetStride)


%% Extract variables

ratios = variable(1:13);

phase = variable(14);


%% Prevent extreme dimensions

if any(ratios < 0.70) || any(ratios > 1.30)

    errorValue = 1e8;

    return;

end


%% Link lengths

L = L0 .* ratios;


%% Mechanism path

thetaNew = theta + phase;

path = jansenPath(L,thetaNew);


%% Invalid configuration

if any(isnan(path(:)))

    errorValue = 1e8;

    return;

end


x = path(:,1);

y = path(:,2);


%% Raw height

height = max(y)-min(y);


if height < 10

    errorValue = 1e8;

    return;

end


%% Scale to target height

scale = 200/height;

x = x*scale;

y = y*scale;


%% Translation

x = x-min(x)+min(xTarget);

y = y-min(y)+min(yTarget);


%% =========================================================
% ERROR 1: COMPLETE TRAJECTORY
%% =========================================================

dx = x-xTarget;

dy = y-yTarget;

trajectoryError = sqrt(mean(dx.^2+dy.^2));


%% =========================================================
% ERROR 2: FLAT STANCE
%
% Give stance 4x more importance.
%% =========================================================

stanceError = sqrt( ...
    mean((y(stanceIndex)-yTarget(stanceIndex)).^2));


%% =========================================================
% ERROR 3: STRIDE
%% =========================================================

stride = max(x)-min(x);

strideError = abs(stride-targetStride);


%% =========================================================
% ERROR 4: PEAK LOCATION
%
% We want the high point to occur at approximately
% the same crank phase as the target.
%% =========================================================

[~,targetPeak] = max(yTarget);

[~,generatedPeak] = max(y);

peakPhaseError = ...
    abs(theta(generatedPeak)-theta(targetPeak));


%% =========================================================
% ERROR 5: HEIGHT
%% =========================================================

heightError = abs( ...
    (max(y)-min(y))-200);


%% =========================================================
% TOTAL OBJECTIVE
%% =========================================================

errorValue =1.00*trajectoryError + 4.00*stanceError + 0.15*strideError + 0.10*peakPhaseError + 2.00*heightError;


end


%% ============================================================
% JANSEN FORWARD KINEMATICS
%% ============================================================

function path = jansenPath(L,theta)

a = L(1);
B = L(2);
C = L(3);
D = L(4);
E = L(5);
F = L(6);
G = L(7);
H = L(8);
I = L(9);
J = L(10);
K = L(11);
l = L(12);
M = L(13);


N = length(theta);

path = nan(N,2);


%% Fixed diagonal

A = sqrt(a^2+l^2);

thetaA = atan2(l,a);


%% Triangle B-D-E

valueD = ...
    (E^2+B^2-D^2)/(2*E*B);

valueD = max(-1,min(1,valueD));

thetaD1 = acos(valueD);


valueB = ...
    B/D*sin(thetaD1);

valueB = max(-1,min(1,valueB));

thetaTriB = asin(valueB);


valueE = ...
    E/D*sin(thetaD1);

valueE = max(-1,min(1,valueE));

thetaTriE = asin(valueE);


%% Triangle H-I-G

valueG = ...
    (H^2+I^2-G^2)/(2*H*I);

valueG = max(-1,min(1,valueG));

thetaTriG = acos(valueG);


valueH = ...
    H/G*sin(thetaTriG);

valueH = max(-1,min(1,valueH));

thetaTriH = asin(valueH);


%% =========================================================
% MAIN LOOP
%% =========================================================

for q = 1:N


    %% FIRST FOUR BAR

    theta2 = deg2rad(180+theta(q))-thetaA;


    [thetaJ,thetaB,valid1] = ...
        fourBar(M,J,B,A,theta2,-1);


    if ~valid1

        continue;

    end


    thetaJ = thetaJ-pi+thetaA;

    thetaB = thetaB-pi+thetaA;


    %% SECOND FOUR BAR

    theta2 = ...
        deg2rad(theta(q))-thetaA-pi;


    [thetaK,thetaC,valid2] = ...
        fourBar(M,K,C,A,theta2,1);


    if ~valid2

        continue;

    end


    thetaK = pi+thetaA+thetaK;

    thetaC = pi+thetaA+thetaC;


    %% DIRECTION OF D

    thetaD = thetaB+thetaTriE;


    %% POINT D

    Dx = D*cos(thetaD);

    Dy = D*sin(thetaD);


    %% POINT C

    Cx = C*cos(thetaC);

    Cy = C*sin(thetaC);


    %% D-C GHOST LINK

    Sx = Dx-Cx;

    Sy = Dy-Cy;

    S = sqrt(Sx^2+Sy^2);


    if S < 1e-8

        continue;

    end


    thetaS = atan2(Sy,Sx);


    %% Keep consistent orientation

    if thetaS < 0

        thetaS = thetaS+pi;

    end


    %% TRIANGLE F-G-S

    valueF = ...
        (G^2+S^2-F^2)/(2*G*S);


    if abs(valueF)>1

        continue;

    end


    thetaTriF = acos(valueF);


    valueS = ...
        S/F*sin(thetaTriF);


    if abs(valueS)>1

        continue;

    end


    thetaG = thetaS+thetaTriF;


    %% TRIANGLE G-H-I

    thetaI = thetaG+thetaTriH;


    %% FOOT

    xFoot = Cx+I*cos(thetaI);

    yFoot = Cy+I*sin(thetaI);


    path(q,:) = [xFoot yFoot];

end

end


%% ============================================================
% FOUR BAR SOLVER
%% ============================================================

function [theta3,theta4,valid] = ...
    fourBar(a,b,c,d,theta2,delta)


valid = true;


k1 = d/a;

k2 = d/c;


k3 = ...
    (a^2-b^2+c^2+d^2)/(2*a*c);


k4 = d/b;


k5 = ...
    (c^2-d^2-a^2-b^2)/(2*a*b);


AA = ...
    cos(theta2)-k1-k2*cos(theta2)+k3;


BB = -2*sin(theta2);


CC = ...
    k1-(k2+1)*cos(theta2)+k3;


DD = ...
    cos(theta2)-k1+k4*cos(theta2)+k5;


EE = -2*sin(theta2);


FF = ...
    k1+(k4-1)*cos(theta2)+k5;


disc1 = BB^2-4*AA*CC;

disc2 = EE^2-4*DD*FF;


if disc1 < 0 || disc2 < 0

    theta3 = NaN;

    theta4 = NaN;

    valid = false;

    return;

end


if abs(AA)<1e-12 || abs(DD)<1e-12

    theta3 = NaN;

    theta4 = NaN;

    valid = false;

    return;

end


if delta == 1

    theta4 = ...
        2*atan( ...
        (-BB-sqrt(disc1))/(2*AA));


    theta3 = ...
        2*atan( ...
        (-EE-sqrt(disc2))/(2*DD));

else

    theta4 = ...
        2*atan( ...
        (-BB+sqrt(disc1))/(2*AA));


    theta3 = ...
        2*atan( ...
        (-EE+sqrt(disc2))/(2*DD));

end

end