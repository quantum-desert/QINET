clear variables
close all
clc

%% Directory containing CSV files

rootp = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/9-29-Phase/phiLO=+33/";   % change if necessary
% rootp = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/calibration data/V_PM_check_9-28/small_sig/";
%% Find all scope CSV files

files = dir(fullfile(rootp, ...
"SDS5104X_HD_CSV_DC*V_C*_*.csv"))

% measuremnt config
config = struct();
config.tau_LIA = 10e-3; % s
config.rolloff_LIA=12;  % dB / oct
config.files=files;
config.cscheme = 'cool';
config.theta_dev = 2*pi; % set on PM input
config.V_max = 6; % PM max voltage for full theta_dev swing

[t_sample,V_sample] = readSiglentCSV(fullfile(configfiles(1).folder,config.files(1).name));
config.step = compute_ds(config,t_sample(end)-t_sample(1),numel(V_sample));

%% Read and plot
figure(1); ax=gca;
stats = read_and_plot(config,ax);
return



%% Read files
traces = struct();

for k = 1:length(files)
% for k = 1:1

    filename = files(k).name;
    filepath = fullfile(files(k).folder, filename);

    % Extract DC voltage and channel from filename
    %
    % Example:
    % SDS5104X_HD_CSV_DC3V_C4_00000001.csv
    %
    % tokens{1} = '3'
    % tokens{2} = '4'

    tokens = regexp(filename, '_DC(-?\d+(?:\.\d+)?)V_C([34])_(\d+)\.csv$', 'tokens', 'once');

    % Skip files that do not match
    if isempty(tokens)
        continue
    end

    DC_voltage = str2double(tokens{1});
    channel    = str2double(tokens{2});
    rep_idx    = str2double(tokens{3});   % NEW

    [t, V] = readSiglentCSV(filepath);

    tag   = strrep(strrep(tokens{1}, '.', 'p'), '-', 'm');
    group = ['DC' tag 'V'];

    traces.(group).DC = DC_voltage;
    traces.(group).rep(rep_idx).t = t;                 % CHANGED: indexed by rep
    if channel == 3
        traces.(group).rep(rep_idx).Y = V;
    elseif channel == 4
        traces.(group).rep(rep_idx).X = V;
    end


end




%% for each group, estimate phase and generate histogram
f1=figure; hold on; theme light; ax1=gca;
f2=figure; hold on; theme light; ax2=gca;
skip = [];
DCgroups = fieldnames(traces);
nG = length(DCgroups);
colororder(ax1, cool(nG-length(skip)));

% total rows = sum of repeats across all non-skipped groups
nRows = 0;
for k = 1:nG
    if ismember(k,skip), continue; end
    nRows = nRows + numel(traces.(DCgroups{k}).rep);
end

DC_V                 = nan(nRows,1);
rep_num              = nan(nRows,1);          % NEW: which repeat this row is
N_pts                = nan(nRows,1);
gamma_mean_rad       = nan(nRows,1);   % first moment (circular mean)
gamma_mean_rad_pred  = nan(nRows,1);
gamma_std_rad        = nan(nRows,1);   % second moment (circular std about the mean)
R_mean               = nan(nRows,1);

row = 0;   % running index across all (group, rep) pairs
for k = 1:nG
    if(ismember(k,skip))
        continue
    end
    group = DCgroups{k};
    nReps = numel(traces.(group).rep);          % CHANGED: was "1:traces.(group).rep" (empty loop)
    for rep_idx = 1:nReps
        r = traces.(group).rep(rep_idx);
        if isempty(r.X) || isempty(r.Y)          % guard: a rep missing its C3 or C4 file
            continue
        end
        t = r.t(1:downsample_step:end);
        X = r.X(1:downsample_step:end);
        Y = r.Y(1:downsample_step:end);
        gamma = atan2(Y,X);

        h = histogram(ax2, gamma./pi, ...
            DisplayName = sprintf('%s (rep %d)', group, rep_idx), ...   % CHANGED: was just group
            Normalization = "probability", BinWidth = 0.025);
% pause(1)
        mu    = angle(mean(exp(1i*gamma)));               % circular mean
        dev   = angle(exp(1i*(gamma - mu)));               % deviations, wrapped to (-pi,pi]
        sigma = sqrt(sum(dev.^2)/(numel(gamma)-1));         % circular std about mu

        row = row + 1;                                      % NEW: one row per repeat, not per group
        DC_V(row)                = traces.(group).DC;
        rep_num(row)             = rep_idx;                 % NEW
        N_pts(row)               = numel(gamma);
        gamma_mean_rad(row)      = mu/pi;
        gamma_mean_rad_pred(row) = traces.(group).DC*theta_dev/V_max;
        gamma_std_rad(row)       = sigma/pi;
        R_mean(row)              = mean(hypot(X,Y));
    end
end

keep = ~isnan(DC_V);   % drops any skipped/empty rows
phaseStatsTable = table(DC_V(keep), rep_num(keep), N_pts(keep), gamma_mean_rad(keep), gamma_mean_rad_pred(keep), gamma_std_rad(keep), R_mean(keep), ...
    'VariableNames', {'DC_V','rep','N_pts','gamma_mean_rad (/pi)','gamma_mean_rad pred. (/pi)','gamma_std_rad (/pi)','R_mean'});
phaseStatsTable = sortrows(phaseStatsTable, ["DC_V","rep"], "ascend");
disp(phaseStatsTable)



%% plotting / styling
set(f1, 'CurrentAxes', ax1)
legend(ax1);

set(f2, 'CurrentAxes', ax2)
legend(ax2);
title('Phase Detection')
xlabel('\gamma/\pi'); ylabel('counts')
xlim([-1 1]);


% plot gamma trend
figure; hold on; theme light;
scatter(phaseStatsTable.DC_V,phaseStatsTable.("gamma_mean_rad (/pi)"),DisplayName='Measured \gamma',LineWidth=2);
scatter(phaseStatsTable.DC_V,phaseStatsTable.("gamma_mean_rad pred. (/pi)"),DisplayName='Predicted \gamma',LineWidth=2);
xlabel('V_{DC} [V]'); ylabel('\gamma'); legend();



%% functions
function [t, V] = readSiglentCSV(filename)
% readSiglentCSV  Read waveform data from a Siglent oscilloscope CSV file
%
%   [t, V] = readSiglentCSV(filename)
%
% Inputs:
%   filename - CSV filename or full file path
%
% Outputs:
%   t - time vector [s]
%   V - measured voltage [V]

    % Import starting at the column-header row
    opts = detectImportOptions(filename);
    opts.DataLines = [15, Inf];   % waveform data starts on line 15

    data = readtable(filename, opts);

    % Extract waveform columns
    t = data{:,1};
    V = data{:,2};

end

function [tw,ENBW] = LIA_timing(T,slope)
switch slope
    case 6
        tw=5*T;
        ENBW=1/(4*T);
    case 12
        tw=7*T;
        ENBW=1/(8*T);
    case 18
        tw=9*T;
        ENBW=3/(32*T);
    case 24
        tw=10*T;
        ENBW=5/(64*T);
end
end

function statsTable = read_and_plot(ax,directory,cscheme,step)
    % read in all files from directory and add histograms to supplied fig wqith
    % unique color scheme
    traces = struct();
    
    for k = 1:length(directory)
    % for k = 1:1
    
        filename = directory(k).name;
        filepath = fullfile(directory(k).folder, filename);
    
        % Extract DC voltage and channel from filename
        %
        % Example:
        % SDS5104X_HD_CSV_DC3V_C4_00000001.csv
        %
        % tokens{1} = '3'
        % tokens{2} = '4'
    
        tokens = regexp(filename, '_DC(-?\d+(?:\.\d+)?)V_C([34])_(\d+)\.csv$', 'tokens', 'once');
    
        % Skip files that do not match
        if isempty(tokens)
            continue
        end
    
        DC_voltage = str2double(tokens{1});
        channel    = str2double(tokens{2});
        rep_idx    = str2double(tokens{3});   % NEW
    
        [t, V] = readSiglentCSV(filepath);
    
        tag   = strrep(strrep(tokens{1}, '.', 'p'), '-', 'm');
        group = ['DC' tag 'V'];
    
        traces.(group).DC = DC_voltage;
        traces.(group).rep(rep_idx).t = t;                 % CHANGED: indexed by rep
        if channel == 3
            traces.(group).rep(rep_idx).Y = V;
        elseif channel == 4
            traces.(group).rep(rep_idx).X = V;
        end
    
    
    end
    
    DCgroups = fieldnames(traces);
    nG = length(DCgroups);
    colororder(ax, feval(cscheme,nG));
    
    % total rows = sum of repeats across all non-skipped groups
    nRows = 0;
    for k = 1:nG
        nRows = nRows + numel(traces.(DCgroups{k}).rep);
    end
    
    DC_V                 = nan(nRows,1);
    rep_num              = nan(nRows,1);          % NEW: which repeat this row is
    N_pts                = nan(nRows,1);
    gamma_mean_rad       = nan(nRows,1);   % first moment (circular mean)
    gamma_mean_rad_pred  = nan(nRows,1);
    gamma_std_rad        = nan(nRows,1);   % second moment (circular std about the mean)
    R_mean               = nan(nRows,1);
    
    row = 0;   % running index across all (group, rep) pairs
    for k = 1:nG
        group = DCgroups{k};
        nReps = numel(traces.(group).rep);          % CHANGED: was "1:traces.(group).rep" (empty loop)
        for rep_idx = 1:nReps
            r = traces.(group).rep(rep_idx);
            if isempty(r.X) || isempty(r.Y)          % guard: a rep missing its C3 or C4 file
                continue
            end
            t = r.t(1:step:end);
            X = r.X(1:step:end);
            Y = r.Y(1:step:end);
            gamma = atan2(Y,X);
    
            h = histogram(ax, gamma./pi, ...
                DisplayName = sprintf('%s (rep %d)', group, rep_idx), ...   % CHANGED: was just group
                Normalization = "probability", BinWidth = 0.025);
    % pause(1)
            mu    = angle(mean(exp(1i*gamma)));               % circular mean
            dev   = angle(exp(1i*(gamma - mu)));               % deviations, wrapped to (-pi,pi]
            sigma = sqrt(sum(dev.^2)/(numel(gamma)-1));         % circular std about mu
    
            row = row + 1;                                      % NEW: one row per repeat, not per group
            DC_V(row)                = traces.(group).DC;
            rep_num(row)             = rep_idx;                 % NEW
            N_pts(row)               = numel(gamma);
            gamma_mean_rad(row)      = mu/pi;
            gamma_mean_rad_pred(row) = traces.(group).DC*theta_dev/V_max;
            gamma_std_rad(row)       = sigma/pi;
            R_mean(row)              = mean(hypot(X,Y));
        end
    end
    
    keep = ~isnan(DC_V);   % drops any skipped/empty rows
    phaseStatsTable = table(DC_V(keep), rep_num(keep), N_pts(keep), gamma_mean_rad(keep), gamma_mean_rad_pred(keep), gamma_std_rad(keep), R_mean(keep), ...
        'VariableNames', {'DC_V','rep','N_pts','gamma_mean_rad (/pi)','gamma_mean_rad pred. (/pi)','gamma_std_rad (/pi)','R_mean'});
    statsTable = sortrows(phaseStatsTable, ["DC_V","rep"], "ascend");

end

function step = compute_ds(config,t_total,numelem)
    % corrections based on LIA BW
    % tau_LIA  % time constant of LIA (s)
    % rolloff_LIA % dB / oct
    [t_wait,~] = LIA_timing(config.tau_LIA,config.rolloff_LIA);
    
    % compute step for independent samples
    % t_total = t(end)-t(1); % (s)
    % dt=t(2)-t(1);
    % target_sample_num = 1e3;
    sample_num = t_total / t_wait;
    step = floor(numelem/sample_num);
    % disp(strcat("To reach N=",num2str(round(target_sample_num))," samples, use total time t=",num2str(round(t_total*target_sample_num/sample_num,3))," [s]"));
    % downsample_step = 1;
end