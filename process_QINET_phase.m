clear variables
close all
clc
format long

%% Find all scope CSV files


% set 1
rootp = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/calibration data/V_PM_mapping_10-1/philo=0/";   % change if necessary
files = dir(fullfile(rootp, ...
"SDS5104X_HD_CSV_DC*V_C*_*.csv"));

% measurement config

config = struct();
config.tau_LIA = 10e-3; % s
config.rolloff_LIA=12;  % dB / oct
config.files=files;
config.cscheme = 'cool'; config.crange = [0.4 0.9]; config.tint = 0.1;
config.theta_dev = 2*pi; % set on PM input
config.V_max = 6; % PM max voltage for full theta_dev swing
config.label = '\phi_{LO} = +90°';
config.legendMode = 'all';

[t_sample,V_sample] = readSiglentCSV(fullfile(config.files(1).folder,config.files(1).name));
config.step = compute_ds(config,t_sample(end)-t_sample(1),numel(V_sample));

% % set 1
% rootp = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/9-29-Phase/phiLO=+33/";   % change if necessary
% files = dir(fullfile(rootp, ...
% "SDS5104X_HD_CSV_DC*V_C*_*.csv"));
% 
% % measuremnt config
% config_33 = struct();
% config_33.tau_LIA = 10e-3; % s
% config_33.rolloff_LIA=12;  % dB / oct
% config_33.files=files;
% config_33.cscheme = 'cool'; config_33.crange = [0.4 0.9]; config_33.tint = 0.1;
% config_33.theta_dev = 2*pi; % set on PM input
% config_33.V_max = 6; % PM max voltage for full theta_dev swing
% config_33.label = '\phi_{LO} = +33°';
% 
% 

% 
% % set 2
% rootp = "/Users/agentatom/Library/CloudStorage/OneDrive-Umich/GraduateSchool/UM/QE_LAB/QINET/data/9-29-Phase/phiLO=-77/";   % change if necessary
% files = dir(fullfile(rootp, ...
% "SDS5104X_HD_CSV_DC*V_C*_*.csv"));
% config_77 = config_33; % copy struct
% config_77.files=files;
% config_77.cscheme = 'hot'; config_77.crange = [0.4 0.9]; config_77.tint = 0.1;
% config_77.label = '\phi_{LO} = -77°';



%% Read and plot
figure(1); theme light; hold on; ax=gca;
% stats_33 = read_and_plot(config,ax);
% stats_77 = read_and_plot(config_77,ax);
stats = read_and_plot(config,ax);
disp(stats);

% styling
title('Phase Detection: d\gamma/dV Calibration')
xlabel('\gamma/\pi'); ylabel('counts')
% xlim([-0.5 0.5]);
legend();


% Auxillary Plot
figure; hold on; theme light;
scatter(stats.DC_V,stats.("gamma_mean_rad"),"filled",LineWidth=2,DisplayName='Small Signal Mean')
xlabel('V_{PM} [V]'); ylabel('\mu(\gamma_i)');
title('Phase/V_{PM} Calibration; \phi_{LO}=0');

% linear fit
coefficients = polyfit(stats.DC_V, stats.("gamma_mean_rad"), 1);
gfit = polyval(coefficients, stats.DC_V);
plot(stats.DC_V,gfit,'--',DisplayName='linear fit',LineWidth=2);
legend();
slope = coefficients(1);

% annotate figure with slope
dim = [.2 .5 .3 .3];
str = strcat("Extracted slope = ",num2str(round(slope,5))," [rad/V]");
annotation('textbox',dim,'String',str,'FitBoxToText','on');





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

function statsTable = read_and_plot(config,ax)
    % read in all files from directory and add histograms to supplied fig wqith
    % unique color scheme
    traces = struct();
    directory = config.files;

    % 'dataset' : one legend entry for the whole dataset (default)
    % 'all'     : one entry per histogram (DC group + rep)
    % 'both'    : the dataset entry, followed by its individual histograms
    if ~isfield(config, 'legendMode'), config.legendMode = 'dataset'; end
    showEach    = any(strcmp(config.legendMode, {'all', 'both'}));
    showDataset = any(strcmp(config.legendMode, {'dataset', 'both'}));

    for k = 1:length(directory)
    
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
    
    % sort by DC voltage
    DCgroups = fieldnames(traces);
    [~, order] = sort(cellfun(@(g) traces.(g).DC, DCgroups));   % ascending DC_V
    DCgroups = DCgroups(order);
    nG = numel(DCgroups);
    
    % total rows = sum of repeats across all non-skipped groups
    nRows = 0;
    for k = 1:nG
        nRows = nRows + numel(traces.(DCgroups{k}).rep);
    end

    % sample colormap
    cmap = makeColors(config.cscheme, nRows, config.crange, config.tint);

    if showDataset
    patch(ax, NaN, NaN, cmap(ceil(end/2), :), ...
        EdgeColor = 'none', FaceAlpha = 0.85, DisplayName = config.label);
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
            t = r.t(1:config.step:end);
            X = r.X(1:config.step:end);
            Y = r.Y(1:config.step:end);
            gamma = atan2(Y,X)
    
            h = histogram(ax, gamma./pi, ...
                DisplayName   = sprintf('%s: DC %g V, rep %d', config.label, traces.(group).DC, rep_idx), ...
                Normalization = "probability", BinWidth = 0.00025, ...
                FaceColor     = cmap(row+1, :), EdgeColor = cmap(row+1, :));
            if ~showEach
                h.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end

    % pause(1)
            mu    = angle(mean(exp(1i*gamma)));              % circular mean
            dev   = angle(exp(1i*(gamma - mu)));               % deviations, wrapped to (-pi,pi]
            sigma = sqrt(sum(dev.^2)/(numel(gamma)-1));         % circular std about mu
    
            row = row + 1;                                      % NEW: one row per repeat, not per group
            DC_V(row)                = traces.(group).DC;
            rep_num(row)             = rep_idx;                 % NEW
            N_pts(row)               = numel(gamma);
            gamma_mean_rad(row)      = mu/pi;
            gamma_mean_rad_pred(row) = traces.(group).DC*config.theta_dev/config.V_max;
            gamma_std_rad(row)       = sigma/pi;
            R_mean(row)              = mean(hypot(X,Y));
        end
    end
    
    keep = ~isnan(DC_V);   % drops any skipped/empty rows

    phaseStatsTable = table(DC_V(keep), rep_num(keep), N_pts(keep), gamma_mean_rad(keep), gamma_mean_rad_pred(keep), gamma_std_rad(keep), R_mean(keep), ...
        'VariableNames', {'DC_V','rep','N_pts','gamma_mean_rad','gamma_mean_rad pred.','gamma_std_rad','R_mean'});
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

function cmap = makeColors(cscheme, n, crange, tint)
% cscheme : colormap name ('hot','cool',...) OR one color ([r g b], '#D55E00', 'red')
% crange  : [lo hi] part of the scheme to use. Reverse it ([hi lo]) to flip the order.
%           Colormap: 0 = start of map, 1 = end.
%           Fixed color: 0 = white, 1 = the color itself, 2 = black.
% tint    : 0-1, blend everything toward white (0 = no change)
    if nargin < 3, crange = [0 1]; end
    if nargin < 4, tint = 0; end
    t = linspace(crange(1), crange(2), max(n,1)).';

    isColor = isnumeric(cscheme) || startsWith(string(cscheme), "#");
    if ~isColor
        try
            base = feval(char(cscheme), 256);                 % colormap name
            cmap = interp1(linspace(0,1,256), base, t);
        catch
            isColor = true;                                   % e.g. 'red'
        end
    end
    if isColor
        c = validatecolor(cscheme);
        cmap = (1 - t).*[1 1 1] + t.*c;                       % white -> color
        cmap(t > 1, :) = (2 - t(t > 1)).*c;                   % color -> black
    end
    cmap = cmap + tint*(1 - cmap);                            % lighten
end