function [ax, h] = overlay_figs(specs, opts)
%OVERLAY_FIGS  Plot data from several .fig files on one axes, one colormap per file.
%
%   overlay_figs({'fig1.fig', 'cool', 'Sample A'
%                 'fig2.fig', 'hot',  'Sample B'})
%
%   Each row is {file, colormap, category label}. The colormap can be any
%   colormap name ('cool', 'hot', 'winter', 'autumn', 'parula', ...) or an
%   N-by-3 matrix. Within a file, series get evenly spaced colors from its
%   colormap; series that originally shared a color (e.g. data + its fit)
%   still share one. Black/gray objects are left alone.
%
%   Name-value options:
%     ColorRange      [0.1 0.8]   slice of each colormap to use (avoids near-white ends)
%     Legend          "category"  "category" (one entry per file) | "series" | "none"
%     LegendLocation  "best"
%     XLabel, YLabel  default: taken from the first file
%     Title           ""
%     LineWidth       1.25        minimum line width
%     FontSize        10
%     Export          ""          e.g. "combined.pdf" or "combined.png"
%
%   [ax, h] = overlay_figs(...) also returns the axes and a struct array with
%   h(k).label, h(k).handles, h(k).colors so you can tweak things afterwards.
%
%   Example:
%     [ax, h] = overlay_figs({'run1.fig', 'cool', '4 K'
%                             'run2.fig', 'hot',  '300 K'}, ...
%                            'YLabel', 'Counts', 'Export', "combined.pdf");
%     set(h(2).handles, 'LineStyle', '--')   % e.g. dash everything from run2

arguments
    specs cell
    opts.ColorRange (1,2) double = [0.1 0.8]
    opts.Legend (1,1) string {mustBeMember(opts.Legend, ["category", "series", "none"])} = "category"
    opts.LegendLocation (1,1) string = "best"
    opts.XLabel = []
    opts.YLabel = []
    opts.Title = ""
    opts.LineWidth (1,1) double = 1.25
    opts.FontSize (1,1) double = 10
    opts.Export (1,1) string = ""
end

if size(specs, 2) ~= 3, specs = reshape(specs.', 3, []).'; end   % allow a flat list
plotTypes = {'line', 'errorbar', 'scatter', 'stair', 'bar', 'area'};

fig = figure('Color', 'w');
ax  = axes(fig);
hold(ax, 'on');
h = struct('label', specs(:, 3), 'handles', [], 'colors', []);

for k = 1:size(specs, 1)
    % --- pull the plotted objects out of every axes in this file
    src  = openfig(char(specs{k, 1}), 'invisible');
    sAx  = flipud(findobj(src, 'Type', 'axes'));
    objs = gobjects(0);
    for a = sAx.'
        kids = flipud(a.Children);
        objs = [objs; kids(arrayfun(@(o) any(strcmp(o.Type, plotTypes)), kids))]; %#ok<AGROW>
    end
    if k == 1   % axis labels and scales come from the first file
        m = sAx(1);
        set(ax, 'XScale', m.XScale, 'YScale', m.YScale);
        xlabel(ax, m.XLabel.String, 'Interpreter', m.XLabel.Interpreter);
        ylabel(ax, m.YLabel.String, 'Interpreter', m.YLabel.Interpreter);
    end
    if isempty(objs)
        warning('overlay_figs:empty', 'No plotted data found in %s.', specs{k, 1});
        delete(src); continue
    end
    new = copyobj(objs, ax);
    delete(src);

    % --- one colormap color per distinct original color
    cols = cell2mat(arrayfun(@mainColor, new, 'UniformOutput', false));
    use  = ~any(isnan(cols), 2) & ~isNeutral(cols);
    [~, ~, grp] = unique(cols(use, :), 'rows', 'stable');
    n = max([grp; 0]);
    if n <= 1, t = mean(opts.ColorRange); else, t = linspace(opts.ColorRange(1), opts.ColorRange(2), n); end
    cmap    = getCmap(specs{k, 2});
    newCols = interp1(linspace(0, 1, size(cmap, 1)), cmap, t(:));

    idx = find(use);
    for j = 1:numel(idx)
        recolor(new(idx(j)), newCols(grp(j), :), opts.LineWidth);
    end
    if opts.Legend == "series"
        for o = new.'
            o.DisplayName = strtrim(sprintf('%s %s', string(h(k).label), o.DisplayName));
        end
    end
    h(k).handles = new;
    h(k).colors  = newCols;
end

% --- labels, style, legend, export
if ~isempty(opts.XLabel), xlabel(ax, opts.XLabel); end
if ~isempty(opts.YLabel), ylabel(ax, opts.YLabel); end
title(ax, opts.Title);
set(ax, 'FontSize', opts.FontSize, 'TickDir', 'out', 'LineWidth', 0.75, 'Box', 'on', 'Layer', 'top');

switch opts.Legend
    case "category"   % one color swatch per file, using a color from its scheme
        p = gobjects(numel(h), 1);
        for k = 1:numel(h)
            c = h(k).colors;
            if isempty(c), c = [0.5 0.5 0.5]; else, c = c(ceil(end / 2), :); end
            p(k) = plot(ax, NaN, NaN, 's', 'MarkerSize', 9, ...
                'MarkerFaceColor', c, 'MarkerEdgeColor', 'none');
        end
        legend(ax, p, cellstr(string(specs(:, 3))), 'Location', opts.LegendLocation, 'Box', 'off');
    case "series"
        legend(ax, vertcat(h.handles), 'Location', opts.LegendLocation, 'Box', 'off');
end

if strlength(opts.Export) > 0
    [~, ~, ext] = fileparts(opts.Export);
    if any(strcmpi(ext, [".pdf", ".eps"])), args = {'ContentType', 'vector'};
    else, args = {'Resolution', 600}; end
    exportgraphics(fig, opts.Export, args{:});
end
end


% ---------------------------------------------------------------------------
function c = mainColor(o)
% The object's single RGB color, or NaNs if it doesn't have one (e.g. colormapped scatter).
    c = nan(1, 3);
    for p = {'Color', 'CData', 'FaceColor'}
        if isprop(o, p{1})
            v = o.(p{1});
            if isnumeric(v) && numel(v) == 3, c = double(reshape(v, 1, 3)); return, end
        end
    end
end

function tf = isNeutral(c)
    tf = (max(c, [], 2) - min(c, [], 2)) < 0.03;   % black, white, grays
end

function cmap = getCmap(c)
    if isnumeric(c), cmap = c; else, cmap = feval(char(c), 256); end
end

function recolor(o, c, minLW)
    for p = {'Color', 'MarkerFaceColor', 'MarkerEdgeColor', 'FaceColor', 'EdgeColor', 'CData'}
        if isprop(o, p{1})
            v = o.(p{1});
            if isnumeric(v) && numel(v) == 3 && ~isNeutral(double(reshape(v, 1, 3)))
                o.(p{1}) = c;
            end
        end
    end
    if any(strcmp(o.Type, {'line', 'errorbar', 'stair'})) && ~strcmp(o.LineStyle, 'none')
        o.LineWidth = max(o.LineWidth, minLW);
    end
end
