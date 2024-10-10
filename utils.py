import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


## General

def mpl_settings():
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['axes.linewidth'] = .5 
    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['xtick.major.size'] = 2

def paper_plot():
    nice_plot = {'ytick.labelsize': 16,
                        'xtick.labelsize': 16,
                        'font.size': 14,
                        'axes.titlesize': 22,
                        'axes.titlepad': 20,
                        'axes.labelsize': 16,
                        'lines.linewidth': 2,
                        'lines.markersize': 3,
                        'legend.fontsize': 11,
                        'mathtext.fontset': 'stix',
                        'font.family': 'sans-serif',
                        'font.sans-serif':'DejaVu Sans',
                        'font.monospace':'DejaVu Sans'}
    plt.style.use(nice_plot)



def intround(vals,decimals=2):
    new = []
    for num in np.round(vals,decimals):
        if num.is_integer():
            new.append(int(num))
        elif num > 100:
            new.append(np.nan)
        else:
            new.append(num)
    return new


def penalty_free_rename_dim(ds,new_dim,old_dim=None):
    if old_dim == None:
        old_dim = list(ds.data_vars)[0]

    try:
        ds = ds.rename({old_dim:new_dim})
        print(f'      {old_dim} renamed to {new_dim}')

    except:

        print(f'       nothing to rename in {list(ds.data_vars)[0]}')
    return ds

def get_season_label(time):
    
    month = time.month
    year = time.year
    
    # DJF
    if month == 12:
        year += 1
    season = (
        ('DJF' if month in [12, 1, 2] else
         'MAM' if month in [3, 4, 5] else
         'JJA' if month in [6, 7, 8] else
         'SON'))
    return f'{year}-{season}'

#prepare seasonal timeseries
def prepare_ts(ts,start_month='09',year_s=None, year_e=None):
	#get first and last year of ts
	if year_s == None: year_s = ts.index.year[0]
	if year_e == None: year_e = ts.index.year[-1]

	#remove leap year
	ts = ts[~((ts.index.month == 2) & (ts.index.day == 29))]

	# get start and end date
	start = f'{year_s}-{start_month}-01'
	end_mmdd = str(np.datetime64(start) - np.timedelta64(1,'D'))[5:]
	end = f'{year_e}-{end_mmdd}'
	# set 29 to 28 Feb
	if end[-2:] == '29': end = end[:-1] + '8'

	if len(ts.loc[f'{year_e-1}-{start_month}-01':end]) != 365:
		end = f'{year_e-1}-{end_mmdd}'
	if len(ts.loc[f'{year_s}-{start_month}-01':f'{year_s+1}-{end_mmdd}']) != 365:
		start = f'{year_s+1}-{start_month}-01'
	ts = ts.loc[start:end]
	print('This timeseries is: ',(len(ts)/365).is_integer())
	return ts

## Figs 2 & 3

def generate_col_labels(order, start_index=1):
    labels = []
    for i, subscript in enumerate(order):
        labels.append(f'$\\alpha^{{{start_index + i}}}_{{{subscript}}}$')
    return labels

def rmse_r2_label(predictions, targets, pos='high', c='k'):   
    rmse_ = np.sqrt(((predictions - targets) ** 2).mean())
    slo, _, r_value, _, _ = stats.linregress(predictions, targets)
    r2 = r_value**2
    if np.round(rmse_,1) < 10:
        label = 'RMSE =   {:.1f}, r² = {:.2f}'.format(rmse_,r2)
    else:
        label = 'RMSE = {:.1f}, r² = {:.2f}'.format(rmse_,r2)

    if pos == 'top':
        plt.gca().text(0.02, 0.92,label,color=c,ha='left',va='center',transform = plt.gca().transAxes);
    elif pos == 'upmid':
        plt.gca().text(0.02, 0.87-.02,label,color=c,ha='left',va='center',transform = plt.gca().transAxes);
    elif pos == 'lowmid':
        plt.gca().text(0.02, 0.82-.04,label,color=c,ha='left',va='center',transform = plt.gca().transAxes);
    elif pos == 'bottom':
        plt.gca().text(0.02, 0.77-.06,label,color=c,ha='left',va='center',transform = plt.gca().transAxes);


def polyline(x,y,alpha=1,c='r'):
    m,b = np.polyfit(x,y,1)
    X = np.array([x.min()-20, x.max()+20])
    plt.gca().plot(X,m*X+b,c=c,alpha=alpha,marker=None,linestyle='-.', linewidth=1.5)

def scatter(ax,X,Y,marker,color,position,alpha=1):    
    plt.sca(ax)
    polyline(X,Y,c=color,alpha=.5)
    ax.scatter(X,Y,marker=marker,ec=color,fc='none',alpha=alpha)
    rmse_r2_label(X,Y,pos=position,c=color)

def plot_unity(xdata, ydata, **kwargs):
    import numpy as np
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)  

def color_table(t,colors,cols=6):
    t.properties()['children'][-4].get_text().set_color(colors[0])
    [cell.get_text().set_color(colors[0]) for cell in t.properties()['children'][0:cols]]
    t.properties()['children'][-3].get_text().set_color(colors[1])
    [cell.get_text().set_color(colors[1]) for cell in t.properties()['children'][cols:cols*2]]
    t.properties()['children'][-2].get_text().set_color(colors[2])
    [cell.get_text().set_color(colors[2]) for cell in t.properties()['children'][cols*2:cols*3]]
    t.properties()['children'][-1].get_text().set_color(colors[3])
    [cell.get_text().set_color(colors[3]) for cell in t.properties()['children'][cols*3:cols*4]]

def table(ax,param_list,col_labels,colors=['Grey','#55C667','#C2B130','#440154']):
    row_labels = ['$INIT_{WRF}$','WRF','CHIRPS','AWS']
    row_values = [intround(param_list[i],2) for i in range(len(param_list))]
    t = ax.table(cellText=row_values,colWidths=[0.09] * len(param_list[0]),
                 rowLabels=row_labels,colLabels=col_labels,loc='lower right')
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    color_table(t=t,colors=colors,cols=len(param_list[0]))

def set_plot_params(ax, label_size=14, tick_size=12, text_size=16):
	ax.xaxis.label.set_size(label_size)
	ax.yaxis.label.set_size(label_size)
	ax.tick_params(axis='both', which='major', labelsize=tick_size)
	ax.tick_params(axis='both', which='minor', labelsize=tick_size)
	for text in ax.texts:
	    text.set_fontsize(text_size)
	   

## Fig 4

def nice_reference_point(self):
    self.samplePoints[0].set_color('k')
    self.samplePoints[0].set_markersize(10)  
    self.samplePoints[0].set_zorder(10)
    self.samplePoints[0].set_markeredgecolor('w')

def add_contours(dia,param):
    contours = dia.add_contours(levels=4, colors='0.5',alpha=.5)  
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
    dia.add_grid()                                 
    dia._ax.axis[:].major_ticks.set_tick_out(True)  
    dia._ax.set_title(param)



## Fig 5


def isnumpyarray(*args):
    import xarray as xr
    np_args = []
    for arg in args:
        if isinstance(arg, xr.DataArray):
            np_args.append(arg.values)
        else:
            np_args.append(arg)
    return tuple(np_args)

def min_max_scale(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).ravel()  # back to 1D

def get_bin_widths(data):
    # define bin widths using Freedman-Diaconis rule
    min_val,max_val = data.min().item(),data.max().item()

    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / np.cbrt(data.count())
    n_bins = int((max_val - min_val) / bin_width)

    bin_edges = np.linspace(min_val,max_val,n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers

def calc_pvalue(X,y,y_hat,weights,slope):
     # 1 Residuals
    residuals = y - y_hat
    weighted_residual_sum_of_squares = np.sum(weights * residuals**2) # weighted_residual_sum_of_squares
    dof = X.shape[0] - 2  # degrees of freedom

    # 2 Variance of the estimated slope
    weighted_X_mean = np.average(X, weights=weights) # Weighted average
    s_xx = np.sum(weights * (X.squeeze() - weighted_X_mean)**2) # 
    var_slope = weighted_residual_sum_of_squares / (dof * s_xx) # Variance
    std_err_slope = np.sqrt(var_slope) # Standard Error

    # 3 T-Statistic
    t_stat = slope / std_err_slope
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))
    
    return p_value


def weighted_linregress(bin_centers, y, weights, scale_data=True):
    # xr -> np conversion
    bin_centers,y,weights = isnumpyarray(bin_centers, y, weights)
   
    # Min-Max scaling
    if scale_data:
        bin_centers = min_max_scale(bin_centers)
        y = min_max_scale(y)
        weights = min_max_scale(weights)
    
    # weighted linreg model
    model = LinearRegression()
    mask = ~np.isnan(bin_centers) & ~np.isnan(y)
    model.fit(bin_centers[mask][:, np.newaxis], y[mask], sample_weight=weights[mask])

    slope = model.coef_[0]
    y_hat = model.predict(bin_centers[:, np.newaxis])
    r2 = model.score(bin_centers[mask][:, np.newaxis], y[mask], sample_weight=weights[mask])
    pvalue = calc_pvalue(bin_centers[mask], y[mask], y_hat[mask], weights[mask], slope)

    return slope, r2, pvalue, y_hat


def add_nan_rows(arr, num_rows_to_add=2, position=None):
    rows, cols = arr.shape
    # inserting @ mid if no position is specified
    if position is None:
        position = rows // 2
    nan_rows = np.full((num_rows_to_add, cols), np.nan)
    arr_with_nan = np.insert(arr, position, nan_rows, axis=0)
    
    return arr_with_nan

def get_font_size(value,dmin=0.1,dmax=1):
    normalized_value = (value - dmin) / (dmax - dmin)
    min_font_size = 10
    max_font_size = 19
    font_size = min_font_size + (max_font_size - min_font_size) * normalized_value
    return font_size

def sized_annot(array_map,array_annot,ax):    
    cmap = ax.collections[0].cmap # extract cmap from ax
    norm = ax.collections[0].norm # and normalization func
    for i in range(array_map.shape[0]):
        for j in range(array_map.shape[1]):
            
            # extract values of each array
            text_val = array_annot[i, j]
            cell_clr = cmap(norm(array_map[i, j]))
            # calculate font size
            font_size = get_font_size(text_val) 
            # calculate luminance and text color
            rgb = mcolors.to_rgb(cell_clr)
            # Source: ITU-R BT.601 color encoding recommendation 
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] * 0.114 * rgb[2]
            txt_col = 'w' if luminance < .24 else 'k'
            
            # annotate!
            ax.text(j + 0.5, i + 0.5, f'{text_val:.2f}', color=txt_col,
                    ha='center', va='center', fontsize=font_size)



## Fig 5
def trend_label_past(label,ax,pos='high'):
    if label != '':
        if pos == 'high':
            ax.text(0.02,0.1,'Past:'+label,color='k',ha='left',va='center',transform = ax.transAxes,zorder=10);
        else:
            ax.text(0.04,0.825,'Past:'+p_label2,color='r',ha='left',va='center',transform = ax.transAxes,zorder=10)
            
def trend_label_futu(label,ax,scen='85',c1='#2D9B4E',c2='#B5A100'):
    if label != '':
        if scen == '85':
            ax.text(0.6, .925,'RCP8.5: ' +label,color=c2,ha='left',va='center',transform = ax.transAxes);
        elif scen == '45':
            ax.text(0.6, .825,'RCP4.5: ' +label,color=c1,ha='left',va='center',transform = ax.transAxes);
            
def sig_trend(x,y,sig_lim=.05):
    try:
        slope, _, _, p, _  = scipy.stats.linregress(x, y)
    except:
        slope, _, _, p, _  = stats.linregress(x, y)
      
    if p < sig_lim:
        if p < .01: n = 3
        elif p < .05: n = 2
        elif p > .05: n = 0
        star = '*'
        label = f'slope = {np.round(slope*10,2)}{n*star}'
        return label
    else:
        return ''

def set_plot_params(ax, label_size=14, tick_size=12, text_size=16):
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    
    for text in ax.texts:
        text.set_fontsize(text_size)