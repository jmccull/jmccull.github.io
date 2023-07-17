from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
import numpy as np
import matplotlib.pyplot as plt
import io
from math import factorial

#read in the data to visualize (Generate tomographic bins and apply magnitude cuts)
cellids = np.arange(150*75).astype(int)

np.random.seed(seed=107)
binnum=150

spec_cellids = np.random.randint(0,150*75,size=10000)
spec_z = np.random.normal(loc=0.4,scale=0.5, size=10000)
spec_mags_Z = np.random.uniform(low=17.0,high=23.0,size=10000)
abund_KV_22 = np.random.randint(0,10000,size=150*75)
#get medians everywhere:
medians = np.array([np.median(spec_z[spec_cellids==i]) for i in cellids])


def savitzky_golay(y, window_size=21, order=4, deriv=0, rate=1):
    #savgol filter for smoothing

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def get_new_abund_KV(magcut,abund_22= abund_KV_22):
    #perform a pseudo cut, can read from file as well
    cut = np.round(np.random.randint(0,high=10000,size=150*75)*magcut/5)
    cut[cut>abund_22] = abund_22[cut>abund_22] #don't drop below zero, unphysical
    return abund_22 - cut

def get_masks(spec_cellids,abund_cut,zrange_mask,cellids=cellids):
    #get the cell mask and the spectroscopic mask for a given range of median zs and wide field abundances
    all_cells = np.unique(spec_cellids)
    cell_env = all_cells[np.in1d(all_cells,cellids[(abund_cut > 0)&(zrange_mask)])]
    spec_mask = np.in1d(spec_cellids,cell_env)
    return cell_env, spec_mask

def get_weights(spec_cellids,abund_cut):
    #get weights to match the wide field distribution
    cells,abund_spec = np.unique(spec_cellids,return_counts=True)
    abund = np.zeros(150*75)
    for i,cell in enumerate(cells):
        abund[cell] = abund_spec[i]

    weights_som = abund_cut.astype(float)/abund
    weights = weights_som[spec_cellids]
    return weights

def do_all_calcs_fixed(med,magcut,spec_mag_cut,spec_z=spec_z,spec_mags_Z=spec_mags_Z,
                       spec_cellids=spec_cellids,abund_KV_22=abund_KV_22, medians=medians,cellids=cellids):
    #performs the calculations to get the tomographic bin and the shift

    med_low = med[0]
    med_high = med[1]

    zrange_mask = (medians < med_high) & (medians > med_low)
    abund_cut = get_new_abund_KV(magcut,abund_22=abund_KV_22)

    #apply the mag cut on our source spec to fix the color space
    spec_mask_magcut = spec_mags_Z < spec_mag_cut
    cells_to_cut_to = zrange_mask & np.in1d(cellids,np.unique(spec_cellids[spec_mask_magcut]))
    cell_env,spec_mask = get_masks(spec_cellids,abund_cut,cells_to_cut_to)

    cells_to_cut_to_nomag = zrange_mask
    cell_env, spec_mask_orig = get_masks(spec_cellids, abund_cut, cells_to_cut_to_nomag)
    bins = np.linspace(0,1.8,num=binnum)
    binmids = np.array([(bins[i]+bins[i+1])/2 for i in np.arange(len(bins)-1)])

    weights_cut = get_weights(spec_cellids[spec_mask],abund_cut)
    weights_orig = get_weights(spec_cellids[spec_mask_orig],abund_cut)

    meanz = np.average(spec_z[spec_mask_orig],weights=weights_orig)
    meanz_cut = np.average(spec_z[spec_mask],weights=weights_cut)
    hist_nospecmag, edges = np.histogram(spec_z[spec_mask_orig],bins=bins,weights=weights_orig,density=True)
    hist_speccut, edges = np.histogram(spec_z[spec_mask],bins=bins,weights=weights_cut,density=True)

    return binmids, hist_speccut, hist_nospecmag, meanz_cut, meanz


#user interface
app_ui = ui.page_fluid(
    ui.panel_title('DESI-KiDS-VIKING Redshift Inference'),  # 1
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_checkbox("smooth","Apply Smoothing",value=True),
            ui.input_numeric("npts","SavGol Smoothing Amount", value=21,step=2),
            ui.input_slider("med_z", "Median cell redshift selection for bin", min=0, max=1.7, value=[0.3,0.6], step=0.01, width='100%',drag_range=True),  # 2
            ui.input_slider("magcut", "Wide-field Magnitude Cut, MAG_GAAP_Z", min=15.0, max=22.0, value=22.0, step=0.01, width='100%'),
            ui.input_slider("spec_mag_cut", "Spectroscopic Magnitude Cut, MAG_GAAP_Z", min=15.0, max=23.0, value=22.0, step=0.01, width='100%'),
            ui.download_button("downloadData", "Download Current Figure"),
            ui.download_button("downloadall", "Download All Data"),

width=4, height='50px'),
        ui.panel_main(
            ui.output_plot("p",height='650px'),  # 4
        width=12)
    ))

#server function backend
def server(input: Inputs, output: Outputs, session: Session):
    # 1
    @reactive.Calc
    def get_data():
        binmids, hist_speccut, hist_nospecmag, meanz_cut, meanz = do_all_calcs_fixed(input.med_z(),
                                                                                     input.magcut(),input.spec_mag_cut())
        return binmids, hist_speccut, hist_nospecmag, meanz_cut, meanz

    # 2
    @output
    @render.plot
    def p():
        fig = plt.figure()
        fig.set_size_inches(10,6)
        binmids, hist_speccut, hist_nospecmag, meanz_cut, meanz = get_data()

        #apply simple smoothing
        if input.smooth():
            hist_speccut = savitzky_golay(hist_speccut,window_size=input.npts())
            hist_nospecmag = savitzky_golay(hist_nospecmag,window_size=input.npts())

        #make the figure
        plt.plot(binmids,hist_speccut,'o--',markersize=1.0,color='red',linewidth=2.0,label='spec-$z < {0:.1f}$'.format(input.spec_mag_cut()))
        plt.plot(binmids,hist_nospecmag,'o-',markersize=1.0,color='black',linewidth=2.0,label='no spec-$z$ cut')
        plt.axvline(x = meanz_cut,linestyle='dashed',color='red',label=r'mean(z) spec-cut$ = {0:.4f}$'.format(meanz_cut))
        plt.axvline(x = meanz,linestyle='dashed',color='gray',label=r'mean(z)$ = {0:.4f}$'.format(meanz))

        plt.xlabel(r'Redshift, $z$',fontsize=14)
        plt.ylabel(r'$N(z)$',fontsize=14)
        plt.legend(loc='upper right',fontsize=14)
        return fig

    @session.download(filename='figure.png')
    def downloadData():
        """
        This is the simplest case returninig bytes, duplicates the plotting function to save
        """


        with io.BytesIO() as buf:
            plt.savefig(buf, dpi=300,format="png")
            yield buf.getvalue()


    @session.download()
    def downloadAll():
        """
        This should return the full data set as a collection of bytes
        """

        path = ('../4c3r2_messenger_selection.png') #placeholder
        return path

app = App(app_ui, server)