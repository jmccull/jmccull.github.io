from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import io
from pathlib import Path

#follow web app design from https://towardsdev.com/develop-a-web-app-in-10min-and-deploy-it-for-free-3c636b2732c7

#read in the data to visualize (start with redshift distributions)

def format_cell_props(unqcells,prop,shape):
    #formats the cell properties into an array the shape of the som
    output = np.ones(shape[0] * shape[1]) * np.NaN
    for i, c in enumerate(unqcells):
        output[c] = prop[i]

    return np.reshape(output, shape)

path = Path(__file__).parent
infile = np.load(path / 'exptimes_mags_bgs_exclude.npy')
print(infile.shape)
succ = infile[(infile[:,3]==0)]
print(succ.shape)
fail = infile[(infile[:,3]==1)]
print(fail.shape)
som_succ = {i:succ[(succ[:,0].astype(int)==i)][:,2] for i in np.arange(150*75)}
som_fail = {i:fail[(fail[:,0].astype(int)==i)][:,2] for i in np.arange(150*75)}
print((som_succ[61*75+55]))

medians = np.array([np.nanmedian(np.hstack((som_succ[i],som_fail[i]))) for i in som_succ.keys()])
#medians = np.array([np.nanmedian(som_succ[i]) for i in som_succ.keys()])

medians = format_cell_props(som_succ.keys(),medians,(150,75))
print(medians)
cellids = np.arange(150*75).reshape((150,75)).astype(int)

#make a new colormap
cmap1 = plt.cm.get_cmap('gnuplot_r', 100)
cmap2 = plt.cm.get_cmap('turbo', 100)
newcolors = cmap1(np.linspace(0, 1, 1000)*0.8)[::-1,:]
newcolors[:,1] = newcolors[:,1]
newcolors[:,0] = newcolors[:,0]
grn = np.array([0/256, 100/256, 0/256, 1])
length = 435
newcolors[:length, :] = cmap2(0.02+np.linspace(0,1,length)*0.2)[:,:]
newcolors[:length, 1] = newcolors[:length,1] - 0.12
newcmp = colors.ListedColormap(newcolors)

#user interface
app_ui = ui.page_fluid(
    ui.panel_title('DESI-KiDS-VIKING Exposure Time Distributions'),  # 1
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider("column", "cell column", min=0, max=74, value=55, width='100%'),  # 2
            ui.input_slider("row", "cell row", min=0, max=149, value=61, width='100%'),  # 3
            ui.download_button("downloadData", "Download Current Figure"),

width=4, height='60px'),
        ui.panel_main(
            ui.output_plot("p",height='650px'),  # 4
        width=12)
    ))

#server function backend
def server(input: Inputs, output: Outputs, session: Session):
    # 1
    @reactive.Calc
    def get_som_data():
        cellid = cellids[int(input.row()), int(input.column())]

        exptimes_succ = som_succ[cellid]
        exptimes_fail = som_fail[cellid]
        mask = (infile[:,0].astype(int) == cellid)
        mags = infile[mask][:,1]
        print(mags)
        return cellid, exptimes_succ, exptimes_fail, mags

    # 2
    @output
    @render.plot
    def p():
        fig,ax = plt.subplot_mosaic('ABB;ACC')
        #fig.set_size_inches((10,10))
        cellid, exptimes_succ, exptimes_fail, mags = get_som_data()

        g = ax['A'].imshow(medians, cmap=newcmp,origin='lower',interpolation='nearest',norm=colors.LogNorm(vmin=0.1,vmax=20))
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed',linewidth=3,color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A'], shrink=0.75).set_label(label=r'Median Exposure Time Required (min)',size=20)

        ax['B'].hist(exptimes_succ, bins=50, range=(0,100), alpha=0.5, label=r'Successful $z$')
        ax['B'].hist(exptimes_fail, bins=50, range=(0, 100), alpha=0.5, label=r'Unsuccessful $z$')
        ax['B'].axvline(x=medians.flatten()[cellid],linestyle='dashed', color='gray', label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=14)
        ax['B'].set_xlabel(r'Exposure Times (Scaled to $\Delta \chi^2 = 40$, $Z_{fiber} = 21.0$)', fontsize=14)
        ax['B'].set_ylabel(r'$p(exptime|$cell$)$', fontsize=14)

        hists, edges = np.histogram(mags,bins=20,range=(18,23))
        binmids = np.array([(edges[i]+edges[i+1])/2 for i in np.arange(len(edges)-1)])
        ax['C'].hist(binmids, weights=hists,bins=edges,histtype='stepfilled', color='blue',alpha=0.5)
        ax['C'].set_xlabel(r'MAG_GAAP_Z',fontsize=14)
        ax['C'].set_ylabel('N(mag)',fontsize=14)

        return fig

    @session.download(filename='figure.png')
    def downloadData():
        """
        This is the simplest case returninig bytes, duplicates the plotting function to save
        """
        fig,ax = plt.subplot_mosaic('ABB;ACC')
        fig.set_size_inches((22,10))
        cellid, exptimes_succ, exptimes_fail, mags = get_som_data()

        g = ax['A'].imshow(medians, cmap=newcmp,origin='lower',interpolation='nearest',norm=colors.LogNorm(vmin=0.1,vmax=20))
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed',linewidth=3,color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A'], shrink=0.75).set_label(label=r'Median Exposure Time Required (min)',size=20)

        ax['B'].hist(exptimes_succ, bins=50, range=(0,100), alpha=0.5, label=r'Successful $z$')
        ax['B'].hist(exptimes_fail, bins=50, range=(0, 100), alpha=0.5, label=r'Unsuccessful $z$')
        ax['B'].axvline(x=medians.flatten()[cellid],linestyle='dashed', color='gray', label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=14)
        ax['B'].set_xlabel(r'Exposure Times (Scaled to $\Delta \chi^2 = 40$, $Z_{fiber} = 21.0$)', fontsize=14)
        ax['B'].set_ylabel(r'$p(exptime|$cell$)$', fontsize=14)

        hists, edges = np.histogram(mags,bins=20,range=(18,23))
        binmids = np.array([(edges[i]+edges[i+1])/2 for i in np.arange(len(edges)-1)])
        ax['C'].hist(binmids, weights=hists,bins=edges,histtype='stepfilled', color='blue',alpha=0.5)
        ax['C'].set_xlabel(r'MAG_GAAP_Z',fontsize=14)
        ax['C'].set_ylabel('N(mag)',fontsize=14)
        plt.tight_layout()

        with io.BytesIO() as buf:
            plt.savefig(buf, dpi=300,format="png")
            yield buf.getvalue()


app = App(app_ui, server)