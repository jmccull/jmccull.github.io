from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
import numpy as np
import matplotlib.pyplot as plt
import io

#follow web app design from https://towardsdev.com/develop-a-web-app-in-10min-and-deploy-it-for-free-3c636b2732c7

#read in the data to visualize (start with redshift distributions)

def format_cell_props(unqcells,prop,shape):
    #formats the cell properties into an array the shape of the som
    output = np.ones(shape[0] * shape[1]) * np.NaN
    for i, c in enumerate(unqcells):
        output[c] = prop[i]

    return np.reshape(output, shape)

som_succ = {i:np.random.rand(11) for i in np.arange(150*75)} #np.load('exposure_time_dict.npy')
som_fail = {i:np.random.rand(11) for i in np.arange(150*75)} #np.load('exposure_time_dict.npy')

medians = np.array([np.median(np.hstack((som_succ[i],som_fail[i]))) for i in som_succ.keys()])
medians = format_cell_props(som_succ.keys(),medians,(150,75))
mags = np.random.rand(150*75*11).reshape((150*75,11)) #Z-band histogram stored like spec
magbins = np.linspace(15,23,num=11) #read in mag bins
cellids = np.arange(150*75).reshape((150,75)).astype(int)


#user interface
app_ui = ui.page_fluid(
    ui.panel_title('DESI-KiDS-VIKING Exposure Time Distributions'),  # 1
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider("column", "cell column", min=0, max=74, value=30, width='100%'),  # 2
            ui.input_slider("row", "cell row", min=0, max=149, value=30, width='100%'),  # 3
            ui.download_button("downloadData", "Download Current Figure"),
            ui.download_button("downloadall", "Download All Data"),

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
        mag_dist = mags[cellid, :]
        return cellid, exptimes_succ, exptimes_fail, mag_dist

    # 2
    @output
    @render.plot
    def p():
        fig,ax = plt.subplot_mosaic('ABB;ACC')
        #fig.set_size_inches((10,10))
        cellid, exptimes_succ, exptimes_fail, mag_dist = get_som_data()

        g = ax['A'].imshow(medians, cmap='hot',origin='lower',interpolation='nearest')
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed',linewidth=3,color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A'], shrink=0.75).set_label(label=r'Median Scaled Exposure Time (min)',size=20)

        ax['B'].hist(exptimes_succ, bins=150, range=(0, 2), alpha=0.5, label=r'Successful $z$')
        ax['B'].hist(exptimes_fail, bins=150, range=(0, 2), alpha=0.5, label=r'Unsuccessful $z$')
        ax['B'].axvline(x=medians.flatten()[cellid],linestyle='dashed', color='gray', label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=14)
        ax['B'].set_xlabel(r'Exposure Times (Scaled to $\Delta \chi^2 = 40$, $Z_{fiber} = 21.0$)', fontsize=14)
        ax['B'].set_ylabel(r'$p(exptime|$cell$)$', fontsize=14)

        ax['C'].fill_between(magbins, mag_dist, color='blue',alpha=0.5)
        ax['C'].set_xlabel(r'MAG_GAAP_Z',fontsize=14)
        ax['C'].set_ylabel('N(mag)',fontsize=14)

        return fig

    @session.download(filename='figure.png')
    def downloadData():
        """
        This is the simplest case returninig bytes, duplicates the plotting function to save
        """
        fig, ax = plt.subplot_mosaic('ABB;ACC')
        fig.set_size_inches((22, 10))
        cellid, exptimes_succ, exptimes_fail, mag_dist = get_som_data()

        g = ax['A'].imshow(medians, cmap='hot', origin='lower', interpolation='nearest')
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed', linewidth=3, color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A'], shrink=0.75).set_label(label=r'Median Scaled Exposure Time (min)', size=20)

        ax['B'].hist(exptimes_succ, bins=150, range=(0, 2), alpha=0.5, label=r'Successful $z$')
        ax['B'].hist(exptimes_fail, bins=150, range=(0, 2), alpha=0.5, label=r'Unsuccessful $z$')
        ax['B'].axvline(x=medians.flatten()[cellid], linestyle='dashed', color='gray',
                        label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=14)
        ax['B'].set_xlabel(r'Exposure Times (Scaled to $\Delta \chi^2 = 40$, $Z_{fiber} = 21.0$)', fontsize=14)
        ax['B'].set_ylabel(r'$p(exptime|$cell$)$', fontsize=14)

        ax['C'].fill_between(magbins, mag_dist, color='blue', alpha=0.5)
        ax['C'].set_xlabel(r'MAG_GAAP_Z', fontsize=14)
        ax['C'].set_ylabel('N(mag)', fontsize=14)
        plt.tight_layout()

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