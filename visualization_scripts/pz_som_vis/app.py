from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle
from pathlib import Path
import matplotlib.transforms as transforms

#follow web app design from https://towardsdev.com/develop-a-web-app-in-10min-and-deploy-it-for-free-3c636b2732c7

#read in the data to visualize (start with redshift distributions)

def format_cell_props(unqcells,prop,shape):
    #formats the cell properties into an array the shape of the som
    output = np.ones(shape[0] * shape[1]) * np.NaN
    for i, c in enumerate(unqcells):
        output[c] = prop[i]

    return np.reshape(output, shape)
path = Path(__file__).parent
with open(path / 'zdictfile.pkl','rb') as f:
    som = pickle.load(f)
#som = np.load('zdictfile.npy') #{i:np.random.rand(11) for i in np.arange(150*75)}
medians = np.array([np.median(som[i]) for i in som.keys()])
medians = format_cell_props(som.keys(),medians,(150,75))
wvlength = np.load(path / 'redrock_wavelengths_lowsample.npy')
spectra = np.load(path / 'mean_redrock_fits_lowsample.npy')
cellids = np.arange(150*75).reshape((150,75)).astype(int)


#user interface
app_ui = ui.page_fluid(
    ui.panel_title('DESI-KiDS-VIKING Redshift Distributions'),  # 1
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_slider("column", "cell column", min=0, max=74, value=55, width='100%'),  # 2
            ui.input_slider("row", "cell row", min=0, max=149, value=61, width='100%'),  # 3
            ui.download_button("downloadData", "Download Current Figure"),
            #ui.download_button("downloadall", "Download All Data"),

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
        try:
            zdist = som[cellid]
        except:
            zdist = np.NaN
        specz = spectra[cellid, :]
        return cellid, zdist, specz

    # 2
    @output
    @render.plot
    def p():
        fig,ax = plt.subplot_mosaic('ABB;ACC')
        #fig.set_size_inches((10,10))
        cellid, zdist, specz = get_som_data()

        g = ax['A'].imshow(medians, cmap='nipy_spectral',origin='lower',interpolation='nearest')
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed',linewidth=3,color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A'], shrink=0.75).set_label(label=r'Median $z$',size=20)

        ax['B'].hist(zdist, bins=150, range=(0, 2), label=None)
        ax['B'].axvline(x=medians.flatten()[cellid],linestyle='dashed', color='black', label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=14)
        ax['B'].set_xlabel(r'$z$', fontsize=14)
        ax['B'].set_ylabel(r'$p(z|$cell$)$', fontsize=14)

        ax['C'].axvline(x=3727,linestyle='dashed', color='gray') #OII
        ax['C'].axvline(x=6562.8, linestyle='dashed', color='gray')  # H-alpha
        ax['C'].axvline(x=4861, linestyle='dashed', color='gray')  # H-beta
        ax['C'].axvline(x=3969, linestyle='dashed', color='gray')  # Ca- H
        ax['C'].axvline(x=3933.7, linestyle='dashed', color='gray')  # Ca- K
        #ax['C'].axvline(x=4959, linestyle='dashed', color='gray')  # OIII
        #ax['C'].axvline(x=5007, linestyle='dashed', color='gray')  # OIII
        trans = transforms.blended_transform_factory(ax['C'].transData, ax['C'].transAxes)
        ax['C'].text(3727-100,0.9,"OII",color='gray',fontsize=12,transform=trans)
        ax['C'].text(6562.8-140,0.9,r"H-$\alpha$",color='gray',fontsize=12,transform=trans)
        ax['C'].text(4861-140,0.9,r"H-$\beta$",color='gray',fontsize=12,transform=trans)

        ax['C'].text(3969+10,0.05,"H",color='gray',fontsize=12,transform=trans)
        ax['C'].text(3933.7-120,0.05,"K",color='gray',fontsize=12,transform=trans)
        #ax['C'].text(5007 - 150, 0.9, "OIII", color='gray', fontsize=12, transform=trans)

        ax['C'].plot(wvlength, specz, 'o-', markersize=0.9, color='blue')
        ax['C'].axvspan(np.min(wvlength), 3600 / (1 + medians.flatten()[cellid]), alpha=0.5, color='lightgray')
        ax['C'].axvspan(9800 / (1 + medians.flatten()[cellid]), np.max(wvlength), alpha=0.5, color='lightgray')
        ax['C'].set_xlabel(r'Rest frame $\lambda$',fontsize=14)
        ax['C'].set_ylabel('Normalized Mean Redrock Fit',fontsize=14)
        ax['C'].set_ylim([np.nanmin([-0.002,-0.3*np.nanmax(specz)]),np.nanmin([0.05,np.nanmax(specz)+0.001])])

        return fig

    @session.download(filename='figure.png')
    def downloadData():
        """
        This is the simplest case returninig bytes, duplicates the plotting function to save
        """
        fig, ax = plt.subplot_mosaic('ABB;ACC')
        fig.set_size_inches((22, 10))
        # fig.set_size_inches((10,10))
        cellid, zdist, specz = get_som_data()

        g = ax['A'].imshow(medians, cmap='nipy_spectral', origin='lower', interpolation='nearest')
        ax['A'].xaxis.set_ticklabels([])
        ax['A'].yaxis.set_ticklabels([])
        ax['A'].axvline(x=input.column(), linestyle='dashed', linewidth=3, color='black')
        ax['A'].axhline(y=input.row(), linestyle='dashed', linewidth=3, color='black')
        plt.colorbar(g, ax=ax['A']).set_label(label=r'Median $z$', size=18)

        ax['B'].hist(zdist, bins=150, range=(0, 2), label=None)
        ax['B'].axvline(x=medians.flatten()[cellid], linestyle='dashed', color='black',
                        label='median: {0:.3f}'.format(medians.flatten()[cellid]))
        ax['B'].legend(loc='upper right', fontsize=18)
        ax['B'].set_xlabel(r'$z$', fontsize=18)
        ax['B'].set_ylabel(r'$p(z|$cell$)$', fontsize=18)

        ax['C'].axvline(x=3727, linestyle='dashed', color='gray')  # OII
        ax['C'].axvline(x=6562.8, linestyle='dashed', color='gray')  # H-alpha
        ax['C'].axvline(x=4861, linestyle='dashed', color='gray')  # H-beta
        ax['C'].axvline(x=3969, linestyle='dashed', color='gray')  # Ca- H
        ax['C'].axvline(x=3933.7, linestyle='dashed', color='gray')  # Ca- K
        # ax['C'].axvline(x=4959, linestyle='dashed', color='gray')  # OIII
        # ax['C'].axvline(x=5007, linestyle='dashed', color='gray')  # OIII
        trans = transforms.blended_transform_factory(ax['C'].transData, ax['C'].transAxes)
        ax['C'].text(3727 - 100, 0.9, "OII", color='gray', fontsize=12, transform=trans)
        ax['C'].text(6562.8 - 140, 0.9, r"H-$\alpha$", color='gray', fontsize=12, transform=trans)
        ax['C'].text(4861 - 140, 0.9, r"H-$\beta$", color='gray', fontsize=12, transform=trans)

        ax['C'].text(3969 + 10, 0.05, "H", color='gray', fontsize=12, transform=trans)
        ax['C'].text(3933.7 - 120, 0.05, "K", color='gray', fontsize=12, transform=trans)

        ax['C'].plot(wvlength, specz, 'o-', markersize=0.9, color='blue')
        ax['C'].set_xlabel(r'Rest frame $\lambda$', fontsize=18)
        ax['C'].set_ylabel('Normalized Mean Redrock Fit', fontsize=18)
        ax['C'].axvspan(np.min(wvlength),3600/(1+medians.flatten()[cellid]),alpha=0.5,color='lightgray')
        ax['C'].axvspan(9800/(1+medians.flatten()[cellid]),np.max(wvlength),alpha=0.5,color='lightgray')
        ax['C'].set_ylim([np.nanmin([-0.002,-0.3*np.nanmax(specz)]),np.nanmin([0.05,np.nanmax(specz)+0.001])])
        plt.tight_layout()

        with io.BytesIO() as buf:
            plt.savefig(buf, dpi=300,format="png")
            yield buf.getvalue()


    # @session.download(filename='DC3R2_data.pkl')
    # def downloadall():
    #     """
    #     This should return the full data set as a collection of bytes
    #     """
    #     with io.BytesIO() as buf:
    #         outputs = {'zdict':som,'wavelength':wvlength,'redrock_fit':spectra}
    #         pickle.dump(outputs,buf)
    #         print(buf)
    #         yield buf.getvalue()

app = App(app_ui, server)