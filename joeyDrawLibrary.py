import csv
import math
import matplotlib.pyplot as pylab
from matplotlib import mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as sl
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.cluster.vq as scv
import numpy as np
from matplotlib.mlab import PCA
import numpy.linalg as npla
import string
import time
import sys, os
import getopt
from mpl_toolkits.mplot3d import proj3d
from KMeansCluster import *

def findIndices(g):
    """findIndices is a  helper function that likely should be deleted
    
    """
    change = [0]
    hold = g[0]
    seen = [g[0]]
    for i in range(1, len(g)):
        if not g[i] in seen:
            change.append(i)
            seen.append(g[i])
    return change

def mapGroups(groupList, letters):
    """mapGroups is a  helper function that maps groups numbers to letters - likely should be deleted

    :param groupList: a list to be mapped
    :type d: list
    :param letters: a list to be mapped onto
    :type letters: list
    :returns: a list with elements of groupList mapped onto the letters
    
    """
    changeList = findIndices(groupList)
    i = 0
    for index in changeList:
        toReplace = groupList[index]
        groupList = listReplace(groupList, toReplace, letters[i])
        i = i+1
    return list(groupList)

def listReplace(l, to, rv):
    """listReplace is a helper function replaces all occurances of to with rv in a list l

    :param l: list to be replaced
    :type l: list
    :param to: item to be replaced
    :type to: string
    :param rv: item to replace with
    :type rv: string
    :returns: a list with all occurances of to replaced with rv
    
    """
    tr = []
    for i in l:
        if i == to:
            tr.append(rv)
        else:
            tr.append(i)
    return tr

def printSortedDict(d):
    """printSortedDict is a helper function to print a dictionary

    :param d: a dictionary to be printed
    :type d: dict
    :returns: a string of the dictionary
    
    """
    k = d.keys()
    k.sort()
    tp = ''
    for i in k:
        tp = tp + str(i) + ":" + str(d[str(i)]) + ", "
    return tp

def performAllReconstructions(xdat, U, Sig, Vh, figure=False, colors=pylab.cm.RdBu):
    """performs all possible reconstructions of the dataset that has been factored by SVD

    :param xdata: the data object to be reconstructed/analyzed
    :type data: dictionary, must contain 'data', 'fractions', 'proteins'
    :param U: the 2D matrix U
    :type U: 2D matrix
    :param Sig: the 2D matrix Sig
    :type Sig: 2D matrix
    :param Vh: the 2D matrix Vh
    :type Vh: 2D matrix
    :param colors: the colormap to be used in tieh figure
    :type colors: colormap
    :param figure: whether to display the figures
    :type figure: boolean
    :returns:  array bearing the calculated residual for each additional component

    """
    clusteredData = xdat['data']
    fractions = xdat['fractions']
    proteins = xdat['proteins']
    ydat = xdat.copy()
    zdat = xdat.copy()
    residualsByComp = []
    for i in range(1, len(fractions)+1):
        recon, resid, totalResid = reconstructWithSVD(clusteredData, U, Sig, Vh, i)
        ydat['data'] = recon
        zdat['data'] = resid
        residualsByComp.append(totalResid)
        if figure:
            drawThreePanelHeatMap(xdat, ydat, zdat, n1="originalData", n2="recon_"+str(i)+"_comp", n3="resid")

    if figure:
        fig = pylab.figure()
        ##Draw heatmap
        axData = fig.add_axes([0.1, 0.1, 0.75, 0.75])
        axData.scatter(range(1,len(residualsByComp)+1), residualsByComp)

    return residualsByComp

def plotComponents(U, comps, proteins=None):
    """plots the specified number of components in a new figure

    :param U: the U matrix from the SVD
    :type data: 2D matrix from SVD
    :param comps: the number of components to plot
    :type comps: int
    :param proteins: the protein names, if none will just list numbers
    :type proteins: list
    :returns:  and figure with the ploted heat map

    """
    toPlot = dict()
    toPlot['data'] = U[0:, 0:comps]
    toPlot['fractions'] = range(comps)
    if proteins is None:
        toPlot['proteins'] = range(len(U))
    else:
        toPlot['proteins'] = proteins

    return drawHeatMap(toPlot, name="components")


def drawThreePanelHeatMap(xdat, ydat, zdat, n1="dat1", n2="dat2", n3="dat3", colors=pylab.cm.RdBu):
    """draws a figure with 3 heatmap panels

    :param xdat: first panel data object
    :type xdat: dictionary, must contain 'data', 'fractions', 'proteins'
    :param ydat: first panel data object
    :type ydat: dictionary, must contain 'data', 'fractions', 'proteins'
    :param zdat: first panel data object
    :type zdat: dictionary, must contain 'data', 'fractions', 'proteins'
    :param n1: name of panel 1
    :type n1: string
    :param n2: name of panel 2
    :type n2: string
    :param n3: name of panel 3
    :type n3: string
    :param colors: the colormap to be used in tieh figure
    :type colors: colormap
    :returns:  array bearing the calculated residual for each additional component

    """

    data = xdat['data']
    data2 = ydat['data']
    data3 = zdat['data']

    fig = pylab.figure()
        
    fig.text(0.1, 0.925, n1)
    ##Draw heatmap
    f1Data = heatMapAxes(data, dims=[0.05, 0.1, 0.225, 0.8], fractions=xdat['fractions'], proteins=xdat['proteins'], fig=fig)
    ##Draw colorbar
    fig.colorbar(f1Data)
    
    fig.text(0.4, 0.925, n2)
    ##Draw heatmap
    f2Data = heatMapAxes(data2, dims=[0.375, 0.1, 0.225, 0.8], fractions=ydat['fractions'], proteins=ydat['proteins'], fig=fig)
    ##Draw colorbar
    fig.colorbar(f2Data)

    fig.text(0.7, 0.925, n3)
    ##Draw heatmap
    f3Data = heatMapAxes(data3, dims=[0.7, 0.1, 0.225, 0.8], fractions=zdat['fractions'], proteins=zdat['proteins'], fig=fig)
    ##Draw colorbar
    fig.colorbar(f3Data)

    return fig

def drawResidDist(toDisplay, name):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    # the histogram of the data
    n, bins, patches = ax.hist(toDisplay, len(toDisplay)/200, normed=1, facecolor='green', alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    y = mpl.mlab.normpdf( bincenters, toDisplay.mean(), toDisplay.std())
    l = ax.plot(bincenters, y, 'r--', linewidth=1)

    ax.set_xlabel('Residual')
    ax.set_ylabel('Fraction')
    ax.set_title('Dist of ' + name + ' comps residulas: mu = ' + str(toDisplay.mean()) + ' std = ' + str(toDisplay.std()))

########################
def drawHeatMap(xdat, name="unnamed", colors=pylab.cm.RdBu, dendro=False, protColors=None, cIndex=None, km=None):
    """drawHeatMap produces a colored heatmap in a new figure window

    :param xdat: a data object (must contain fields 'data', 'fractions', 'proteins')
    :type xdat: dict
    :param colors: a color scale (a cmap)
    :type colors: cmap
    :param name: figure name and title
    :type name: str.
    :param dendro: a boolean to draw the dendrogram on the left
    :type dendro: bool.
    :param protColors: a color map used to label the protein names with group colors
    :type protColors: cmap
    :param cIndex: a list of groupIds for the proteins
    :type cIndex: list
    :param km: if present, will draw the kmeans cluster profiles at the top of the figure- input is a 2d-matrix - rowVectors for each centroid, each column is a fraction
    :type km: matrix
    :returns:  int -- the return code.
    :raises: AttributeError, KeyError
    :returns: a figure object

    """

    data = xdat['data']
    fractions = xdat['fractions']
    proteins = xdat['proteins']
    fig = pylab.figure()
    fig.suptitle(name)
    ##Draw heatmap
    offset = 0.1
    if dendro:
        xStart = 0.35
        xLength = 0.5
    else:
        xStart = 0.1
        xLength = 0.8
    if km is None:
        yStart = 0.1
        yLength = 0.8
    else:
        yStart = 0.1
        yLength = 0.7
    figData = heatMapAxes(data, dims = [xStart, yStart, xLength, yLength], fractions=fractions, proteins=proteins, protColors=protColors, cIndex=cIndex, fig=fig)
    ##Draw colorbar
    fig.colorbar(figData)

    if dendro:
        ax2Data = fig.add_axes([offset, offset, xLength-0.3, yLength])
        sch.dendrogram(xdat['rightDendro'], orientation='right', color_threshold=0.0)
        ax2Data.set_xticks([])
        ax2Data.set_yticks([])
        
    if not km is None:
        small = data.min()
        big = data.max()
        if math.fabs(small) > math.fabs(big):
            big = 0-small
        else:
            small = 0-big
        ax3Data = fig.add_axes([xStart, yLength+offset, xLength-0.1, 0.1])
        ax3Data.matshow(km, aspect='auto', origin='lower', cmap=colors, vmin=small, vmax=big)
        for i in range(len(km)):
            ax3Data.text(-0.75, i, 'clus'+str(i), verticalalignment="center", horizontalalignment="right", fontsize=12, color=cIndex(float(i)/(protColors.max()+1)))
        ax3Data.set_xticks([])
        ax3Data.set_yticks([])
    return fig

def heatMapAxes(data, dims=[0.1, 0.1, 0.7, 0.7], colors=pylab.cm.RdBu, fractions=None, proteins=None, protColors=None, cIndex=None, fig=None):
    """heatMapAxes draws a heatmap

    :param data: a datamatrix to draw
    :type xdat: a 2D Matrix
    :param dims: the size of the plot to draw - defaults to full window
    :type dims: list (4 elements)
    :param colors: color index to use - defaults to redblue
    :type colors: cmap
    :param fractions: fraction names
    :type fractions: list
    :param proteins: protein names
    :type proteins: list
    :param protColors: a color map used to label the protein names with group colors
    :type protColors: cmap
    :param cIndex: a list of groupIds for the proteins
    :type cIndex: list
    :param fig: where to plot the axes (which figure); defaults to new figure
    :type fig: matplotlib figure
    :returns:  an axes

    """
    if fig is None:
        fig = pylab.figure()
    axData = fig.add_axes(dims)
    for i in range(len(fractions)):
        axData.text(i, -0.5 , ' '+str(fractions[i]), rotation=270, verticalalignment="top", horizontalalignment="center", fontsize=12)
    if protColors == None:
        for i in range(len(proteins)):
            axData.text(-0.75, i, '  '+str(proteins[i]), verticalalignment="center", horizontalalignment="right", fontsize=12)
    else:
        for i in range(len(proteins)):
            axData.text(-0.75, i, '  '+str(proteins[i]), verticalalignment="center", horizontalalignment="right", fontsize=12, color=cIndex(float(protColors[i])/(protColors.max()+1)))
    small = data.min()
    big = data.max()
    if math.fabs(small) > math.fabs(big):
        big = 0-small
    else:
        small = 0-big
    figData = axData.matshow(data, aspect='auto', origin='lower', cmap=colors, vmin=small, vmax=big)
    #fig.colorbar(figData)
    axData.set_xticks([])
    axData.set_yticks([])

    return figData

def drawProfiles(km, cIndex=pylab.get_cmap('hsv')):
    """drawProfiles draws a scater plot of the given row vectors (points evenly spaced)
    the points are shaded along the given cmap by position in the array of arrays (km)

    :param km: a 2D array where each row vector is a separate profile to plte
    :type km: 2D Matrix
    :param cIndex: a color map, each row will take a differnt (equally spaced) color from the cmap
    :type km: cmap
    :returns: a handle to the resulting figure

    """
    
    fig = pylab.figure()
    asdf = fig.add_subplot(111)
    elements = len(km)
    plots = [asdf.scatter(range(len(km[i])), km[i], color=cIndex(float(i)/(elements+1))) for i in range(elements)]
    return fig

def drawSVDResidualPlot(meanAltArray, stdAltArray, meanNullArray, stdNullArray, sigmas, fig=None):
    """drawSVDResidualPlot draws a scater plot with the mean alternate hypothesis and mean null hypothesis.
    the plots are shadded by the desired number of sigmas. a blue dot shows the difference between the two means

    :param meanAltArray: a list of alternate hypothesis mean values
    :type meanAltArray: list (length of number of components)
    :param stdAltArray: a list of alternate hypothesis std values
    :type stdAltArray: list (length of number of components)
    :param meanNullArray: a list of null hypothesis mean values
    :type meanNullArray: list (length of number of components)
    :param stdNullArray: a list of null hypothesis std values
    :type stdNullArray: list (length of number of components)
    :param sigmas: number of sigmas to plot
    :type sigmas: int
    :param fig: a figure handle to draw in
    :type fig: matplotlib figure, defaults to new figure
    :returns: a handle to the resulting figure

    """
    if fig is None:
        fig = pylab.figure()
    axData = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    it = 1
    while it <= sigmas:
        axData.fill_between(range(1,len(meanNullArray)+1), meanNullArray-stdNullArray*it, meanNullArray+stdNullArray*it, alpha=1/float(sigmas), color='red', edgecolor='black')
        axData.fill_between(range(1,len(meanAltArray)+1), meanAltArray-stdAltArray*it, meanAltArray+stdAltArray*it, alpha=1/float(sigmas), color='green', edgecolor='black')
        it = it+1
    
    axData.scatter(range(1,len(meanNullArray)+1), meanNullArray, color='black')
    axData.scatter(range(1,len(meanAltArray)+1), meanAltArray, color='black')
    axData.scatter(range(1,len(meanAltArray)+1), meanNullArray-meanAltArray, color='b', s=50)
    return fig

def drawKmClusteredData(clusteredData, kClusters, finalPIndex, colorIndex, finalCents):
    """drawKmClusteredData draws a heatmap of the clusteredData with the centroid above

    :param clusteredData: a data object (must contain fields 'data', 'fractions', 'proteins')
    :type clusteredData: dict
    :param kClusters: number of centroids
    :type kClusters: int
    :param finalPIndex: a list of the centroids each protein is in (sorted same as clusteredData['proteins']
    :type finalPIndex: 2D datamatrix
    :param colorIndex: a list of the centroids each protein is in (sorted same as clusteredData['proteins']
    :typecolorIndexd: 2D datamatrix
    :param finalPIndex: a list of the centroids each protein is in (sorted same as clusteredData['proteins']
    :type d: 2D datamatrix
    :returns:  a resampled 2D datamatrix

    """
    finalPIndex = allKmRuns['protColorIndex'][-1]
    finalCents = allKmRuns['cents'][-1]
    kmcluteredMap = drawHeatMap(clusteredData, name="KMclusteredMap_"+str(kClusters), dendro=True, protColors=finalPIndex, cIndex=colorIndex, km=finalCents)



'''
def 2DProjection(xdata, axIndex1, axIndex2, byPCA, protColors=None, cIndex=None):
    xdat = xdata.copy()
    clusteredData = xdat['data']
    fractions = xdat['fractions']
    proteins = xdat['proteins']
    pi = xdat['pi']

    if byPCA:
        titleString = "PCA"
        #create PCA class with dataset
        myPCA = PCA(clusteredData)
        transformedData = myPCA.Y
        
        #determine loadings for first 2 PCs
        xLoad = myPCA.Wt[axIndex1]
        yLoad = myPCA.Wt[axIndex2]
    else:
        titleString = "SVD"
        #perform SVD
        U, Sig, Vh = SVD(clusteredData)
        #????transformedData = np.array(reconstructWithSVD(clusteredData, U, Sig, Vh, 3)[0])

        #determine loadings for first 2 comps
        xLoad = Vh[axIndex1]
        yLoad = Vh[axIndex2]
    #transform data using first 3 PCs
    x = []
    y = []
    dotCol = []
    iterator = 0

    for item in transformedData:
        x.append(item[0])
        y.append(item[1])
        if cIndex is None:
            dotCol.append('black')
        else:
            dotCol.append(cIndex(float(protColors[pi[iterator]])/(protColors.max()+1)))
        iterator = iterator+1

    #setup scores plot
    fig2D = pylab.figure()
    ax2D = fig2D.add_subplot(121)
    ax2D.scatter(x, y, s=50, c=dotCol, marker='o')
    
    #label scores plots
    for pindex, prot, xl, yl, zl in zip(pi, proteins, x, y, z):
        ax2D.annotate(str(prot), xy = (float(xl), float(yl)), xytext = (-15,15), textcoords = 'offset points', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    #label scores axis
    ax2D.set_xlabel(titleString + "1") 
    ax2D.set_ylabel(titleString + "2")
    ax2D.set_title("2D projection_scores " + titleString)

    #setup loadings plot
    axLoad2D = fig2D.add_subplot(122)
    axLoad2D.scatter(xLoad, yLoad, s=50, c='red', marker='o')
    #label loadings plot
    for frac, xl, yl in zip(fractions, xLoad, yLoad):
        axLoad2D.annotate(str(frac), xy = (float(xl), float(yl)), xytext = (-15,15), textcoords = 'offset points', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    #label scores axis
    axLoad2D.set_xlabel("Loadings on " + titleString + "1") 
    axLoad2D.set_ylabel("Loadings on " + titleString + "2")
    axLoad2D.set_title("2D projection_loadings " + titleString)
    #center axes
    axLoad2D.set_xlim([0-max(map(abs,xLoad))*1.25, max(map(abs,xLoad))*1.25])
    axLoad2D.set_ylim([0-max(map(abs,yLoad))*1.25, max(map(abs,yLoad))*1.25])



def doProjections(xdata, plt2D, plt3D, byPCA, protColors=None, cIndex=None):
    xdat = xdata.copy()
    clusteredData = xdat['data']
    fractions = xdat['fractions']
    proteins = xdat['proteins']
    pi = xdat['pi']

    if byPCA:
        titleString = "PCA"
        #create PCA class with dataset
        myPCA = PCA(clusteredData)
        transformedData = myPCA.Y
        
        #determine loadings for first 3 PCs
        xLoad = myPCA.Wt[0]
        yLoad = myPCA.Wt[1]
        zLoad = myPCA.Wt[2]

    else:
        titleString = "SVD"
        #perform SVD
        U, Sig, Vh = SVD(clusteredData)
        transformedData = np.array(reconstructWithSVD(clusteredData, U, Sig, Vh, 3)[0])

        #determine loadings for first 3 PCs
        xLoad = Vh[0]
        yLoad = Vh[1]
        zLoad = Vh[2]

    #transform data using first 3 PCs
    x = []
    y = []
    z = []
    dotCol = []
    iterator = 0

    for item in transformedData:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
        if cIndex is None:
            dotCol.append('black')
        else:
            dotCol.append(cIndex(float(protColors[pi[iterator]])/(protColors.max()+1)))
        iterator = iterator+1

    #make 2D plots of scores and loadings
    if plt2D:
        #setup scores plot
        fig2D = pylab.figure()
        ax2D = fig2D.add_subplot(121)
        ax2D.scatter(x, y, s=50, c=dotCol, marker='o')
        #label scores plots
        for pindex, prot, xl, yl, zl in zip(pi, proteins, x, y, z):
            ax2D.annotate(str(prot), xy = (float(xl), float(yl)), xytext = (-15,15), textcoords = 'offset points', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        #label scores axis
        ax2D.set_xlabel(titleString + "1") 
        ax2D.set_ylabel(titleString + "2")
        ax2D.set_title("2D projection_scores " + titleString)

        #setup loadings plot
        axLoad2D = fig2D.add_subplot(122)
        axLoad2D.scatter(xLoad, yLoad, s=50, c='red', marker='o')
        #label loadings plot
        for frac, xl, yl in zip(fractions, xLoad, yLoad):
            axLoad2D.annotate(str(frac), xy = (float(xl), float(yl)), xytext = (-15,15), textcoords = 'offset points', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        #label scores axis
        axLoad2D.set_xlabel("Loadings on " + titleString + "1") 
        axLoad2D.set_ylabel("Loadings on " + titleString + "2")
        axLoad2D.set_title("2D projection_loadings " + titleString)
        #center axes
        axLoad2D.set_xlim([0-max(map(abs,xLoad))*1.25, max(map(abs,xLoad))*1.25])
        axLoad2D.set_ylim([0-max(map(abs,yLoad))*1.25, max(map(abs,yLoad))*1.25])

    #make 3D plots of scores and loadings
    if plt3D:
        #setup scores plot
        fig3d = pylab.figure()
        ax3d = Axes3D(fig3d)
        
        ax3d.scatter(x, y, z, c=dotCol, s=100)
    
        xAxisLine = ((min(x), max(x)), (0,0), (0,0))
        ax3d.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
        yAxisLine = ((0, 0), (min(y), max(y)), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
        ax3d.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
        zAxisLine = ((0, 0), (0,0), (min(z), max(z))) # 2 points make the z-axis line at the data extrema along z-axis
        ax3d.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

        #label scores plots and update with moves
        labels = []
        def define_position(e):
            for pindex, prot, x1, y1, z1 in zip(pi, proteins, x, y, z):
                x2, y2, _ = proj3d.proj_transform(x1, y1, z1, ax3d.get_proj())
                labels.append([pylab.annotate(
                    prot, 
                    xy = (x2, y2), xytext = (10, 10),
                    textcoords = 'offset points', color = cIndex(float(protColors[pindex])/(protColors.max()+1)),
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')), prot, x1, y1, z1])
            fig3d.canvas.draw()
        
        def update_position(e):
            if len(labels)!=0:
                for label in labels:
                    x2, y2, _ = proj3d.proj_transform(label[2], label[3], label[4], ax3d.get_proj())
                    label[0].xy = x2,y2
                    label[0].update_positions(fig3d.canvas.renderer)
                fig3d.canvas.draw()

        
        fig3d.canvas.mpl_connect('scroll_event', define_position)
        fig3d.canvas.mpl_connect('button_release_event', update_position)

        #labels scores axis
        ax3d.set_xlabel(titleString + "1") 
        ax3d.set_ylabel(titleString + "2")
        ax3d.set_zlabel(titleString + "3")
        ax3d.set_title("3D " + titleString + " projection_scores")
        
        #setup loadings plot
        fig3DLoad = pylab.figure()
        ax3DLoad = Axes3D(fig3DLoad)
        
        ax3DLoad.scatter(xLoad, yLoad, zLoad, 'bo', s=100)
    
        xAxisLine = ((min(xLoad), max(xLoad)), (0,0), (0,0))
        ax3DLoad.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
        yAxisLine = ((0, 0), (min(yLoad), max(yLoad)), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
        ax3DLoad.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
        zAxisLine = ((0, 0), (0,0), (min(zLoad), max(zLoad))) # 2 points make the z-axis line at the data extrema along z-axis
        ax3DLoad.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

        #labels scores axis
        ax3DLoad.set_xlabel(titleString + "1") 
        ax3DLoad.set_ylabel(titleString + "2")
        ax3DLoad.set_zlabel(titleString + "3")
        ax3DLoad.set_title("3D " + titleString + " projection_Loadings")

        #label loadings plots and update with moves
        labelsLoad = []
        def define_position(e):
            for frac, x1, y1, z1 in zip(fractions, xLoad, yLoad, zLoad):
                x2, y2, _ = proj3d.proj_transform(x1, y1, z1, ax3DLoad.get_proj())
                labelsLoad.append([pylab.annotate(
                    frac, 
                    xy = (x2, y2), xytext = (10, 10),
                    textcoords = 'offset points',
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')), frac, x1, y1, z1])
            fig3DLoad.canvas.draw()
        
        def update_position(e):
            if len(labelsLoad)!=0:
                for label in labelsLoad:
                    x2, y2, _ = proj3d.proj_transform(label[2], label[3], label[4], ax3DLoad.get_proj())
                    label[0].xy = x2,y2
                    label[0].update_positions(fig3d.canvas.renderer)
                fig3DLoad.canvas.draw()

        
        fig3DLoad.canvas.mpl_connect('scroll_event', define_position)
        fig3DLoad.canvas.mpl_connect('button_release_event', update_position)
'''
