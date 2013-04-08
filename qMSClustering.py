import csv
import matplotlib.pyplot as pylab
import scipy.linalg as sl
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.cluster.vq as scv
import numpy as np
import numpy.linalg as npla
import networkx as nx
import matplotlib.colors as mpc
import random
import matplotlib._pylab_helpers

from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import proj3d
from jhdavisVizLib import *
from string import ascii_letters
from matplotlib import mpl
from mpl_toolkits.mplot3d import Axes3D

def readDataFile(filename, scale=1.0):
    """readDataFile reads a datafile. The datafile should be tab separated and have both column and row headers listing the proteins/fractions.
    Empty values should be empty, they will be treated as np.NAN - see lambda below
    readDataFile takes an optional scale that will multiply the data by the specified factor - useful for "scaling" error

    :param filename: a string of the filename to be read
    :type filename: string
    :param scale: a float indicating how to scale the data, default is unscaled (1)
    :type scale: float
    :returns:  a data ditionary. 'data', 'fractions', 'proteins' are filled in, others are None

    """
    with open(filename, 'rb') as inputFile:
        csvFile = list(csv.reader(inputFile, delimiter = '\t'))
        header = csvFile[0]
        proteins = [row[0] for row in csvFile[1:]]
        insertValue2 = lambda x, y: np.NAN if x=='' else float(x)*y
        data = np.array([[insertValue2(col, scale) for col in row[1:]] for row in csvFile[1:]])
        inputFile.close()
        return {'fractions':header[1:], 'proteins':proteins, 'fi':None, 'pi':None, 'data':data}

def transformData(x, y, rowMethod, columnMethod):
    """transformData transforms an object x according to indicies in object y; used to sort different datasets by the same protein/fraction indicies

    :param x: a data dictionary - the one to be transformed
    :type x: dict, must contain 'data'
    :param y: a data dictionary - the one containing the flipping indices
    :type y: dict, must contain 'data', 'fi', 'pi'
    :param rowMethod: a boolean asking if you want to flip on the rows (proteins get sorted appropriately)
    :type rowMethod: bool
    :param columnMethod: a boolean asking if you want to flip on the columns (fractions get sorted appropriately)
    :type columnMethod: bool
    :returns:  a data ditionary. All fields(except 'data') in x are replaced by those in y, the data is transformed appropriately

    """
    toReturn = y.copy()
    xt = x['data']
    idx2 = y['fi']
    idx1 = y['pi']
    if rowMethod:
        xt = xt[idx1,:]
    if columnMethod:
        xt = xt[:,idx2]
    toReturn['data'] = xt
    return toReturn
    
def clusterData(xdata, rowMethod=True, columnMethod=False, method='average', metric='euclidean'):
    """clusterData clusters the data either by row, by column, or both

    :param xdata: a data dictionary - the one to be transformed
    :type x: dict, must contain 'data', 'proteins', 'fractions'
    :param rowMethod: a boolean asking if you want to flip on the rows (proteins get clustered)
    :type rowMethod: bool
    :param columnMethod: a boolean asking if you want to flip on the columns (fractions get clustered)
    :type columnMethod: bool
    :param method: string defining the linkage type, defaults to 'average' - 'ward' might be a good option
    :type method: string
    :param metric: string defining the distance metric, defaults to 'euclidean'
    :type metric: string
    :returns:  a data ditionary. 'data', 'proteins', 'fractions', 'fi', 'pi', 'topDendro', 'rightDendro' are updated

    """
        
    xdat = xdata.copy()
    x = xdat['data']
    ind1 = xdat['proteins']
    ind2 = xdat['fractions']
    xt = x
    idx1 = None
    idx2 = None
    
    toReturn = xdat
    Y1 = None
    Y2 = None
    if rowMethod:
        d1 = ssd.pdist(x)
        D1 = ssd.squareform(d1)  # full matrix
        Y1 = sch.linkage(D1, method=method, metric=metric) ### gene-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z1 = sch.dendrogram(Y1, no_plot=True, orientation='right')
        idx1 = Z1['leaves'] ### apply the clustering for the gene-dendrograms to the actual matrix data
        xt = xt[idx1,:]   # xt is transformed x
        newIndex = []
        for i in idx1:
            newIndex.append(ind1[i])
        toReturn['proteins'] = newIndex
        toReturn['pi'] = idx1
    if columnMethod:
        d2 = ssd.pdist(x.T)
        D2 = ssd.squareform(d2)
        Y2 = sch.linkage(D2, method=method, metric=metric) ### array-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z2 = sch.dendrogram(Y2, no_plot=True)
        idx2 = Z2['leaves'] ### apply the clustering for the array-dendrograms to the actual matrix data
        xt = xt[:,idx2]
        newIndex = []
        for i in idx2:
            newIndex.append(ind2[i])
        toReturn['fractions'] = newIndex
        toReturn['fi'] = idx2
    toReturn['data'] = xt
    toReturn['topDendro'] = Y2
    toReturn['rightDendro'] = Y1
    return toReturn

def determineSVDResiduals(origDataAlt, err, testNumber=100):
    """determineSVDResiduals calculates the distribution of residuals for each compoents on either
    the Null (scrambled) or Alt(real, resampled from error values) datasets

    :param origDataAlt: a data dictionary with the dataset to be analyzed
    :type origDataAlt: dict, must contain 'data', 'proteins', 'fractions'
    :param err: a data dictionary with the error values to be analyzed, must be standard deviations
    :type err: dict, must contain 'data', 'proteins', 'fractions'
    :param testNumber: an integer of the number of times to run the test to build the distribution
    :type testNumber: int
    :returns:  a list bearing
        the mean array for the Alt hypothesis at each components (length is number of fractions),
        the std array for the Alt hypothes,
        the mean array for the Null hypothesis,
        the std array for the Null hypothes,

    """
    if not(origDataAlt['fractions'] == err['fractions'] and origDataAlt['proteins'] == err['proteins']):
        print "error in determineSVDResiduals"
        return None
    else:
        it = 0
        altHash = dict()
        nullHash = dict()
        newDataAlt = origDataAlt.copy()
        while it < testNumber:
            newDataAlt['data'] = resampleUsingErr(origDataAlt['data'], err['data'])
            altHash = calcSVDResid(newDataAlt, altHash)
            newDataNull = newDataAlt.copy()
            newDataNull['data'] = resampleRandFracs(newDataNull['data'])
            nullHash = calcSVDResid(newDataNull, nullHash)
            it = it + 1
        meanAltArray, stdAltArray = fillMeanStdArray(altHash)
        meanNullArray, stdNullArray = fillMeanStdArray(nullHash)
        return [meanAltArray, stdAltArray, meanNullArray, stdNullArray]

def calcSVDResid(dataset, hashToAdd):
    """calcSVDResid is a helper function that adds a new residual value to a hash bearing
    lists of residuals for each number of components

    :param dataset: a data dictionary with the dataset to be analyzed
    :type dataset: dict, must contain 'data', 'fractions'
    :param hashToAdd: a hash with keys as number of components used and values as list of variance observed
    :type hashToAdd: dict with values as lists, this func will append one value to each list
    :returns:  a dict bearing lists for each number of components, the lists are the residuals observed at
        each component

    """
    U, Sig, Vh = SVD(dataset['data'])
    resids = [reconstructWithSVD(dataset['data'], U, Sig, Vh, x)[2] for x in range(1, len(dataset['fractions'])+1)]
    for i in range(1, len(resids)+1):
        if (not i in hashToAdd.keys()):
            hashToAdd[i] = []
        hashToAdd[i].append(resids[i-1])
    return hashToAdd

def fillMeanStdArray(residHash):
    """fillMeanStdArray is a helper function that takes a hash with lists of residuals and returns
    lists with the mean or std deviation in those lists (keeps the returned lists ordered by
    number of components

    :param residHash: a dict with keys as number of components, values as lists of observed resids
    :type data: dict
    :returns:  two lists bearing
        the mean of the residuals (sorted by number of components used
        the std of the residuals (sorted by number of components used

    """
    meanArray = []
    stdArray = []
    for key in residHash.keys():
        meanArray.append(np.array(residHash[key]).mean())
        stdArray.append(np.array(residHash[key]).std())
        
    meanArray = np.array(meanArray)
    stdArray = np.array(stdArray)
    return meanArray, stdArray

def SVD(data):
    """SVD performs singular value decomposition on a 2D matrix dataset

    :param data: the 2D matrix to factor
    :type data: 2D matrix
    :returns:  array bearing U, Sig, Vh in that order

    """
    M,N = data.shape
    U,s,Vh = sl.svd(data)
    Sig = sl.diagsvd(s,M,N)

    return [U,Sig,Vh]

def reconstructWithSVD(data, U, Sig, Vh, comps):
    """reconstructs a dataset from a SVD using a set number of components

    :param data: the 2D matrix to be reconstructed
    :type data: 2D matrix
    :param U: the 2D matrix U
    :type data: 2D matrix
    :param Sig: the 2D matrix Sig
    :type data: 2D matrix
    :param Vh: the 2D matrix Vh
    :type data: 2D matrix
    :param comp: the integer number of components to reconstruct with
    :type data: int
    :returns:  array bearing the reconstructed data, the residual data,
        and a float with the sum of the residual in that order

    """
    M,N = data.shape
    test = np.zeros((N,N))
    i = 0
    while i < comps:
        test[i,i] = 1
        i = i +1
    Sig = np.asmatrix(Sig)*np.asmatrix(test)

    recon = np.asmatrix(U)*np.asmatrix(Sig)*np.asmatrix(Vh)
    resid = np.abs(data - recon)

    diff = np.subtract(data, recon)
    diff = np.abs(diff)
    return [recon, resid, np.sum(diff)]

def doSwap(r):
    """doSwap is a helper function that randomly swaps two elements in a row

    :param r: a row
    :type r: list
    :returns:  a list with two elements randomly swapped

    """
    row = r.copy()
    ind1 = np.random.randint(0, len(row))
    ind2 = np.random.randint(0, len(row))
    hold = row[ind1]
    row[ind1] = row[ind2]
    row[ind2] = hold
    return row

def resampleRandFracs(d):
    """resampleRandFracs is a helper function that generates a randomzied (Null hypothesis)
    dataset by randomly swapping elements of each row

    :param d: a 2D datamatrix to be randomized
    :type d: 2D datamatrix
    :returns:  a randomized 2D datamatrix

    """
    allVects = []
    data = d.copy()
    for rowVect in data:
        j = 0
        while j < len(rowVect):
            rowVect = doSwap(rowVect)
            j = j+1
        allVects.append(rowVect)
    newData = np.array(allVects)
    return newData

def resampleUsingErr(data, err):
    """resampleUsingErr is a helper function that resamples a dataset using an error
    matrix dataset in the SAME orientation, error should be in standard deviations

    :param data: a 2D datamatrix to be resampled
    :type d: 2D datamatrix
    :param err: a 2D datamatrix to be resampled from - should bear standard deviations
    :type d: 2D datamatrix
    :returns:  a resampled 2D datamatrix

    """
    newData = []
    for i in range(0,len(data)):
        newRow = []
        for j in range (0, len(data[0])):
            newRow.append(np.random.normal(loc=data[i][j], scale=err[i][j]))
        newData.append(newRow)
    return np.array(newData)

def calcDistortion(cents, idx, data, m='euclidean'):
    """calcDistortion is a  helper function that determines the distance between datasets and centroids

    :param cents: a 2D matrix of the centroids (ordered)
    :type cents: 2D matrix
    :param idx: a list of which vector in data is grouped with which centroid
    :type idx: list
    :param data: a 2D matrix of the data, we will determine the summed distance from each row in data to it centroid
        and then sum over all rows
    :type data: 2D matrix
    :returns: a float with the total distortion
    
    """
    distor = 0.0
    for i in range(0, len(data)):
        X = [cents[idx[i]], data[i]]
        additionalDistor = ssd.pdist(X, metric=m)[0]
        distor = distor + additionalDistor
    return distor

def arePaired(a, b, groupings):
    """arePaired is a helper function, returns 1 if groupings[a] == groupings[b]

    :param a: index a
    :type a: int
    :param b: index b
    :type b: int
    :param groupings: list of grouping to check
    :type groupings: list
    :returns:  1 if a, b are in the same group, 0 else
    
    """
    if groupings[a] == groupings[b]:
        return 1
    else:
        return 0

def kMeansCluster(x, k, trials):
    """kMeansCluster performs k means clustering on a dataset

    :param x: a data object (must contain field 'data')
    :type x: dict
    :param k: the number of centroids to cluster to
    :type k: int
    :param trials: the number of times to run kmeans2 (will be run with both 'random'
        and 'points'. The best of the two trials will be used.
    :type trials: int
    :returns:  a dictionary with keys idx and cents.
        idx is the group number for each protein (in the orde given in the x data object
        cents is a list of rowVectors with the centroids for each cluster

    """
    data = x['data']
    oldDistR = np.inf
    oldDistP = np.inf

    centsR, idxR = scv.kmeans2(data.copy(), k, iter=trials, minit='random')
    centsP, idxP = scv.kmeans2(data.copy(), k, iter=trials, minit='points')
    distR = calcDistortion(centsR, idxR, data)
    distP = calcDistortion(centsP, idxP, data)

    if distR > distP:
        centsR = centsP
        idxR = idxP
        distR = distP
    return {'idx': idxR, 'cents':centsR}

def iterateKMeansCluster(origData, errData, kClusters, kMeansTests, kMeansRuns):
    """iterateKMeansCluster performs k means clustering on a dataset multiple times,
    resmapling the dataset from teh distribution defined by errData each time.
    It clusters the data into kClusters

    :param origData: a data object (must contain fields 'data', 'fractions', 'proteins')
    :type origData: dict
    :param errData: a data object (must contain field 'data') and be in teh same orientation
        as the origData object
    :type errData: dict
    :param kClusters: the number of centroids to generate
    :type kClusters: int
    :param kMeansTests: the number of iterations to try k means clustering with (in each round
    :type kMeansTests: int
    :param kMeansRuns: the number of resampling runs do to
    :type kMeansRuns: int
    :returns:  an array of dictionaries, each row is 1 run of kMeansCluster
        each dictionary has a 'protColorIndex' field that lists the group that protein was in
        and a 'cents' field that gives the profile for that centroid

    """
    clusteredData = origData.copy()

    allKmRuns = {'cents':[], 'protColorIndex':[]}

    datToKM = clusteredData.copy()
    while (kMeansRuns > 0):
        kmclus = kMeansCluster(datToKM, kClusters, kMeansTests)
        allKmRuns['protColorIndex'].append(kmclus['idx'])
        allKmRuns['cents'].append(kmclus['cents'])
        #allKmRuns.append(protColorIndex)
        datToKM['data'] = resampleUsingErr(clusteredData['data'], errData['data'])
        kMeansRuns = kMeansRuns-1
    return allKmRuns

    
def makeKMeansCorrelationMatrix(allKmRuns, clusteredData):
    """makeKMeansCorrelationMatrix builds a correlation matrix data object using the groupings given in allKmRuns
    each entry is the number of times element i and j were in the same cluster (i.e. each row vector of allKmRuns
    that has protein i and protein j in the same group increments the returned matrix[i][j] by one

    :param allKmRuns: a 2D matrix
    :type allKmRuns: dict
    :param origData: a 2D datamatrix to be resampled
    :type origData: 2D datamatrix
    :param err: a 2D datamatrix to be resampled from - should bear standard deviations
    :type err: 2D datamatrix
    :returns:  a data object that is a copy of clusteredData with the following changes,
        'fractions' has the 'proteins' of clusteredData
        'data' has the number of times that two proteins co-grouped based on the allKmRuns data input

    """
    correlationData = clusteredData.copy()
    kmRunCorrs = np.zeros(shape=(len(clusteredData['proteins']), len(clusteredData['proteins'])))

    for groupings in allKmRuns['protColorIndex']:
        for i in range(0, len(correlationData['proteins'])):
            for j in range(0, len(correlationData['proteins'])):
                kmRunCorrs[i][j] = kmRunCorrs[i][j] + arePaired(i,j, groupings)
    
    correlationData['data'] = kmRunCorrs
    correlationData['fractions'] = correlationData['proteins']
    return correlationData


def drawSprings(corrMatrix, kclusters, initialPosFileName, mult=1.1):
    """drawSprings draws two figures, one of the spring evolution at 9 timepoints, one of the initial vs. final.

    :param corrMatrix: a data object generated by kmeans cluster - has the correlations between objects.
    :type corrMatrix: dict - must have 'data' and 'proteins'
    :param kclusters: the number of kclusters used in the analysis (what was passed to kmeans)
    :type kclusters: int
    :param initialPosFileName: a string pointing to the initial positions of each node - see nierhausPositions.txt
    :type initialPosFileName: string
    :param mult: a float of the base of the exponent to exapand on in each spring interation
    :type mult: float
    :returns:  an array of figures - first is the evolution, second is the start and end.
    
    """
    G = nx.Graph()
    positions = readDataFile(initialPosFileName)
    initialNodePos = {positions['proteins'][i]: [float(positions['data'][i,0])/4.0, float(positions['data'][i,1])/5.0] for i in range(len(positions['proteins']))}
       
    [G.add_node(x) for x in corrMatrix['proteins']]

    connection = (lambda x,y: corrMatrix['data'][x][y])

    [G.add_edge(corrMatrix['proteins'][x],corrMatrix['proteins'][y], weight=connection(x,y)) for x in range(len(corrMatrix['proteins'])) for y in range(len(corrMatrix['proteins'])) if (connection(x,y)!=0)]
    weights = [G.get_edge_data(x[0],x[1])['weight'] for x in G.edges()]

    notes = pylab.figure()
    ax = notes.add_subplot(1,2,1, title="nierhaus positions_kclusters=" + str(kclusters))
    drawNodePos = initialNodePos
    yDiff = [initialNodePos[x][1] - drawNodePos[x][1] for x in drawNodePos.keys()]
    nx.draw_networkx(G, pos = drawNodePos,
                    node_size=600, node_color = yDiff, cmap = pylab.cm.RdBu, vmin=-1, vmax=1,
                    edge_color=weights, edge_vmin = 0, edge_vmax = kMeansRuns, edge_cmap = pylab.cm.autumn_r, width=2,
                    font_size=10, font_weight='bold')

    iters = int(mult**8)
    ax = notes.add_subplot(1,2,2, title="spring iteration=" + str(iters) + "_kclusters=" + str(kclusters))
    drawNodePos = nx.spring_layout(G, pos=initialNodePos, iterations=iters)
    yDiff = [initialNodePos[x][1] - drawNodePos[x][1] for x in drawNodePos.keys()]
    nx.draw_networkx(G, pos = drawNodePos,
                    node_size=600, node_color = yDiff, cmap = pylab.cm.RdBu, vmin=-1, vmax=1,
                    edge_color=weights, edge_vmin = 0, edge_vmax = kMeansRuns, edge_cmap = pylab.cm.autumn_r, width=2,
                    font_size=10, font_weight='bold')

    cb1ax = notes.add_axes([0.025, 0.1, 0.05, 0.8])
    pylab.colorbar(cmap=pylab.cm.autumn_r, cax=cb1ax)
    cb2ax = notes.add_axes([0.925, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb2 = mpl.colorbar.ColorbarBase(cb2ax, cmap=pylab.cm.RdBu_r, norm=norm, orientation='vertical')
        
    notes2 = pylab.figure()
    drawNodePos = initialNodePos
    for i in range(9):
        ax = notes2.add_subplot(3,3,i+1, title="iterations=" + str(int(mult**i)) + "_k=" + str(kclusters))
        drawNodePos = nx.spring_layout(G, pos=drawNodePos, iterations=int(mult**i))

        yDiff = [initialNodePos[x][1] - drawNodePos[x][1] for x in drawNodePos.keys()]
        nx.draw_networkx(G, pos = drawNodePos,
                        node_size=600, node_color = yDiff, cmap = pylab.cm.RdBu, vmin=-1, vmax=1,
                        edge_color=weights, edge_vmin = 0, edge_vmax = kMeansRuns, edge_cmap = pylab.cm.autumn_r, width=2,
                        font_size=10, font_weight='bold')

if __name__ == '__main__':
####initialize stuff####
    colors = pylab.cm.RdBu
    method = 'ward'
    metric = 'euclidean'
    path = '/home/jhdavis/scripts/python/qMSClustering/testData/'
    fileToRead = path+'ssc_50S_medLN_zeros.txt'
    errorFile = path+'ssc_50S_stdErrLN_zeros.txt'
    testNumber = 5
    doPCA = True
    pltPCA = True
    showPlots = True
    savePlots = False
    sigmas = 3
    nodeColorRange = pylab.get_cmap('hsv')
    kClusters = 6

    kMeansTests = 600
    kMeansRuns = 400

    pylab.close('all')
    
    rawData = readDataFile(fileToRead, 1)
    errData = readDataFile(errorFile, 1)

    rawMap = drawHeatMap(rawData, "rawMap")

    
    clusteredData = clusterData(rawData, True, False, method, metric)
    clusteredErr = transformData(errData, clusteredData, True, False)

    cluteredMap = drawHeatMap(clusteredData, "clusteredMap", dendro=True)
    cluteredMap = drawHeatMap(errData, "stdErrorMeanMap", dendro=False)
    
    U, Sig, Vh = SVD(clusteredData['data'])

    performAllReconstructions(clusteredData, U, Sig, Vh, figure=False)
        
    resampleUsingErr(clusteredData['data'], clusteredErr['data'])

    [uAlt, sigAlt, uNull, sigNull] = determineSVDResiduals(clusteredData, clusteredErr, testNumber=100)
    drawSVDResidualPlot(uAlt, sigAlt, uNull, sigNull, sigmas)
    #doProjections(clusteredData, pltPCA, pltPCA, (not doPCA))
    #doProjections(clusteredData, pltPCA, pltPCA, doPCA, protColors=protColorIndex, cIndex=colorIndex)
    kclusters = kClusters
    kmRuns = iterateKMeansCluster(clusteredData, errData, kclusters, kMeansTests, kMeansRuns)
    finalPIndex = kmRuns['protColorIndex'][-1]
    finalCentroids = kmRuns['cents'][-1]
    cluteredMap = drawHeatMap(clusteredData, name="MapColoredbyKm" + str(kclusters), dendro=True, protColors=finalPIndex, cIndex=nodeColorRange, km=finalCentroids)
    corrMatrix = makeKMeansCorrelationMatrix(kmRuns, clusteredData)
    corrMap = drawHeatMap(corrMatrix, name="corrMapKm " + str(kclusters), dendro=False, protColors=finalPIndex, cIndex=nodeColorRange)
    drawSprings(corrMatrix, kclusters, path+"nierhausPositions.txt", mult=3)

    
    '''usefull calls:

    rawMap = drawHeatMap(rawData, "rawMap")
    reconstructedData = clusteredData.copy()
    reconstructedData['data'] = reconstructWithSVD(clusteredData['data'], U, Sig, Vh, 3)[0]
    kmRuns = iterateKMeansCluster(reconstructedData, errData, kClusters, kMeansTests, kMeansRuns, True)
    reconMap = drawHeatMap(reconstructedData, "reconstructed_3comps", dendro=True)
    plotComponents(U, 10, proteins=clusteredData['proteins'])
    drawProfiles(clusteredData['data'])

    kmRuns = iterateKMeansCluster(clusteredData, errData, kClusters, kMeansTests, kMeansRuns)
    finalPIndex = kmRuns['protColorIndex'][-1]
    finalCentroids = kmRuns['cents'][-1]
    cluteredMap = drawHeatMap(clusteredData, name="MapColoredbyKm", dendro=True, protColors=finalPIndex, cIndex=nodeColorRange, km=finalCentroids)
    corrMatrix = makeKMeansCorrelationMatrix(kmRuns, clusteredData)
    corrMap = drawHeatMap(corrMatrix, name="corrMap", dendro=False, protColors=finalPIndex, cIndex=nodeColorRange)
    
    '''
    if savePlots:
        figures=[manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        print figures
        
        for i, figure in enumerate(figures):
            figure.set_size_inches(24, 18)
            figure.savefig('figure%d.png' % i, bbox_inches='tight', dpi=100)
    if showPlots:
        pylab.show() # show the plot
        
        
