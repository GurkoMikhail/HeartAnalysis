import numpy as np
import pyqtgraph as pg
import matplotlib.image as mi
import scipy.ndimage as si
from pathlib import Path
from pyqtgraph.Qt import QtGui
pg.setConfigOptions(imageAxisOrder='row-major')

def loadPhantom(phantomName, phantomSize):
    return np.loadtxt(f'Dat phantoms/{phantomName}.dat').reshape(phantomSize, order='F')

def rotatePhantom(phantom, angles):
    phantom = si.rotate(phantom, angle=angles['xz'], axes=(0, 2), reshape=False, order=1)
    phantom = si.rotate(phantom, angle=angles['xy'], axes=(0, 1), reshape=False, order=1)
    return phantom

def cutHeart(phantom, numSlices):
    x = numSlices['x']
    y = numSlices['y']
    z = numSlices['z']
    return phantom[x[0]:x[1], y[0]:y[1], z[0]:z[1]]

def sliceToImages(data, axis):
    data = np.rot90(data, k=1, axes=(1, 2))
    if axis == 0:
        pass
    elif axis == 1:
        data = np.rot90(data, k=1, axes=(0, 2))
    elif axis == 2:
        data = np.rot90(data, k=-1, axes=(0, 1))
    return data

def saveImage(data, name, format='jpg', cmap='gnuplot2', levels=None, zoom=10):
    if levels is None:
        levels = [np.min(data), np.max(data)]
    else:
        levels = [levels['min'](data), levels['max'](data)]
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = si.zoom(data, zoom, order=1)
        mi.imsave(f'{name}.{format}', data, cmap=cmap, vmin=levels[0], vmax=levels[1])
    else:
        for i, image in enumerate(data, 1):
            image = si.zoom(image, zoom, order=1)
            mi.imsave(f'{name}{i}.{format}', image, format=format, cmap=cmap, vmin=levels[0], vmax=levels[1])

def initVis():
    pg.mkQApp('Heart visualuzation')
    win = QtGui.QMainWindow()
    win.resize(800, 700)
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)
    win.setWindowTitle('Heart visualuzation')
    return win, layout

def addVis(layout, data, levels=None, zoom=10, row=0, column=0):
    data = si.zoom(data, (1, zoom, zoom), order=1)
    if levels is None:
        levels = [np.min(data), np.max(data)]
    else:
        levels = [levels['min'](data), levels['max'](data)]
    imv = pg.ImageView()
    cmap = pg.colormap.get('gnuplot2', 'matplotlib')
    imv.setColorMap(cmap)
    layout.addWidget(imv, row, column)
    imv.setImage(data, levels=levels)


if __name__ == '__main__':
    phantomNames = [
        'efg3_cut',
        'fgr3-osem-nonAC',
        'fgr3-osem-AC',
        'efg3cutDefect',
        'fgr3-osem-nonAC-iscemija',
        'fgr3-osem-AC-iscemija',
    ]
    comparisonRowsNumber = 3
    axes = [
        'Short',
        'Vertical',
        'Horizontal'
    ]
    phantomSize = np.array([
        128,
        128,
        100
    ])
    angles = {
        'xz': 23,
        'xy': -37
    }
    numSlices = {
        'x':[50, 90],
        'y':[53, 83],
        'z':[50, 80]
    }
    levelsSet = {
        'Normal': {
            'min': lambda data: np.min(data),
            'max': lambda data: np.max(data)
        },
        'Clipped': {
            'min': lambda data: np.min(data)*2,
            'max': lambda data: np.max(data)*0.6
        }
    }

    win, layout = initVis()

    hearts = {}
    for phantomName in phantomNames:
        phantom = loadPhantom(phantomName, phantomSize)
        phantom = rotatePhantom(phantom, angles)
        heart = cutHeart(phantom, numSlices)
        hearts.update({phantomName: heart})
    for i, (levelsType, levels) in enumerate(levelsSet.items()):
        for axis, axisName in enumerate(axes):
            heartImages = []
            for phantomName, heart in hearts.items():
                slices = sliceToImages(heart, axis=axis)
                saveImage(slices, f'Images/Heart/{levelsType}/{phantomName}/{axisName}/{axisName.lower()}', levels=levels)
                slices /= slices.max()
                heartImages.append(slices)
            comparisonImage = []
            for i in range(len(heartImages)//comparisonRowsNumber):
                i0 = comparisonRowsNumber*i
                ix = i0 + comparisonRowsNumber
                comparisonImage.append(np.concatenate(heartImages[i0:ix], axis=1))
            comparisonImage = np.concatenate(comparisonImage, axis=2)
            saveImage(comparisonImage, f'Images/Heart/Comparison{levelsType}/{axisName}/{axisName.lower()}', levels=levels)
            addVis(layout, comparisonImage, levels, row=i, column=axis)

    win.show()
    pg.exec()
    
