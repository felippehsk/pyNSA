import math
import pickle
import numpy as np
import scipy.signal
import pandas as pd
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as pyplot


def latconv(lat,minc,F_lat):
   return (lat-minc)*F_lat

def longconv(long,mind,F_long):
   return (long-mind)*F_long

######################################

root = Tk()
filename = filedialog.askopenfilename(filetypes = (("Template files", "*.txt"), ("All files", "*")))
root.destroy()
if len(filename) > 0:
    print("You chose %s" % filename)


nsaData = pd.read_table(filename, sep='\t', header='infer', names=None, index_col=False, usecols=None)
labels=nsaData.columns.values

useTime=False
if(useTime):
    startingColumn=2
else:
    startingColumn=3

######################################

if 'Elevation' in nsaData.columns:
    h=nsaData['Elevation'].mean()
else:
    h=200
a=6378137
b=6356752.3142
c=nsaData['Latitude'].mean()
d=math.sqrt((a*math.cos(c))**2+(b*math.sin(c))**2)
F_long=(np.pi*math.cos(c)/180)*((a**2/d)+h)
F_lat=(np.pi/180)*(((a*b)**2/d**3)+h)

LongMin = nsaData['Longitude'].min()
LongMax = nsaData['Longitude'].max()
LatMin = nsaData['Latitude'].min()
LatMax = nsaData['Latitude'].max()
convParams = [F_long,LongMin,F_lat,LatMin]

nsaData['Lat_y'] = nsaData['Latitude'].apply(lambda row : latconv(row,LatMin,F_lat)) # planer matrix Latitude
nsaData['Long_x'] = nsaData['Longitude'].apply(lambda row : longconv(row,LongMin,F_long))# Planer matrix Longitude

########################################

gdsz = 20
gdc = gdsz / 2
X = np.array(nsaData['Long_x'])
Y = np.array(nsaData['Lat_y'])
Z = np.array(nsaData.iloc[:,startingColumn:-2])
xmin=X.min()
xmax=X.max()
ymin=Y.min()
ymax=Y.max()
Xr = np.linspace(xmin, xmax, int((xmax - xmin)/gdsz))
Xc=Xr[0:-1]+gdsz/2
Yr = np.linspace(ymin, ymax, int((ymax - ymin)/gdsz))
Yc=Yr[0:-1]+gdsz/2
ngx=len(Xc)
ngy=len(Yc)
[nd,nv]=Z.shape
ar=np.zeros((nv, ngy, ngx))
zar=np.zeros((ngy,ngx),dtype=int)
for l in range(ngx):
    for m in range(ngy):
        for n in range(nd):
            if max(abs(X[n]-Xc[l]),abs(Y[n]-Yc[m]))<=gdsz/2:
                zar[m,l]+=1
for l in range(ngx):
    for m in range(ngy):
        i = 0
        aux = np.zeros((zar[m, l],nv))
        for n in range(nd):
            if max(abs(X[n]-Xc[l]),abs(Y[n]-Yc[m]))<=gdsz/2:
                aux[i, :] = Z[n, :]
                i += 1
        if(zar[m, l]==0): # Masking by field area by outside 0
            ar[:, m, l]=np.zeros((1,nv))
        else:
            ar[:, m, l]=np.mean(aux, axis=0)
for l in range(ngx):
    for m in range(ngy):
        if(zar[m,l]>0):
            zar[m, l]=1

######################################

for i in range(nv):
    ar[i,:,:]=scipy.signal.medfilt2d(ar[i,:,:], 5)
    arMasked=np.ma.masked_where(ar[i,:,:]==0, ar[i,:,:])
    pyplot.figure()
    pyplot.imshow(arMasked, interpolation='none', origin='lower',extent=[0,ngx*gdsz,0,ngy*gdsz])
    pyplot.title(labels[i+startingColumn])
    pyplot.ylabel('Northing')
    pyplot.xlabel('Easting')
    pyplot.colorbar()
    ax=pyplot.gca()
    ax.set_xticks(np.arange(gdsz, gdsz*(ngx+1), gdsz), minor=True)
    ax.set_yticks(np.arange(gdsz, gdsz*(ngy+1), gdsz), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.1)
    pyplot.savefig(labels[i+startingColumn]+'.png', dpi=200)
    pyplot.close()    

 
z = ar.shape

with open('NSATemp.pickle', 'wb') as outfile:
    pickle.dump([zar,ar,z,gdsz,labels,convParams,startingColumn],outfile)