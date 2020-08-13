##You need to change the A, B points setting for EdgeTrak data which start from tongue root
## Point A is x1, y1 Point B is x2, y2 for EdgeTrak
##You also need to change the column number as AAA has 42 x,y coordinates but EdgeTrak has 100 points.

import csv
import numpy as np
import pandas as pd
import sys
import glob
import math
import os

from scipy import integrate
from scipy.signal import butter, filtfilt


def Curve(start, end, result):
    temp = data.loc[start:end]
    xy_pairs = []
    for i in range(0, end - start):
        pair = [temp.loc[start + i][0], temp.loc[start + i][1]]
        xy_pairs.append(pair)
    xy_pairs = np.array(xy_pairs)
    mci = curvature_index(xy_pairs)
    rl, im, mod = fourier_analysis(xy_pairs)
    str1 = temp.iloc[1, 2]
    x1 = float(temp.head(1)['x'])
    y1 = float(temp.head(1)['y'])
    x2 = float(temp.tail(1)['x'])
    y2 = float(temp.tail(1)['y'])
    px = 0
    py = 0
    Max_distance = 0
    print('\nCalculating the Maximum distance in ', str1, '...')

    distance_list = list()
    for i in range(len(temp)):
        temp_px = temp.iloc[i]['x']
        temp_py = temp.iloc[i]['y']
        distance = GetCD(x1, y1, x2, y2, temp_px, temp_py)
        distance_list.append(distance)
        if distance > Max_distance:
            px = temp_px
            py = temp_py
            Max_distance = distance
    print('The calculation is Successful')
    print ('Point A is', x2, ',', y2)
    print ("Point B is", x1, ',', y1)
    print ('Point C is', px, ',', py)
    array_ab = np.array([x1 - x2, y1 - y2])
    array_ac = np.array([px - x2, py - y2])
    array_oa = np.array([x2, y2])
    array_ob = np.array([x1, y1])
    array_oc = np.array([px, py])
    AB = GetDistance(x1, y1, x2, y2)
    AD = array_ab.dot(array_ac) / AB
    array_ad = ((array_ab) / AB) * AD
    array_d = array_oa + array_ad
    BD = GetDistance(x1, y1, array_d[0], array_d[1])

    print ('Point D is', array_d[0], ',', array_d[1])
    print (' CD:', Max_distance)
    print (' AB:', AB)
    print (' CD/AB:', Max_distance / AB)
    print (' BD/AB:', BD / AB)
    result = result.append(
        {'word': str1, 'C': array_oc, 'D': array_d, 'CD': Max_distance, 'AB':AB, 'CD/AB': Max_distance / AB,
         'BD/AB': BD / AB, 'MCI': mci, 'real_1': rl[1], 'imag_1': im[1],
         'mod_1': mod[1], 'real_2': rl[2], 'imag_2': im[2], 'mod_2': mod[2], 'real_3': rl[3], 'imag_3': im[3],
         'mod_3': mod[3]}, ignore_index=True)
    return result


def GetCD(x2, y2, x1, y1, px, py):
    array_oa = np.array([x2, y2])
    array_ob = np.array([x1, y1])
    array_oc = np.array([px, py])
    d = np.linalg.norm(np.cross(array_oa - array_ob, array_ob - array_oc)) / np.linalg.norm(array_oa - array_ob)
    return d


def GetDistance(x2, y2, x1, y1):
    distance = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return distance


#Attribute: kdawson2/tshape_analysis
def curvature_index(data):

  # compute signed curvature
  dx = np.gradient(data[:,0])
  dy = np.gradient(data[:,1])
  ddx = np.gradient(dx)
  ddy = np.gradient(dy)
  cur = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5;

  s = np.cumsum(np.sqrt(np.sum(np.diff(data,axis=0)**2,axis=1)))
  s = np.insert(s,0,0)

  b, a = butter(5,1./4.)
  n = len(data)
  r = cur[::-1]
  fcur = filtfilt(b, a, np.concatenate((r,cur,r)))
  fcur = fcur[n:-n]

  fcurA = np.abs(fcur)
  # mci = simps(fcurA,s)
  mci = integrate.simps(fcurA,s)

  return mci

#Attribute: kdawson2/tshape_analysis
def fourier_analysis(data):

  ta = np.arctan2(np.gradient(data[:,1]),np.gradient(data[:,0]))

  ntfm = np.fft.rfft(ta)

  rl = np.real(ntfm)
  im = np.imag(ntfm)
  mod = np.absolute(ntfm)

  return rl, im, mod

try:
    if len(sys.argv)>1:
        print ('Reading Document',sys.argv[1],"...")
        data=pd.read_csv(sys.argv[1])
        
    else:
        print ("\nReading Document...")
        data= pd.read_csv('p2-sample.csv')
         
except IOError:
    print ("Error: Pleas check the filename.")
else:
    print ('Read successfully')
    print ('Data types \n',data.dtypes)
    print ('Data shape: ',data.shape)
    result=pd.DataFrame(columns=['word','C', 'D', 'CD', 'AB','CD/AB', 'BD/AB', 'MCI','real_1','imag_1','mod_1','real_2','imag_2','mod_2','real_3','imag_3','mod_3'])
    column=float(data.shape[0])
    if column>0:
        times=int(column/42)
        start=0
        end=41
        i=0
        while i < times:
            result=Curve(start,end,result)
            start+=42
            end+=42
            i +=1
    if len(sys.argv)>2:
        result.to_csv(sys.argv[2])
    else:    
        result.to_csv('p2-result.csv')
