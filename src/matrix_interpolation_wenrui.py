# -*- coding: utf-8 -*-
''' Program for decomposition of input 4D matrix into several 3D matrices (domain) for each time step - a single .npy file 
    and linear interpolation of input matrix to achieve higher spatial and time resolution

	Author:   Andraz Kocjan, May 2023
	Version:  2.0.0
	Description:  Supplemental module for postprocessing output data from FEM analysis.
'''

import os
import json
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random, math, time
plt.ion()


####################################### PRIVATE METHODS #####################################


''' domain limits ----> reduction of FE domain to size a bit larger than the molten track, where the crystallization occurs .. '''
#
def Domain_Size_Reduction(domain, threshold):
   global z_min, z_max, x_min, x_max, y_min, y_max
   ax2=np.where(np.any(domain >= threshold, axis=2))
   ax1=np.where(np.any(domain >= threshold, axis=1))
   z_min = np.min(ax2[0]);  z_max = np.max(ax2[0])+1
   x_min = np.min(ax2[1]);  x_max = np.max(ax2[1])+1
   y_min = np.min(ax1[1]);  y_max = np.max(ax1[1])+1
   
   reduced_domain = domain[z_min:z_max, x_min:x_max, y_min:y_max]
   return reduced_domain



#
def f(x, tresh):
    T = np.load(PATH_work+mapa+'/fem_'+str(x)+'.npy')
    P = np.zeros(T.shape)
    P[T >(tresh+273)]=1
    return T, P

#
def p(t_field, tresh):
	p=np.zeros(t_field.shape)
	p[t_field>(tresh+273)]=1
	plt.imshow(p)

#
def Matrix_Interpolation(A, it):
   
   #B_tuple = (it[0], (z_max-z_min)*it[1], (x_max-x_min)*it[2], (y_max-y_min)*it[3])
   B_tuple = (it[0], A.shape[1]*it[1], A.shape[2]*it[2], A.shape[3]*it[3])
   new_dims = []
   for original_length, new_length in zip(A.shape, B_tuple):
      new_dims.append(np.linspace(0, original_length-1, new_length))

   coords = np.meshgrid(*new_dims, indexing='ij')
   B = map_coordinates(A, coords, order=1)
   return B

#
def TimeScaleCounter(tf,sf,tr):
   tsc = [i for i in range(tr[0]*tf*sf, tr[1]*tf*sf)]
   return tsc

def Make_Y_Partitions(Ymin, Ymax, slices):
    dDY = (Ymax-Ymin)/slices
    a=[Ymin+i*dDY for i in range(slices)]
    b=[Ymin+i*dDY+1 for i in range(1,slices+1)]
    YP={}
    for num,(i,j) in enumerate(zip(a,b)):
        YP['YP'+str(num)] = (int(i), int(j))
    return YP


#############################################################################################

#**** SETUP ****#

# GET current directory and replace backslash with forward slash
curDir = os.getcwd().replace("\\", "/")

# Check, if current directory is set as workDirectory in settings.json
with open("settings.json", "r", encoding='utf-8-sig') as readJson:
	appSettings = json.load(readJson)

if appSettings["workDirectory"] != curDir:
    print();print();print(80*"~");print()
    print("Welcome to Matrix_Interpolation 2.0.0, written by Andraz Kocjan, May 2023"); print(); print(80*"~");print()
    print("Setting up the environment.. Please, stand by..");print()



    appSettings["workDirectory"] = curDir
    with open('settings.json', 'w', encoding='utf-8-sig') as writeJson:
        json.dump(appSettings, writeJson, ensure_ascii=False)
        print("workDirectory SET "+50*"."+" OK..")


# Check, if required folders exist in current (work) directory
if not os.path.isdir(curDir+"/WORK/"):
   os.mkdir(curDir+"/WORK/")
   print("WORK folder CREATED "+50*"."+" OK..")
if not os.path.isdir(curDir+"/OUTPUT_data/"):
   os.mkdir(curDir+"/OUTPUT_data/")
   print("OUTPUT_data folder CREATED "+50*"."+" OK..")

if not os.path.isdir(curDir+"/INPUT_data/"):
   os.mkdir(curDir+"/INPUT_data/")
   print("INPUT_data folder CREATED "+50*"."+" OK..")
   # -TODO- input, and create README
   print();print()
   print("Please, insert (1 ONLY!) .NPY file with FEM data, which was converted from .ODB, into folder 'INPUT_data'")
   print();
   input(20*" "+"PRESS ENTER to CONTINUE..")
#*** END of SETUP ***# 

'''--------------------------Reading settings-------------------------------------------------<START>'''
#
with open("settings.json", "r", encoding='utf-8-sig') as readJson:
	appSettings = json.load(readJson)

PATH = appSettings["workDirectory"]    # APPLICATION DIRECTORY
PATH_input = PATH+"/INPUT_data/"       # INPUT_data directory
PATH_work = PATH+"/WORK/"              # WORK directory
PATH_output = PATH+"/OUTPUT_data/"     # OUTPUT_data directory

FEM = appSettings["FEM parameters"]             # FEM analysis parameters
INTER = appSettings["matrix interpolation"]     # INTERPOLATION parameters
'''-------------------------------------------------------------------------------------------<END>'''
#
#
#
#
#
'''--------------------------Input data validation--------------------------------------------<START>'''

# 1. Checks, if there is exactly one input .npy file in the INPUT_data folder. Else, exceptions are raised:
inputs = os.listdir(PATH_input)
file_2d = ""
npy_counter = 0
for i in inputs:
	if i.endswith(".npy"):
		npy_counter+=1
		file_2d += i
	if npy_counter > 1:
		raise Exception("Only 1 .NPY input FEM data file is allowed in INPUT_data folder!")

if npy_counter == 0:
	raise Exception("No .NPY input FEM data file found in INPUT_data folder.")

# Read the domain size data from settings.json:
X = FEM["domain size"]["axis-3"]
Y = FEM["domain size"]["axis-1"]
Z = FEM["domain size"]["axis-2"]

array_2d = np.load(PATH_input+file_2d)  # Reading input data from disk
frames = array_2d.shape[0]  # Number of FEM frames (e.g. time steps)


# This is the tricky part! After FEM in Abaqus, correct combination of X,Y,Z must be obtained for 'reshape' command: 
array_4d = array_2d.reshape((frames, Y, Z, X))
# After that transposition of axes must be done in order to use CA code properly:
''' Explanation:

                         matrix         |   after FEM (Abaqus)   |    after transposition (CORRECT)    |
 =======================================|========================|=====================================|
                         time frames    |          0             |                  0                  |
----------------------------------------|------------------------|-------------------------------------|
          parallel to laser (Z-axis)    |          2             |                  1                  |
          track cross-section (X-axis)  |          3             |                  2                  |
          along the track (Y-axis)      |          1             |                  3                  |
========================================================================================================
'''
Matrix_4D_not_flipped = array_4d.transpose(0,2,3,1)   # transposition of axes

Matrix_4D = np.flip(Matrix_4D_not_flipped, axis=3)

np.save("C:/temp/Matrix_4D.npy", Matrix_4D)

# 2. Checks, if input .npy file has the right 4d dimensionality, else exception is raised:
if Matrix_4D.ndim != 4:
	raise Exception("The input Numpy array does not have 4 dimensions.")
'''-------------------------------------------------------------------------------------------<END>'''
#
#
#
#
#
'''--------------------------Increasing the resolution by linear interpolation----------------<START>'''

space_factor = INTER["space factor"] # Cell size decrease factor (higher resolution)
FEM_time_factor = 1 # FEM time factor to make FEM analysis shorter ---> for this factor , i.e. diminish number of time steps
extra_time_factor = 1  # ::: special time factor ::: to catch the mesh dependency effect !! !  !
time_factor = FEM_time_factor * extra_time_factor
increase_tuple  = (time_factor * space_factor, space_factor, space_factor, space_factor)  # resolution increase factor by linear interpolation
Tmelt_Celsius = FEM["material properties"]["melting point"] # Melting point, UNIT:  degrees C [deg. C]
Tmelt = Tmelt_Celsius + 273.15 #   Melting point, :  KELVIN [K]
N = INTER["number of partitions"]    # Number of equally sized partitions of the track

x_min, x_max, y_min, y_max, z_min, z_max = 0, X, 0, Y, 0, Z


#limits = [ [], [], [], [], [], [] ]
#for i in range(total_time_range[0], total_time_range[1]):
#    Domain_Size_Reduction(Matrix_4D[i], Tmelt)
#    limits[0].append(z_min); limits[1].append(z_max)
#    limits[2].append(x_min); limits[3].append(x_max)
#    limits[4].append(y_min); limits[5].append(y_max)

#z_min, z_max, x_min, x_max, y_min, y_max = min(limits[0]), max(limits[1]), min(limits[2]), max(limits[3]), min(limits[4]), max(limits[5])

#z_max = 1

y_min = y_min if INTER["domain limits"]["axis-1_limits"][0] == None else INTER["domain limits"]["axis-1_limits"][0]
y_max = y_max if INTER["domain limits"]["axis-1_limits"][1] == None else INTER["domain limits"]["axis-1_limits"][1]

z_min = z_min if INTER["domain limits"]["axis-2_limits"][0] == None else INTER["domain limits"]["axis-2_limits"][0]
z_max = z_max if INTER["domain limits"]["axis-2_limits"][1] == None else INTER["domain limits"]["axis-2_limits"][1]

x_min = x_min if INTER["domain limits"]["axis-3_limits"][0] == None else INTER["domain limits"]["axis-3_limits"][0]
x_max = x_max if INTER["domain limits"]["axis-3_limits"][1] == None else INTER["domain limits"]["axis-3_limits"][1]

YP = Make_Y_Partitions(y_min, y_max-1, N)

# Name of interpolated temperature fields main folder
mapa =   'INTER  time='+str(time_factor)+', space='+str(space_factor)+'  Z['+str(z_min)+'-'+str(z_max)+'], X['+str(x_min)+'-'+str(x_max)+'], Y['+str(y_min)+'-'+str(y_max)+'], '+str(Tmelt_Celsius)+'degC'+', N='+str(N)+'/'                  

if not os.path.isdir(PATH_work+mapa):
   os.mkdir(PATH_work+mapa)



for yp in list(YP)[INTER["start partition index"]:INTER["end partition index"]+1]:
   
    TR_list = []
    m4dm = Matrix_4D[:, z_min:z_max, x_min:x_max,YP[yp][0]:YP[yp][1]]

    for i in range(m4dm.shape[0]):
        if np.any(m4dm[i]>Tmelt):
            TR_list.append(i)
    TR_list.append(TR_list[-1]+1)

    start_total = time.time()
    
    for time_snap in range(len(TR_list)-1):
        time_range = (TR_list[time_snap], TR_list[time_snap+1])
        TSC = TimeScaleCounter(time_factor, space_factor, time_range)
        g = time_range[0]
        counter = 0


        for n in range(time_range[0], time_range[1]):
            yp_mapa = yp+'  ['+str(YP[yp][0])+','+str(YP[yp][1])+']'
            if not os.path.isdir(PATH_work+mapa+yp_mapa):
                os.mkdir(PATH_work+mapa+yp_mapa+'/')
            print(35*'~')
            print('Processing time step ',n,'/',len(TR_list),' ..')
            start = time.time()
            inp_mat = Matrix_4D[n:n+2, z_min:z_max, x_min:x_max,YP[yp][0]:YP[yp][1]]  # same sizes of y ranges, i.e. y_max-y_min,  as defined in YP dictionary 
            out_mat = Matrix_Interpolation(inp_mat,increase_tuple)
            if counter==0:
                od=None
            else:
                od=1
            del inp_mat

            for i in out_mat[1:]:
                try:
                    subfolder = '/TR'+str(n)+'  ['+str(n)+','+str(n+1)+']'
                    np.save(PATH_work+mapa+yp_mapa+subfolder+'/fem_'+str(TSC[counter]-g)+'.npy', i)
                except FileNotFoundError:
                    os.mkdir(PATH_work+mapa+yp_mapa+subfolder)
                    np.save(PATH_work+mapa+yp_mapa+subfolder+'/fem_'+str(TSC[counter]-g)+'.npy', i)
                counter+=1

            # for connecting indivudal TR folders, i.e. last .npy file from working directory (say TR0) puts into next (TR1), which is created if not yet..
            try:
                subf = '/TR'+str(n+1)+'  ['+str(n+1)+','+str(n+2)+']'
                np.save(PATH_work+mapa+yp_mapa+subf+'/fem_'+str(TSC[counter-1]-g)+'.npy', out_mat[-1])
            except FileNotFoundError:
                os.mkdir(PATH_work+mapa+yp_mapa+subf)
                np.save(PATH_work+mapa+yp_mapa+subf+'/fem_'+str(TSC[counter-1]-g)+'.npy', out_mat[-1])

            end = time.time()
            print('Computing time: ',round(end-start, 3),'  sec.')
            print(35*'~'); print()
        
    print('Total computing time =  ',round(time.time()-start_total, 3),'  seconds.')

















