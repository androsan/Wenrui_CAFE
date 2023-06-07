import numpy as np
import matplotlib.pyplot as plt
plt.ion()


folder_where_you_keep_your_files = "C:/npy files from ODB conversion/"
folder_where_you_keep_your_files = 'C:/Users/andraz.kocjan/Desktop/my private Apps/'

raw_data_from_Abaqus = np.load(folder_where_you_keep_your_files+'TEMPERATURE_2D.npy')
time_frames = raw_data_from_Abaqus.shape[0]
all_nodes = raw_data_from_Abaqus.shape[1]

print('Matrix converted from .ODB to .NPY is a 2D matrix. 1st dimension is number of time iterations: '+str(time_frames)+
' whereas the second dimensions is number of ALL nodes in a domain: '+str(all_nodes)+' This is so-called flattened matrix, which must be'+
'shaped back to 3D domain..')


# RESHAPING ----> conversion of flattened 1D matrix to a 3D USING   ::: Numpy.reshape  command
# To do this, we have to know two things:

# 1. Values of nodes in X,Y,Z directions

Z = 3
Y = 6
X = 76

# 2. Right order (combination) of X,Y,Z nodes in the following formula:

shaped_3D_matrix_ready_for_interpolation = np.reshape(raw_data_from_Abaqus,(raw_data_from_Abaqus.shape[0],X,Z,Y))

# This combination can be guessed, if you are not sure. There are only 6-combinations or so..
# To check, if you have the right combination you plot of  plane through laser spot and it should give you nice picture as you get it on Abaqus UI.
# Not some defragmented, blurred figure..

time_step = 555
melting_point = 1100


#plt.imshow(shaped_3D_matrix_ready_for_interpolation[time_step,:,0,:])   # temperature

plt.imshow(shaped_3D_matrix_ready_for_interpolation[time_step,:,0,:]>melting_point)   # phase



'''

You have the process of RESHAPING and TRANSPOSING coded and described in the INTERPOLATION code I sent you last time..

'''


