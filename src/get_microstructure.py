''' Program to build one final microstructure RGB image after Long_track_CA_NEW_wenrui has successfully finished,
    from partial RGB images (cuts) stored in cuts_RGB folder

    Author: Andraz Kocjan, June 2023
'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


'''--------------------------Reading settings-------------------------------------------------<START>'''
#
with open("settings.json", "r", encoding='utf-8-sig') as readJson:
	appSettings = json.load(readJson)

workDIR = appSettings["workDirectory"]             # APPLICATION DIRECTORY
PATH_work = workDIR+"/WORK/"                       # WORK directory
CA = appSettings["CA parameters"]                  # CELLULAR AUTOMATA parameters
mapa = CA["interpolated temperature fields"]+"/"   # folder with interpolated temperature matrices (defined in settings.json)



# Folders for systematic mesh dependency study
tracks_database = {
    'iso': ['2D 1st order Moore, iso field/', '2D 2nd order Moore, iso field/', '3D 1st order Moore, iso field/', '3D 2nd order Moore, iso field/'],
    'real': ['2D 1st order Moore, real field/', '2D 2nd order Moore, real field/', '3D 1st order Moore, real field/', '3D 2nd order Moore, real field/'],
}

track    =     tracks_database['real'][1]   # '2D 2nd order Moore, real field/'
cuts_RGB =     track+'cuts_RGB/'            #  Subfolder with cut 3D matrices, i.e. cuts



PATH = PATH_work+mapa+cuts_RGB
q= os.listdir(PATH)

a = []
for cut_rgb_file in q:
    a.append(np.load(PATH+cut_rgb_file))


microstructure_image = np.dstack(tuple(a))/255

plt.imshow(microstructure_image[0])