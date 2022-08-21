import os
import math
import numpy as np
    # T = np.array([[math.cos(rx)*math.cos(ry),
    #         math.cos(rx)*math.sin(ry)*math.sin(rz) - math.cos(rx)*math.sin(rz),
    #         math.cos(rx)*math.sin(ry)*math.cos(rz) + math.sin(rx)*math.sin(rz), 
    #         tx],
            
    #         [math.sin(rx)*math.cos(ry),
    #         math.sin(rx)*math.sin(ry)*math.sin(rz) + math.cos(rx)*math.cos(rz),
    #         math.sin(rx)*math.sin(ry)*math.cos(rz) - math.cos(rx)*math.sin(rz),
    #         ty],

    #         [-math.sin(ry),
    #         math.cos(ry)*math.sin(rz),
    #         math.cos(ry)*math.cos(rz),
    #         tz],

    #         [0, 0, 0, 1]
            
    #         ])
def rtvec2matrix(rw, rx, ry, rz, tx, ty, tz):

    T = np.array([[
        1 - 2*ry**2 - 2*rz**2,
        2*rx*ry - 2*rz*rw,
        2*rx*rz + 2*ry*rw,
        tx],
    
        [2*rx*ry + 2*rz*rw,
        1 - 2*rx**2 - 2*rz**2,
        2*ry*rz - 2*rx*rw,
        ty],
    
        [2*rx*rz - 2*ry*rw,
        2*ry*rz + 2*rx*rw,
        1 - 2*rx**2 - 2*ry**2,
        tz],
    
        [0, 0, 0, 1]])
    return np.array(T)

# with open('images.txt') as f:
#     image_f = f.read()

# image_line = image_f.split('\n')
# image_doc = []
# for line in image_line:
#     image_doc.append(line.strip(' ').split(' '))

# count = 1
# file_cout = ""
# while count != 6:
    
#     for line in image_doc:
#         try:
#             if int(line[0]) == count:

#                 # print(line)
#                 # print('\n',count)
#                 T = rtvec2extrinsic(float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]))
#                 # print('\n'.join([' '.join([str(e) for e in row]) for row in T ]))
#                 cout = "{0} {0} {1}\n{2}\n".format(count-1,count,'\n'.join([' '.join([str(e) for e in row]) for row in T ]))
#                 print(cout)
#                 file_cout += cout

#                 break
#         except:
#             print("",end='')
#     count += 1

# with open('traj.log', 'w') as f:
#     f.write(file_cout)
