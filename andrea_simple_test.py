#TRYING TO APPLY THE CODE TO CHANNEL MODEL 
#%%
import numpy as np
import gempy as gp


#! DEFINE INTERFACE POINTS FOR START GEOMODEL
ipoints = 4
z = np.zeros(ipoints) + 20  # Control Points in Z direction +
zdown = z * -1  # Control Points in Z direction
x = np.linspace(50,350,len(z))
# x = np.zerps(ipoints)+100
y = np.zeros(ipoints) + 50
interface_stacked = np.stack((x, y, z)).T


#%%
#! CREATING START GEOMODEL
extent = [0, 400, 0, 100, -50, 0]
resolution = [50, 50, 25]
surfaces = [ "regolith", "basement"]



def make_gempy_model(interface, surfaces, extent=[0, 400, 0, 100, -50, 0], resolution=[50, 50, 25], make3d=True, plot3d=False):
    """_summary_

    Args:
        interface (list, [x,y,z] numpy array): [x,y,z interface points that create the interfaces]
        surfaces (list, 'str'): list of strings that correspond to layer names. 
        extent (list, optional): _description_. Defaults to [0, 400, 0, 100, -50, 0].
        resolution (list, optional): _description_. Defaults to [50, 50, 25].
        make3d (bool, optional): _description_. Defaults to True.
        plot3d (bool, optional): _description_. Defaults to False.
    """
    # Creating geomodel
    geomodel = gp.create_model('geomodel')
    gp.init_data(geomodel, extent=extent,
             resolution=resolution)
    geomodel.add_surfaces(surfaces)
    # Adding surface points
    for x, y, z in interface:
         z = z*-1
         geomodel.add_surface_points(X=x, Y=y, Z=z, surface=surfaces[0])

    geomodel.add_orientations(X=170, Y=40, Z=0,
                          surface=surfaces[0], orientation=[90, 0, 1])  
                          
    gp.set_interpolator(geomodel, compile_theano=True,
                    theano_optimizer="fast_compile")
    sol = gp.compute_model(geomodel)
    gempymesh = sol.s_regular_grid 
    if plot3d:
        plotter = gp.plot_3d(geomodel, plotter_type='background')
        plotter.plot_structured_grid()
        pass
    sol = gp.compute_model(geomodel)
    return (geomodel, sol)
#%%

geo_model, sol = make_gempy_model(interface_stacked, surfaces, make3d=True)
#%%
gp.plot_3d(geo_model)

#%%
import copy
fault_ind=[]
repre_pts=[]

#     Initialization of the Gempy model
df_int_X      = copy.copy(geo_model.surface_points.df['X'])
df_int_Y      = copy.copy(geo_model.surface_points.df['Y'])
df_int_Z      = copy.copy(geo_model.surface_points.df['Z'])
df_or_X       = copy.copy(geo_model.orientations.df['X'])
df_or_Y       = copy.copy(geo_model.orientations.df['Y'])
df_or_Z       = copy.copy(geo_model.orientations.df['Z'])
df_or_dip     = copy.copy(geo_model.orientations.df['dip'])
df_or_azimuth = copy.copy(geo_model.orientations.df['azimuth'])#%%
geo_model.solutions.lith_block = np.round(geo_model.solutions.lith_block)
fault_ind.append (sol.fault_block.T[0:sol.grid.values.shape[0]]) # %%

layer_surfpoints = geo_model.surface_points.df.surface.isin(['unit_1']) #GETTINGS THE SURFACE POINTS PER LAYER
indexes_La_sp = geo_model.surface_points.df[layer_surfpoints].index #GETTING THE INDEXES OF THE SURFACE POINTS EPR LAYER 
layer_orient = geo_model.orientations.df.surface.isin(['unit_1'])
index_La_o = geo_model.orientations.df[layer_orient]
ver = list(geo_model.solutions.vertices)

# %%
grid=geo_model.grid.values
z_resolution = abs (grid[0,-1] - grid[1,-1])
lays_fault_name=geo_model.surface_points.df.loc[:, 'surface'].unique()
all_vers=ver #FLIPS AND NOW THERE ARE 4 LAYERS WHERE EACH HAS 10 ITERATIONS 

# #TODO clean up vers so that all_vers[2] doesn't have values higher than all_vers[1]
# for i, vert in enumerate(all_vers[1:]):
#     rows_to_remove = np.where(all_vers[i+1][:, 2] > np.max(all_vers[i][:, 2]))
# # Eliminate the rows from array1
#     all_vers[i+1] = np.delete(all_vers[i+1], rows_to_remove, axis=0)


no_layers = len(all_vers)
df=geo_model.series.df
if len (np.unique (sol.fault_block))>1:
    no_of_faults=df.groupby(by='BottomRelation').count().iloc[1,0]
else:
    no_of_faults=0
len_rough = np.array([[len(all_vers[y])] for y in range(len(all_vers))])
np.savetxt('len_rough.csv', len_rough, delimiter=',')

rough_vers = []
for d in all_vers:
    rough_vers.append(d.tolist())

#%%
for ind, vertices in enumerate(rough_vers):
    np.savetxt(f'rough_vertices_{ind}.csv', np.array(vertices), delimiter=',')
np.savetxt('extent.csv', extent, delimiter=',')
np.savetxt('z_resolution.csv', np.array([z_resolution]), delimiter=',')

# %%
import glob
from numpy import genfromtxt
import numpy as np

extent = genfromtxt('extent.csv', delimiter=',') # comes from Gempy export cell
z_resolution = genfromtxt('z_resolution.csv', delimiter=',') # comes from Gempy export cell or can be easily defines
# by the user. this value is an important one because it should be based on the expected mesh size around your surface
len_rough = genfromtxt('len_rough.csv', delimiter=',') # comes from Gempy export cell
all_roughs = []
files_ve = glob.glob("rough_vertices_*.csv")
files_ver = sorted(files_ve, key=lambda name: int(name[15:-4]))
for name_ver in files_ver:
    data = genfromtxt(name_ver, delimiter=',')
    all_roughs.append(data)
#all_roughs = [[i] for i in all_roughs]
all_vers = []
for vert in all_roughs:
    new_arrs = np.split(vert, np.cumsum(len_rough).astype('int').tolist())[:-1]
    all_vers.append(new_arrs)
no_of_faults = 0 # can be also exported from Gempy export cell
n_iter = 10 # can be also exported from Gempy export cell
name_of_layers = geo_model.surfaces # can be also exported from Gempy export cell
resolution = np.array([50,50,20]) # can be also exported from Gempy export cell


# %%
from GeoMeshPy import vmod # the class vmod allows you for doing all the required calculations

fr = np.array([[]])
mesh_rosultion = 5
no_of_faults = 0
model = vmod.vertice_modifier(1, 0, all_vers, name_of_layers, mesh_rosultion, fr, extent, resolution)
new_result_list = model.contact_generator()[0]
length_layers = model.contact_generator()[1]
repre_pts = model.contact_generator()[2]
# %%

# 2D visualization Cell --> This plot visualizes the redundant irregular vertices cretaed in Gempy
import matplotlib.pyplot as plt

Gempy_out_1 = all_vers[0][0] # Vertices of Gempy for first contact in first iteration

First_iter_cl = np.array(new_result_list[0]) # vertices exported from GeoMeshPy in the first iteration
length = np.array(length_layers[0]).astype('int')# it recals the lengths of conatcts of the layers
le = np.cumsum (length) # accumulative sime of lengths

x = First_iter_cl[:le[0],0]
y = First_iter_cl[:le[0],1]

fig, axs = plt.subplots(1, 1, figsize=(10.,10.))
axs.scatter(x, y, c = 'r', s = 20, label='GeoMeshpy_output', facecolor='None', linewidths=1.1)
axs.scatter(Gempy_out_1[:,0], Gempy_out_1[:,1], c = 'b', s = 1.5, label='GemPy_output')
axs.legend()
plt.show()

# %%
# 3D visualization Cell --> This plot visualizes the redundant irregular vertices created in Gempy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

length = np.array(length_layers[0]).astype('int')# it recals the lengths of conatcts of the layers
le = np.cumsum (length) # accumulative sime of lengths

x = First_iter_cl[:,0]; y = First_iter_cl[:,1]; z = First_iter_cl[:,2]

x_contact_1=x[:le[0]]; y_contact_1=y[:le[0]]; z_contact_1=z[:le[0]]
x_contact_2=x[le[0]:le[1]]; y_contact_2=y[le[0]:le[1]]; z_contact_2=z[le[0]:le[1]]
x_contact_3=x[le[1]:le[2]]; y_contact_3=y[le[1]:le[2]]; z_contact_3=z[le[1]:le[2]]

Gempy_out_2 = all_vers[1][0] # Vertices of Gempy for second contact in first iteration
Gempy_out_3 = all_vers[2][0] # Vertices of Gempy for third contact in first iteration

reps = np.array(repre_pts[0])[:,:-1].astype('float') # these point are where we are sure about the formation


fig = plt.figure()
ax = fig.add_subplot (111, projection="3d")

ax.scatter3D(x_contact_1, y_contact_1, z_contact_1, color='b', s=15, label='First contact', facecolor='None', linewidths=1)
ax.scatter3D(x_contact_2, y_contact_2, z_contact_2, color='m', s=15, label='Second contact', facecolor='None', linewidths=1)
ax.scatter3D(x_contact_3, y_contact_3, z_contact_3, color='r', s=15, label='Third contact', facecolor='None', linewidths=1)

ax.scatter3D(Gempy_out_1[:,0], Gempy_out_1[:,1], Gempy_out_1[:,2], color='k', s=1, label='Gempy output')
ax.scatter3D(Gempy_out_2[:,0], Gempy_out_2[:,1], Gempy_out_2[:,2], color='k', s=1)
ax.scatter3D(Gempy_out_3[:,0], Gempy_out_3[:,1], Gempy_out_3[:,2], color='k', s=1)

ax.scatter3D(reps[:,0], reps[:,1], reps[:,2], color='r', s=50, marker = '*', label='representative points')

plt.legend()
ax.grid(None)
plt.show()

#%%
middle_point = (50, 50)
# Create a list of points along a straight line passing through the middle point
num_points = 25
line_length = 75
line_points = [(i * line_length/(num_points-1) + middle_point[0] - line_length/2, middle_point[1]) for i in range(num_points)]
import math
# Define a function to rotate a point around the origin
def rotate_point(point, angle):
    x, y = point
    rad = math.radians(angle)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)
    x_rot = x * cos_theta - y * sin_theta + middle_point[0] - middle_point[0] * cos_theta + middle_point[1] * sin_theta
    y_rot = x * sin_theta + y * cos_theta + middle_point[1] - middle_point[0] * sin_theta - middle_point[1] * cos_theta
    z_rot = np.zeros_like(x_rot) + 10
    return (x_rot, y_rot, z_rot)

# Create a dictionary to store the rotated points
rotated_lines = {}
# Rotate the line by 10 degrees and store the points in the dictionary
listangles = np.linspace(0, 360, 13)
for angle in listangles:
    rotated_points = [rotate_point(p, angle) for p in line_points]
    rotated_lines[angle] = rotated_points

# PLOTTING THE LINES 
for angle, points in rotated_lines.items():
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    plt.scatter(x_vals, y_vals, label=f'Rotated {angle}°')

# Add legend and labels
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rotated Lines')
plt.show()



# %%

import gmsh
import itertools
from collections import defaultdict
from itertools import chain

def cleanup_and_mesh(entities_to_preserve):
    # remove all embedded constraints, i.e. the entities that are not on the
    # boundary of others
    entities = gmsh.model.getEntities()
    for e in entities:
        emb = gmsh.model.mesh.getEmbedded(e[0], e[1])
        gmsh.model.mesh.removeEmbedded([e])
        for p in entities_to_preserve:
            if p in emb:
                gmsh.model.mesh.embed(p[0], [p[1]], e[0], e[1])
    # remove all surfaces, curves and points that are not connected to any
    # higher-dimensional entities
    gmsh.model.removeEntities(gmsh.model.getEntities(2), True)
    cc = gmsh.model.getEntities(1)
    for c in curves_to_preserve:
        print(c)
        cc.remove(c)
    gmsh.model.removeEntities(cc, True)
    gmsh.model.removeEntities(gmsh.model.getEntities(0))
    # get all surfaces that are not of type "Plane", i.e. all surfaces except the

        # box
    surfaces = [s[1] for s in gmsh.model.getEntities(2) if gmsh.model.getType(s[0], s[1])
                != 'Plane']
    layers_contacts = np.array(surfaces).reshape(-1,1).tolist()
    surface_after = gmsh.model.getEntities(2)
    points = copy.deepcopy(surface_new_tag)
    check_values = [row[-1] for row in surface_after]
    extracted = []
    for sublist in points:
        second_vals = [sec for fir, sec in sublist]
        if all(val in check_values for val in second_vals):
            extracted.append(second_vals)
    
    surrounding_box = [x for x in extracted if x not in layers_contacts]

    # create a distance + threshold mesh size field
    gmsh.model.mesh.field.add("Distance", 1) # refiement close to the well
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", np.array(curves_to_preserve)[:,1].tolist())
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 2)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 20)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 6)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 30)
    
    gmsh.model.mesh.field.add("Distance", 3) # refiement close to contacts of layers
    gmsh.model.mesh.field.setNumbers(3, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(3, "Sampling", 100)
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", 20)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 10)
    gmsh.model.mesh.field.setNumber(4, "DistMax", 30)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 30)

    # don't extend mesh sizes from boundaries and use new 3D algo
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)

    # using representative points to create physical volumes
    rep = [list(x) for _,x in itertools.groupby(rep_pnt,lambda x:x[3])]
    vol_num = np.arange(1, 1+len(rep))
    for ind, surfaces in enumerate (rep):
        tags = []
        for sects in surfaces:
            eleTag = gmsh.model.mesh.getElementByCoordinates(float (sects[0]), float (sects[1]), float (sects[2]))[0]
            eleType, eleNodes, entDim, entTag = gmsh.model.mesh.getElement(eleTag)
            tags.append(entTag)
        gmsh.model.addPhysicalGroup(3, tags, vol_num[ind])
        gmsh.model.setPhysicalName(3, vol_num[ind], sects[-1])  
        
    # adding wells as physical lines
    for lines in sp_well:
        gmsh.model.addPhysicalGroup(1, lines.tolist())
        #gmsh.model.setPhysicalName(1, l1, well_name)
        
    # adding surrounding surfaces as physical surfaces to be useable as boundary conditions
    around_box = ['in', 'out', 'front', 'back', 'bottom', 'top']       
    for tag_nu, name in zip (surrounding_box, around_box):
        ps1 = gmsh.model.addPhysicalGroup(2, tag_nu)
        gmsh.model.setPhysicalName(2, ps1, name)
    gmsh.write("Channel_" + str(kk) + ".msh")
    gmsh.fltk.run()


#%%
gmsh.clear()
gmsh.initialize()
degree = 3
numPointsOnCurves = 10
numIter = 1
anisotropic = False
tol2d = 0.001
tol3d = 1
tolAng = 1
tolCurv = 10
maxDegree = 3
maxSegments = 100
sensor_coords = rotated_lines.get(90)
sets = [new_result_list, repre_pts, sensor_coords]
#There is no need for an outer loop because there are no more iterations, just one. 
ar = np.array(new_result_list)[0]
l_tags = []
gmsh.model.occ.addBox(min(ar[:,0]),min(ar[:,1]),extent[-2],
                        max(ar[:,0])-min(ar[:,0]),max(ar[:,1])-min(ar[:,1]),extent[-1])
spl_num = np.cumsum(length_layers).tolist()[:-1] # each layer is separated
spl_num = [int(i) for i in spl_num]
sep_ar = np.split(ar,spl_num)

for ind, point_clouds in enumerate(sep_ar):
    point_clouds=point_clouds[np.lexsort((point_clouds[:,1],point_clouds[:,0]))]
    i_l = point_clouds.tolist()
    for k, [x, y, z] in enumerate(i_l):
        point_xyz = gmsh.model.occ.addPoint(x, y, z)
        print(point_xyz)
    corn_layers= point_clouds[(point_clouds[:,0] == np.min(point_clouds[:,0])) | 
                    (point_clouds[:,0] == np.max(point_clouds[:,0])) |
                    (point_clouds[:,1] == np.min(point_clouds[:,1])) |
                    (point_clouds[:,1] == np.max(point_clouds[:,1]))]
    corn_layers = corn_layers[np.lexsort((corn_layers[:,1],corn_layers[:,0]))]
    h = corn_layers
    pnt = h[:,0:-1].tolist()
    arround_pts = model.rotational_sort(pnt, (np.mean(np.array(pnt)[:,0]),np.mean(np.array(pnt)[:,1])),True)
    arround_pts = np.array(arround_pts)
    tags = np.where((point_clouds[:,:-1] == arround_pts[:,None]).all(-1))[1]+1
    l_tags.append(len(tags)) #length of tags 
    start_point = int(8+np.sum(length_layers[0][0:ind]))  #the ind just starts naming the points again for the next layer 
    start_line = int(13+np.sum(l_tags[0:ind])) #same here as the line tags 
    for i in range(len(tags)): # this for loop creates the exterior lines of each cloud
        if i!=len(tags)-1:
            gmsh.model.occ.addSpline([tags[i] + start_point, tags[i+1] + start_point])
        else:
            gmsh.model.occ.addSpline([tags[i] + start_point,tags[0] + start_point])
    gmsh.model.occ.addCurveLoop([i for i in range (start_line, start_line + len(tags))], start_line)
    gmsh.model.occ.addSurfaceFilling(start_line, start_line,
                                        [m for m in range (start_point+1, start_point+np.max(tags))
                                        if m not in tags + start_point],
                                        degree,
                                        numPointsOnCurves,
                                        numIter,
                                        anisotropic,
                                        tol2d,
                                        tol3d,
                                        tolAng,
                                        tolCurv,
                                        maxDegree,
                                        maxSegments) # create surface by connecting exterior lines
                                                                        # and inclding interior ones

gmsh.model.occ.synchronize()
in_surf = gmsh.model.occ.getEntities(2)

#%%
#Add sensors 
newtags = []
for k, [x, y, z] in enumerate(sensor_coords):
     tagsn = gmsh.model.occ.addPoint(x, y, z, 0.001)
     newtags.append(tagsn)

#%%
# Importing Wells:
    #if the fragment operation is applied to entities of different dimensions, 
    # the lower dimensional entities will be automatically embedded in the higher 
    # dimensional entities if they are not on their boundary.



# #tag_well = np.arange(10000, 10000+len(np.array(well_cord)))
# well_p = np.array(well_p[0]).astype('int')
# tag_well_po = np.split (tag_well, np.cumsum(well_p)[:-1])
# well_po = np.split (np.array(well_cord), np.cumsum(well_p)[:-1])  
# for cord, tag_nu in zip (well_po, tag_well_po):
#     for [x, y, z], num in zip (cord,tag_nu):
#         gmsh.model.occ.addPoint(x, y, z, tag=num)
# well_l = well_p-1        
# tag_w_l = np.arange(5000, 5000+np.sum(well_l))
# wl = [i.tolist() for i in tag_w_l]
# tag_well_l = np.split (tag_w_l, np.cumsum(well_l)[:-1])

# for po, tag_num in zip (tag_well_po, tag_well_l):
#     for i in range (len(po)-1):
#         gmsh.model.occ.addLine (po[i], po[int(i+1)], tag = tag_num[i])
# in_wells = [(1, i) for i in tag_w_l]
# out_all = gmsh.model.occ.fragment(in_surf+in_wells, gmsh.model.occ.getEntities(3))#[1]
# out = out_all[1]
# surface_new_tag = out[0:len(in_surf)]
# c = out[len(in_surf):len(in_surf+in_wells)]
# curves_to_preserve = [item for sublist in c for item in sublist]           
# gmsh.model.occ.synchronize()
# points_to_preserve = gmsh.model.getBoundary(curves_to_preserve, combined=False)
# line_sp = np.array([])
# for i in range (len(points_to_preserve)-1):
#     if i%2 != 0:
#         if points_to_preserve[i][1] != points_to_preserve[i+1][1]:
#             brk=int ((i+1)/2)
#             line_sp=np.append(line_sp, brk)
# sp_well = np.split(np.array(curves_to_preserve)[:,1],line_sp.astype('int'))
# cleanup_and_mesh(curves_to_preserve + points_to_preserve)
# gmsh.clear()
# gmsh.finalize()

# tag_well = np.arange(10000, 10000+len(np.array(sensor_coords)))
# newtags = []
# for k, [x, y, z] in enumerate(sensor_coords):
#      tagsn = gmsh.model.occ.addPoint(x, y, z, 0.001)
#      newtags.append(tagsn)

# out_all = gmsh.model.occ.fragment(in_surf, gmsh.model.occ.getEntities(3))#[1]
# out = out_all[1]
# surface_new_tag = out[0:len(in_surf)]

c = out[len(in_surf):(len(in_surf)+len(sensor_coords))]
curves_to_preserve = [item for sublist in c for item in sublist]           
gmsh.model.occ.synchronize()
points_to_preserve = gmsh.model.getBoundary(curves_to_preserve, combined=False)
line_sp = np.array([])
for i in range (len(points_to_preserve)-1):
    if i%2 != 0:
        if points_to_preserve[i][1] != points_to_preserve[i+1][1]:
            brk=int ((i+1)/2)
            line_sp=np.append(line_sp, brk)

sp_well = np.split(np.array(curves_to_preserve)[:,1],line_sp.astype('int'))
cleanup_and_mesh(curves_to_preserve + points_to_preserve)
gmsh.finalize()


# %%
