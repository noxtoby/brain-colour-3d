#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
import subprocess

mls2020 = "/Applications/meshlab.app/Contents/MacOS/meshlabserver" # Requires 2020.06 or earlier (meshlabserver executable removed: "Starting from MeshLab 2020.07, XML plugins won't compile and they cannot be loaded anymore.")
mls2016 = "/Applications/meshlab2016.app/Contents/MacOS/meshlabserver"
mls2014 = "/Applications/meshlab2014.app/Contents/MacOS/meshlabserver"
mls = mls2020

wd = os.path.curdir
df_ref = pd.read_csv(os.path.join(wd,"aparc.annot.ctab.csv"))

inner_outer = "brainpainter_outer.csv" # "brainpainter_outer.csv"
df_sustain_cdf = pd.read_csv(os.path.join(wd,inner_outer))

custom_roi_bool = False
if custom_roi_bool:
    roi_label = 'Custom ROI'
else:
    roi_label = 'ROI'

#* Choose a subtype
# subtypes = ['Typical','Cortical','Subcortical']
# subtype = subtypes[0]

#* FIXME: Choose cortical / subcortical view

#* Eline's
subtype_names = ['Subtype1','Subtype2']
subtype_name = subtype_names[0]

#* Select some stages
stages_sustain = [1,5,9,13,17,21,25,29,33]
stages = stages_sustain

cortAreasIndexMapDK = {
  'bankssts':                'TemporalLobe',
  'caudalanteriorcingulate': 'Cingulate',
  'caudalmiddlefrontal':     'FrontalLobe',
  'corpuscallosum':          -1,
  'cuneus':                  'OccipitalLobe',
  'entorhinal':              'TemporalLobe',
  'frontalpole':             'FrontalLobe',
  'fusiform':                'TemporalLobe',
  'inferiorparietal':        'ParietalLobe',
  'inferiortemporal':        'TemporalLobe',
  'insula':                  'Insula',
  'isthmuscingulate':        'Cingulate',
  'lateraloccipital':        'OccipitalLobe',
  'lateralorbitofrontal':    'FrontalLobe',
  'lingual':                 'OccipitalLobe',
  'medialorbitofrontal':     'FrontalLobe',
  'middletemporal':          'TemporalLobe',
  'paracentral':             'FrontalLobe',
  'parahippocampal':         'TemporalLobe',
  'parsopercularis':         'FrontalLobe',
  'parsorbitalis':           'FrontalLobe',
  'parstriangularis':        'FrontalLobe',
  'pericalcarine':           'OccipitalLobe',
  'postcentral':             'ParietalLobe',
  'posteriorcingulate':      'Cingulate',
  'precentral':              'FrontalLobe',
  'precuneus':               'ParietalLobe',
  'rostralanteriorcingulate':'Cingulate',
  'rostralmiddlefrontal':    'FrontalLobe',
  'superiorfrontal':         'FrontalLobe',
  'superiorparietal':        'ParietalLobe',
  'superiortemporal':        'TemporalLobe',
  'supramarginal':           'ParietalLobe',
  'temporalpole':            'TemporalLobe',
  'transversetemporal':      'TemporalLobe',
  'unknown':-1 # this is actually the middle region inside the cortical surface. color it as gray
}
subcortAreasIndexMap = {
  'Left-Accumbens-area':          'AccumbensArea',
  'Left-Caudate':                 'Caudate',
  'Left-Cerebellum-White-Matter':-1,
  'Left-Inf-Lat-Vent':-1,
  'Left-Pallidum':                'Pallidum',
  'Left-Thalamus-Proper':         'Thalamus',
  'Left-Amygdala':                'Amygdala',
  'Left-Cerebellum-Cortex':-1,
  'Left-Hippocampus':             'Hippocampus',
  'Left-Lateral-Ventricle':-1,
  'Left-Putamen':                 'Putamen',
  'Left-VentralDC':-1,
  'Right-Accumbens-area':          'AccumbensArea',
  'Right-Caudate':                 'Caudate',
  'Right-Cerebellum-White-Matter':-1,
  'Right-Inf-Lat-Vent':-1,
  'Right-Pallidum':                'Pallidum',
  'Right-Thalamus-Proper':         'Thalamus',
  'Right-Amygdala':                'Amygdala',
  'Right-Cerebellum-Cortex':-1,
  'Right-Hippocampus':             'Hippocampus',
  'Right-Lateral-Ventricle':-1,
  'Right-Putamen':                 'Putamen',
  'Right-VentralDC':-1
}

subcortAreasIndexMapEline = {
  'Left-Accumbens-area':          'Accumbens-area'          ,
  'Left-Caudate':                 'Caudate'                 ,
  'Left-Cerebellum-White-Matter': 'Cerebellum-White-Matter' ,
  'Left-Inf-Lat-Vent':            'Inf-Lat-Vent'            ,
  'Left-Pallidum':                'Pallidum'                ,
  'Left-Thalamus-Proper':         'Thalamus-Proper'         ,
  'Left-Amygdala':                'Amygdala'                ,
  'Left-Cerebellum-Cortex':       'Cerebellum-Cortex'       ,
  'Left-Hippocampus':             'Hippocampus'             ,
  'Left-Lateral-Ventricle':       'Lateral-Ventricle'       ,
  'Left-Putamen':                 'Putamen'                 ,
  'Left-VentralDC':               'VentralDC'               ,
  'Right-Accumbens-area':         'Accumbens-area'          ,
  'Right-Caudate':                'Caudate'                 ,
  'Right-Cerebellum-White-Matter':'Cerebellum-White-Matter' ,
  'Right-Inf-Lat-Vent':           'Inf-Lat-Vent'            ,
  'Right-Pallidum':               'Pallidum'                ,
  'Right-Thalamus-Proper':        'Thalamus-Proper'         ,
  'Right-Amygdala':               'Amygdala'                ,
  'Right-Cerebellum-Cortex':      'Cerebellum-Cortex'       ,
  'Right-Hippocampus':            'Hippocampus'             ,
  'Right-Lateral-Ventricle':      'Lateral-Ventricle'       ,
  'Right-Putamen':                'Putamen'                 ,
  'Right-VentralDC':              'VentralDC',
  'unknown':                      -1 # this is actually the middle region inside the cortical surface. color it as gray
}


rois_selected = ['AccumbensArea', 'Insula', 'Amygdala', 'Caudate', 'Hippocampus', 'Pallidum', 'Putamen', 'Thalamus', 'FrontalLobe', 'ParietalLobe', 'TemporalLobe', 'OccipitalLobe', 'Cingulate']
rois_all_inner = [
    'bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus','entorhinal','frontalpole','fusiform','inferiorparietal','inferiortemporal','insula',
    'isthmuscingulate','lateraloccipital','lateralorbitofrontal','lingual','medialorbitofrontal','middletemporal','paracentral','parahippocampal','parsopercularis',
    'parsorbitalis','parstriangularis','pericalcarine','postcentral','posteriorcingulate','precentral','precuneus','rostralanteriorcingulate','rostralmiddlefrontal',
    'superiorfrontal','superiorparietal','superiortemporal','supramarginal','temporalpole','transversetemporal','unknown','Accumbens-area','Caudate','Cerebellum-White-Matter',
    'Inf-Lat-Vent','Pallidum','Thalamus-Proper','Amygdala','Cerebellum-Cortex','Hippocampus','Lateral-Ventricle','Putamen','VentralDC'
]
rois_all_outer = [r.replace('-','.') for r in rois_all_inner]

if inner_outer=="brainpainter_inner.csv":
    rois = rois_all_inner
elif inner_outer=="brainpainter_outer.csv":
    rois = rois_all_outer

#* Map from DK to FreeSurfer's aparc
df_ref['Custom ROI'] = df_ref['ROI'].map(cortAreasIndexMapDK)

#* Add colours according to SuStaIn CSV
n_shades = 256
viridis = cm.get_cmap('viridis', n_shades)
viridis_ = viridis(range(0,n_shades))
viridis_ = np.flipud(viridis_)
viridis_[0,] = [1,1,1,1] # start with white
blues = cm.get_cmap('Blues', n_shades)
blues_ = blues(range(0,n_shades))
blues_[0,] = [1,1,1,1] # start with white

#SuStaIn paper: COLORS_RGB = [(1,1,1), (1,0,0), (1,0,1), (0,0,1)] # white -> red -> magenta -> blue
# cmap_sustain =
from matplotlib.colors import LinearSegmentedColormap
cmap_name_classic_SuStaIn = 'classic_SuStaIn'

#* LinearSegmentedColormap
# colors_classic_SuStaIn = [(1,1,1), (1,0,0), (1,0,1), (0,0,1)] # white (to be added) -> red -> magenta -> blue
# cmap_classic_SuStaIn = LinearSegmentedColormap.from_list(cmap_name_classic_SuStaIn, colors_classic_SuStaIn, N=n_shades)

#* Directly myself: tough because I need to go from white to red
cdict_SuStaIn = {
    'red':   ((0.0 ,  1.0, 1.0),
              (0.33,  1.0, 1.0),
              (0.66,  1.0, 1.0),
              (1.0 ,  0.0, 0.0)),
    'blue':  ((0.0 ,  1.0, 1.0),
              (0.33,  0.0, 0.0),
              (0.66,  1.0, 1.0),
              (1.0 ,  1.0, 1.0)),
    'green': ((0.0 ,  1.0, 1.0),
              (0.33,  0.0, 0.0),
              (0.66,  0.0, 0.0),
              (1.0 ,  0.0, 0.0))
}
# plt.register_cmap(name=cmap_name_classic_SuStaIn,data=cdict_SuStaIn) # deprecated
plt.register_cmap(cmap=LinearSegmentedColormap(cmap_name_classic_SuStaIn,cdict_SuStaIn))
cmap_classic_SuStaIn = plt.get_cmap(cmap_name_classic_SuStaIn)

#import colorcet as cc
#list(cc.kbc()) # Blues
import csv

#*** Select colormap
# #* Blues
# cmap_ = blues_
# cmap_name = "cmap_blues"
# cmap_[0,] = [1,1,1,1] # start with white
#* Classic SuStaIn: white - red - magenta - blue
cmap = cmap_classic_SuStaIn
cmap_ = cmap(range(0,n_shades))
cmap_name = cmap_name_classic_SuStaIn

for k in stages:
    subtype_name_k = subtype_name+'_'+str(k)
    df_ref_copy = df_ref.copy()
    row = df_sustain_cdf["Image-name-unique"]==subtype_name_k
    zscore_dict = df_sustain_cdf.loc[row.values,rois].to_dict('records')[0]
    zscores = np.around(df_ref_copy[roi_label].map(zscore_dict),2)
    df_ref_copy['z-score'] = zscores.values
    df_ref_copy['R_new'] = [int(cmap_[int(j),][0]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [l/3*(n_shades-1) for l in zscores.values] ]
    df_ref_copy['G_new'] = [int(cmap_[int(j),][1]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [l/3*(n_shades-1) for l in zscores.values] ]
    df_ref_copy['B_new'] = [int(cmap_[int(j),][2]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [l/3*(n_shades-1) for l in zscores.values] ]
    # df_ref_copy['R_new'] = [int(cmap_[int(j),][0]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [(l/3*n_shades)-1 for l in zscores.values] ]
    # df_ref_copy['G_new'] = [int(cmap_[int(j),][1]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [(l/3*n_shades)-1 for l in zscores.values] ]
    # df_ref_copy['B_new'] = [int(cmap_[int(j),][2]*(n_shades-1)) if ~np.isnan(j) else 255 for j in [(l/3*n_shades)-1 for l in zscores.values] ]
    df_ref_copy['T_new'] = df_ref_copy['T']
    #* Add whitespace for FreeSurfer aparc.ctab file
    df_ref_copy["ID"] = df_ref_copy["ID"].astype(str).str.pad(3, side ='left').str.pad(1, side ='right')
    df_ref_copy["ROI"] = df_ref_copy["ROI"].astype(str).str.pad(32, side ='right')
    df_ref_copy['R_new'] = df_ref_copy['R_new'].astype(str).str.pad(3, side ='left')
    df_ref_copy['G_new'] = df_ref_copy['G_new'].astype(str).str.pad(3, side ='left')
    df_ref_copy['B_new'] = df_ref_copy['B_new'].astype(str).str.pad(3, side ='left')
    df_ref_copy['T_new'] = df_ref_copy['T_new'].astype(str).str.pad(4, side ='left')
    fname = os.path.join(wd,"aparc_%s.annot.ctab" % (subtype_name_k))
    fname_tmp = fname + "_tmp"
    df_ref_copy[["ID","ROI","R_new","G_new","B_new","T_new"]].to_csv(fname_tmp,sep=r' ',index=False,header=False)
    #* Remove quote characters from the file automatically
    with open(fname_tmp, 'r') as infile, open(fname, 'w') as outfile:
        data = infile.read()
        data = data.replace('"', '')
        outfile.write(data)
        os.remove(fname_tmp)

#* Convert to 3D model
fs_model = os.path.join(os.path.curdir,"fs_model")
srf2obj_color_loc = os.path.join(os.path.curdir,"srf2obj_color")
meshlabserver_script_texture = os.path.join(os.path.curdir,"meshlab","simplify_clean_texture_mls2020.mlx")
meshlabserver_script_vertex  = meshlabserver_script_texture.replace("texture","vertex")
meshlabserver_script_merge  = os.path.join(os.path.curdir,"meshlab","merge_lh_rh_mls2020.mlx")
from fsto3d import *
import nipype.interfaces.freesurfer as fs
for stage in stages:
    parc = subtype_name+'_'+str(stage)
    ctab = os.path.join(wd,"aparc_"+parc+".annot.ctab")
    if os.path.exists(ctab):
        #* Output folder
        outp = os.path.join(wd,parc)
        os.makedirs(outp,exist_ok=True)
        #* Prep work on FreeSurfer model brain
        #* Convert ctab to better format with python utilities
        ctab_24bit = ctab.replace(".ctab",".24bit.ctab")
        convert_ctab(ctab, ctab_24bit)
        #* Convert surface(s) to ASCII format with color integer values
        mris = fs.MRIsConvert()
        mris.inputs.parcstats_file = ctab_24bit
        for hemisphere in ['lh','rh']:
            pial = os.path.join(fs_model,"%s.pial" % hemisphere)
            pial_asc = pial + ".asc"
            pial_asc_custom = os.path.join(outp,"%s.pial.%s_%s.asc" % (hemisphere,parc,cmap_name))
            pial_asc_custom2 = pial_asc_custom.replace("asc","combined.asc")
            pial_asc_custom_obj = pial_asc_custom2.replace(".asc",".obj")
            x3d = pial_asc_custom_obj.replace(".obj",".x3d")
            annot = os.path.join(fs_model,"%s.aparc.annot" % hemisphere)
            mris.inputs.annot_file     = annot.encode('unicode-escape').decode()
            mris.inputs.in_file        = pial.encode('unicode-escape').decode()
            mris.inputs.out_file       = pial_asc_custom.encode('unicode-escape').decode()
            mris.run()
            #* Paste together the two ASCII files to get both vertex/face info and color info
            combine_asc_color(pial_asc,pial_asc_custom,pial_asc_custom2)
            #* Convert to vertex-colored obj file (Mac prerequisite: brew install gawk)
            srf2obj_color_flag = subprocess.call([srf2obj_color_loc,pial_asc_custom2],stdout=open(pial_asc_custom_obj,"w"))
            #* Run meshlabserver script to reduce complexity, fix some potential issues, and convert vertex coloring to texture map
            meshlabserver_script_tmp = meshlabserver_script_texture.replace(".mlx","_temp.mlx")
            meshlabserver_script_tmp2 = meshlabserver_script_tmp.replace(".mlx","2.mlx")
            mlx_flag = subprocess.call(["cp",meshlabserver_script_texture,meshlabserver_script_tmp])
            #* I can't get the meshlab filter "Transfer: Vertex Color to Texture" to work. Need to include mesh info somehow
            with open(meshlabserver_script_tmp, 'r') as infile, open(meshlabserver_script_tmp2, 'w') as outfile:
                data = infile.read()
                data = data.replace("TEMP_TEXTURE",pial_asc_custom2) # mesh info file goes here
                outfile.write(data)
                #os.remove(meshlabserver_script_tmp)
            mlx_flag = subprocess.call([mls,"-i",pial_asc_custom_obj,"-s",meshlabserver_script_tmp2,"-o",x3d,"-m","vc fc fn wc wn wt"],shell=False)
        #* Merge hemispheres
        ply = os.path.join(outp,"%s_3D_%s.ply" % (parc,cmap_name))
        mlx_flag = subprocess.call([mls,"-i",x3d.replace("lh","rh").replace("rh","lh"),"-i",x3d.replace("rh","lh").replace("lh","rh"),"-s",meshlabserver_script_merge,"-o",ply,"-m","vc fc fn wc wn wt"],shell=False)
        #* Export RH by itself
        ply_rh = ply.replace('.ply','_rh.ply')
        #mlx_flag_rh = subprocess.call([mls,"-i",x3d.replace("lh","rh"),"-s",meshlabserver_script_tmp2,"-o",ply_rh,"-m","vc fc fn wc wn wt"],shell=False)
    else:
        print("ERROR: CTAB file not found: "+ctab)
