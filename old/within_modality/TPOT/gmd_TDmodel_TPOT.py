### This script creates two TPOT regressor objects using gmd data to predict age; 
### one for females and one for males
###
### Ellyn Butler
### January 31, 2019


import numpy as np
import scipy as sp
import sklearn as skl 
import pandas as pd
import deap as dp
import update_checker as uc
import tqdm
import stopit 
import xgboost as xgb
#import skrebate as skr
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
from tpot import TPOTRegressor

import warnings
warnings.simplefilter("ignore")

plotly.tools.set_credentials_file(username='flutist4129', api_key='LDwtUzfoFrgVSwpTIwwt')

def label_sex (row):
   if row['sex'] == 1:
      return 'Male'
   if row['sex'] == 2:
      return 'Female'

# import data
data = pd.read_csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv")
columns = list(data.head(0)) 
print(columns)

# Relabel sex variable
data['sex'] = data.apply (lambda row: label_sex (row),axis=1)

# import data
data = pd.read_csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv")
columns = list(data.head(0)) 
print(columns)

# Relabel sex variable
data['sex'] = data.apply (lambda row: label_sex (row),axis=1)

# Subset data to only include columns for subsequent analyses
idcols = ['ageAtScan1', 'goassessDxpmr7', 'sex']
gmdcols = ['mprage_jlf_gmd_R_Accumbens_Area', 'mprage_jlf_gmd_L_Accumbens_Area', 
           'mprage_jlf_gmd_R_Amygdala', 'mprage_jlf_gmd_L_Amygdala', 'mprage_jlf_gmd_Brain_Stem', 
           'mprage_jlf_gmd_R_Caudate', 'mprage_jlf_gmd_L_Caudate', 'mprage_jlf_gmd_R_Cerebellum_Exterior', 
           'mprage_jlf_gmd_L_Cerebellum_Exterior', 'mprage_jlf_gmd_R_Hippocampus', 'mprage_jlf_gmd_L_Hippocampus', 
           'mprage_jlf_gmd_R_Pallidum', 'mprage_jlf_gmd_L_Pallidum', 'mprage_jlf_gmd_R_Putamen', 
           'mprage_jlf_gmd_L_Putamen', 'mprage_jlf_gmd_R_Thalamus_Proper', 'mprage_jlf_gmd_L_Thalamus_Proper', 
           'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VI.VII', 
           'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_gmd_R_ACgG', 'mprage_jlf_gmd_L_ACgG', 
           'mprage_jlf_gmd_R_AIns', 'mprage_jlf_gmd_L_AIns', 'mprage_jlf_gmd_R_AOrG', 'mprage_jlf_gmd_L_AOrG', 
           'mprage_jlf_gmd_R_AnG', 'mprage_jlf_gmd_L_AnG', 'mprage_jlf_gmd_R_Calc', 'mprage_jlf_gmd_L_Calc', 
           'mprage_jlf_gmd_R_CO', 'mprage_jlf_gmd_L_CO', 'mprage_jlf_gmd_R_Cun', 'mprage_jlf_gmd_L_Cun', 
           'mprage_jlf_gmd_R_Ent', 'mprage_jlf_gmd_L_Ent', 'mprage_jlf_gmd_R_FO', 'mprage_jlf_gmd_L_FO', 
           'mprage_jlf_gmd_R_FRP', 'mprage_jlf_gmd_L_FRP', 'mprage_jlf_gmd_R_FuG', 'mprage_jlf_gmd_L_FuG', 
           'mprage_jlf_gmd_R_GRe', 'mprage_jlf_gmd_L_GRe', 'mprage_jlf_gmd_R_IOG', 'mprage_jlf_gmd_L_IOG', 
           'mprage_jlf_gmd_R_ITG', 'mprage_jlf_gmd_L_ITG', 'mprage_jlf_gmd_R_LiG', 'mprage_jlf_gmd_L_LiG', 
           'mprage_jlf_gmd_R_LOrG', 'mprage_jlf_gmd_L_LOrG', 'mprage_jlf_gmd_R_MCgG', 'mprage_jlf_gmd_L_MCgG', 
           'mprage_jlf_gmd_R_MFC', 'mprage_jlf_gmd_L_MFC', 'mprage_jlf_gmd_R_MFG', 'mprage_jlf_gmd_L_MFG', 
           'mprage_jlf_gmd_R_MOG', 'mprage_jlf_gmd_L_MOG', 'mprage_jlf_gmd_R_MOrG', 'mprage_jlf_gmd_L_MOrG', 
           'mprage_jlf_gmd_R_MPoG', 'mprage_jlf_gmd_L_MPoG', 'mprage_jlf_gmd_R_MPrG', 'mprage_jlf_gmd_L_MPrG', 
           'mprage_jlf_gmd_R_MSFG', 'mprage_jlf_gmd_L_MSFG', 'mprage_jlf_gmd_R_MTG', 'mprage_jlf_gmd_L_MTG', 
           'mprage_jlf_gmd_R_OCP', 'mprage_jlf_gmd_L_OCP', 'mprage_jlf_gmd_R_OFuG', 'mprage_jlf_gmd_L_OFuG', 
           'mprage_jlf_gmd_R_OpIFG', 'mprage_jlf_gmd_L_OpIFG', 'mprage_jlf_gmd_R_OrIFG', 'mprage_jlf_gmd_L_OrIFG', 
           'mprage_jlf_gmd_R_PCgG', 'mprage_jlf_gmd_L_PCgG', 'mprage_jlf_gmd_R_PCu', 'mprage_jlf_gmd_L_PCu', 
           'mprage_jlf_gmd_R_PHG', 'mprage_jlf_gmd_L_PHG', 'mprage_jlf_gmd_R_PIns', 'mprage_jlf_gmd_L_PIns', 
           'mprage_jlf_gmd_R_PO', 'mprage_jlf_gmd_L_PO', 'mprage_jlf_gmd_R_PoG', 'mprage_jlf_gmd_L_PoG', 
           'mprage_jlf_gmd_R_POrG', 'mprage_jlf_gmd_L_POrG', 'mprage_jlf_gmd_R_PP', 'mprage_jlf_gmd_L_PP', 
           'mprage_jlf_gmd_R_PrG', 'mprage_jlf_gmd_L_PrG', 'mprage_jlf_gmd_R_PT', 'mprage_jlf_gmd_L_PT', 
           'mprage_jlf_gmd_R_SCA', 'mprage_jlf_gmd_L_SCA', 'mprage_jlf_gmd_R_SFG', 'mprage_jlf_gmd_L_SFG', 
           'mprage_jlf_gmd_R_SMC', 'mprage_jlf_gmd_L_SMC', 'mprage_jlf_gmd_R_SMG', 'mprage_jlf_gmd_L_SMG', 
           'mprage_jlf_gmd_R_SOG', 'mprage_jlf_gmd_L_SOG', 'mprage_jlf_gmd_R_SPL', 'mprage_jlf_gmd_L_SPL', 
           'mprage_jlf_gmd_R_STG', 'mprage_jlf_gmd_L_STG', 'mprage_jlf_gmd_R_TMP', 'mprage_jlf_gmd_L_TMP', 
           'mprage_jlf_gmd_R_TrIFG', 'mprage_jlf_gmd_L_TrIFG', 'mprage_jlf_gmd_R_TTG', 'mprage_jlf_gmd_L_TTG']

usecols = idcols + gmdcols
relcoldata = data[usecols]

# Split data by diagnosis
TD_criteria = relcoldata['goassessDxpmr7'] == "TD"
TD_data = relcoldata[TD_criteria]
TD_data_F = TD_data.loc[TD_data['sex'] == 'Female']
TD_data_M = TD_data.loc[TD_data['sex'] == 'Male']

PS_criteria = relcoldata['goassessDxpmr7'] == "PS"
PS_data = relcoldata[PS_criteria]
PS_data_F = PS_data.loc[PS_data['sex'] == 'Female']
PS_data_M = PS_data.loc[PS_data['sex'] == 'Male']

######################### Females #########################
gmd_df_td_F = TD_data_F
gmd_df_td_F = gmd_df_td_F.dropna()
gmd_X_td_F = gmd_df_td_F.drop(columns=['ageAtScan1', 'goassessDxpmr7', 'sex'])
for column in gmd_X_td_F: 
    pd.to_numeric(gmd_X_td_F[column], errors='coerce')
gmd_y_td_F = gmd_df_td_F.ageAtScan1
pd.to_numeric(gmd_y_td_F, errors='coerce')

TDmodel_gmd_F = TPOTRegressor(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, memory='auto', scoring='r2')
TDmodel_gmd_F.fit(gmd_X_td_F, gmd_y_td_F)

# TD R2
print(TDmodel_gmd_F.score(gmd_X_td_F, gmd_y_td_F))

# PS R2
gmd_df_ps_F = PS_data_F
gmd_df_ps_F = gmd_df_ps_F.dropna()
gmd_X_ps_F = gmd_df_ps_F.drop(['ageAtScan1', 'goassessDxpmr7', 'sex'], axis=1)
for column in gmd_X_ps_F: 
    pd.to_numeric(gmd_X_ps_F[column], errors='coerce')
gmd_y_ps_F = gmd_df_ps_F.ageAtScan1
pd.to_numeric(gmd_y_ps_F, errors='coerce')
print(TDmodel_gmd_F.score(gmd_X_ps_F, gmd_y_ps_F))

# Create new columns in dataframe
# --- TD
# 1) real and predicted
gmd_df_td_F['pred_age'] = TDmodel_gmd_F.predict(gmd_X_td_F)
real_age_td_F = gmd_df_td_F.ageAtScan1
pred_age_td_F = TDmodel_gmd_F.predict(gmd_X_td_F)
gmd_df_td_F['diff_real_pred_age'] = real_age_td_F - pred_age_td_F
gmd_df_td_F['real_over18'] = gmd_df_td_F.ageAtScan1 >= 216
gmd_df_td_F['pred_over18'] = gmd_df_td_F.pred_age >= 216
# 2) age group indicators
gmd_df_td_F['8_9'] = ((gmd_df_td_F.ageAtScan1 >= 96) & (gmd_df_td_F.ageAtScan1 < 120))
gmd_df_td_F['10_11'] = ((gmd_df_td_F.ageAtScan1 >= 120) & (gmd_df_td_F.ageAtScan1 < 144))
gmd_df_td_F['12_13'] = ((gmd_df_td_F.ageAtScan1 >= 144) & (gmd_df_td_F.ageAtScan1 < 168))
gmd_df_td_F['14_15'] = ((gmd_df_td_F.ageAtScan1 >= 168) & (gmd_df_td_F.ageAtScan1 < 192))
gmd_df_td_F['16_17'] = ((gmd_df_td_F.ageAtScan1 >= 192) & (gmd_df_td_F.ageAtScan1 < 216))
gmd_df_td_F['18_19'] = ((gmd_df_td_F.ageAtScan1 >= 216) & (gmd_df_td_F.ageAtScan1 < 240))
gmd_df_td_F['20_21'] = ((gmd_df_td_F.ageAtScan1 >= 240) & (gmd_df_td_F.ageAtScan1 < 264))
gmd_df_td_F['22_23'] = ((gmd_df_td_F.ageAtScan1 >= 264) & (gmd_df_td_F.ageAtScan1 < 388))

# --- PS
# 1) real and predicted
gmd_df_ps_F['pred_age'] = TDmodel_gmd_F.predict(gmd_X_ps_F)
real_age_ps_F = gmd_df_ps_F.ageAtScan1
pred_age_ps_F = TDmodel_gmd_F.predict(gmd_X_ps_F)
gmd_df_ps_F['diff_real_pred_age'] = real_age_ps_F - pred_age_ps_F
gmd_df_ps_F['real_over18'] = gmd_df_ps_F.ageAtScan1 >= 216
gmd_df_ps_F['pred_over18'] = gmd_df_ps_F.pred_age >= 216
# 2) age group indicators
gmd_df_ps_F['8_9'] = ((gmd_df_ps_F.ageAtScan1 >= 96) & (gmd_df_ps_F.ageAtScan1 < 120))
gmd_df_ps_F['10_11'] = ((gmd_df_ps_F.ageAtScan1 >= 120) & (gmd_df_ps_F.ageAtScan1 < 144))
gmd_df_ps_F['12_13'] = ((gmd_df_ps_F.ageAtScan1 >= 144) & (gmd_df_ps_F.ageAtScan1 < 168))
gmd_df_ps_F['14_15'] = ((gmd_df_ps_F.ageAtScan1 >= 168) & (gmd_df_ps_F.ageAtScan1 < 192))
gmd_df_ps_F['16_17'] = ((gmd_df_ps_F.ageAtScan1 >= 192) & (gmd_df_ps_F.ageAtScan1 < 216))
gmd_df_ps_F['18_19'] = ((gmd_df_ps_F.ageAtScan1 >= 216) & (gmd_df_ps_F.ageAtScan1 < 240))
gmd_df_ps_F['20_21'] = ((gmd_df_ps_F.ageAtScan1 >= 240) & (gmd_df_ps_F.ageAtScan1 < 264))
gmd_df_ps_F['22_23'] = ((gmd_df_ps_F.ageAtScan1 >= 264) & (gmd_df_ps_F.ageAtScan1 < 388))

# Check for the number of TD's whose predicted adulthood status did not match their actual # unnecessary
gmd_df_td_F[['real_over18', 'pred_over18']]
td_mismatch_F = 0
td_pred_gt_real_F = 0
for row in gmd_df_td_F.index:
    if gmd_df_td_F.at[row,'real_over18'] != gmd_df_td_F.at[row,'pred_over18']:
        td_mismatch_F = td_mismatch_F + 1
    if (gmd_df_td_F.at[row,'real_over18'] == False) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        td_pred_gt_real_F = td_pred_gt_real_F + 1
print(td_mismatch_F, td_pred_gt_real_F)

# Check for the number of PS's whose predicted adulthood status did not match their actual # unnecessary
gmd_df_ps_F[['real_over18', 'pred_over18']]
ps_mismatch_F = 0
ps_pred_gt_real_F = 0
for row in gmd_df_ps_F.index:
    if gmd_df_ps_F.at[row,'real_over18'] != gmd_df_ps_F.at[row,'pred_over18']:
        ps_mismatch_F = ps_mismatch_F + 1
    if (gmd_df_ps_F.at[row,'real_over18'] == False) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        ps_pred_gt_real_F = ps_pred_gt_real_F + 1
print(ps_mismatch_F, ps_pred_gt_real_F)

# Scatterplot of Pred Age vs. Real Age: TD
plt.scatter(gmd_df_td_F['ageAtScan1'], gmd_df_td_F['pred_age'])
plt.title("TD Females")
plt.xlabel("Real Age (months)")
plt.ylabel("Predicted Age (months)")
plt.plot( [0,300],[0,300] )

# Scatterplot of Pred Age vs. Real Age: PS
plt.scatter(gmd_df_ps_F['ageAtScan1'], gmd_df_ps_F['pred_age'])
plt.title("PS Females")
plt.xlabel("Real Age (months)")
plt.ylabel("Predicted Age (months)")
plt.plot( [0,300],[0,300] )

# Barplot of % in age bin inaccurately predicted as being over 18: TD
# 8_9
pred_over18_8_9 = 0
num_8_9 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'8_9'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_8_9 = pred_over18_8_9 + 1
    if gmd_df_td_F.at[row,'8_9'] == True:
        num_8_9 = num_8_9 + 1
   
percent_td_8_9_pred_over18_F = pred_over18_8_9/num_8_9

# 10_11
pred_over18_10_11 = 0
num_10_11 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'10_11'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_10_11 = pred_over18_10_11 + 1
    if gmd_df_td_F.at[row,'10_11'] == True:
        num_10_11 = num_10_11 + 1
   
percent_td_10_11_pred_over18_F = pred_over18_10_11/num_10_11

# 12_13
pred_over18_12_13 = 0
num_12_13 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'12_13'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_12_13 = pred_over18_12_13 + 1
    if gmd_df_td_F.at[row,'12_13'] == True:
        num_12_13 = num_12_13 + 1
   
percent_td_12_13_pred_over18_F = pred_over18_12_13/num_12_13

# 14_15
pred_over18_14_15 = 0
num_14_15 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'14_15'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_14_15 = pred_over18_14_15 + 1
    if gmd_df_td_F.at[row,'14_15'] == True:
        num_14_15 = num_14_15 + 1
   
percent_td_14_15_pred_over18_F = pred_over18_14_15/num_14_15

# 16_17
pred_over18_16_17 = 0
num_16_17 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'16_17'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_16_17 = pred_over18_16_17 + 1
    if gmd_df_td_F.at[row,'16_17'] == True:
        num_16_17 = num_16_17 + 1
   
percent_td_16_17_pred_over18_F = pred_over18_16_17/num_16_17

# 18_19
pred_over18_18_19 = 0
num_18_19 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'18_19'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_18_19 = pred_over18_18_19 + 1
    if gmd_df_td_F.at[row,'18_19'] == True:
        num_18_19 = num_18_19 + 1
   
percent_td_18_19_pred_over18_F = pred_over18_18_19/num_18_19

# 20_21
pred_over18_20_21 = 0
num_20_21 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'20_21'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_20_21 = pred_over18_20_21 + 1
    if gmd_df_td_F.at[row,'20_21'] == True:
        num_20_21 = num_20_21 + 1
   
percent_td_20_21_pred_over18_F = pred_over18_20_21/num_20_21

# 22_23
pred_over18_22_23 = 0
num_22_23 = 0

for row in gmd_df_td_F.index:
    if (gmd_df_td_F.at[row,'22_23'] == True) & (gmd_df_td_F.at[row,'pred_over18'] == True):
        pred_over18_22_23 = pred_over18_22_23 + 1
    if gmd_df_td_F.at[row,'22_23'] == True:
        num_22_23 = num_22_23 + 1
   
percent_td_22_23_pred_over18_F = pred_over18_22_23/num_22_23

# Barplot of % in age bin inaccurately predicted as being over 18: PS
# 8_9
pred_over18_8_9 = 0
num_8_9 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'8_9'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_8_9 = pred_over18_8_9 + 1
    if gmd_df_ps_F.at[row,'8_9'] == True:
        num_8_9 = num_8_9 + 1
   
percent_ps_8_9_pred_over18_F = pred_over18_8_9/num_8_9

# 10_11
pred_over18_10_11 = 0
num_10_11 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'10_11'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_10_11 = pred_over18_10_11 + 1
    if gmd_df_ps_F.at[row,'10_11'] == True:
        num_10_11 = num_10_11 + 1
   
percent_ps_10_11_pred_over18_F = pred_over18_10_11/num_10_11

# 12_13
pred_over18_12_13 = 0
num_12_13 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'12_13'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_12_13 = pred_over18_12_13 + 1
    if gmd_df_ps_F.at[row,'12_13'] == True:
        num_12_13 = num_12_13 + 1
   
percent_ps_12_13_pred_over18_F = pred_over18_12_13/num_12_13

# 14_15
pred_over18_14_15 = 0
num_14_15 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'14_15'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_14_15 = pred_over18_14_15 + 1
    if gmd_df_ps_F.at[row,'14_15'] == True:
        num_14_15 = num_14_15 + 1
   
percent_ps_14_15_pred_over18_F = pred_over18_14_15/num_14_15

# 16_17
pred_over18_16_17 = 0
num_16_17 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'16_17'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_16_17 = pred_over18_16_17 + 1
    if gmd_df_ps_F.at[row,'16_17'] == True:
        num_16_17 = num_16_17 + 1
   
percent_ps_16_17_pred_over18_F = pred_over18_16_17/num_16_17

# 18_19
pred_over18_18_19 = 0
num_18_19 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'18_19'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_18_19 = pred_over18_18_19 + 1
    if gmd_df_ps_F.at[row,'18_19'] == True:
        num_18_19 = num_18_19 + 1
   
percent_ps_18_19_pred_over18_F = pred_over18_18_19/num_18_19

# 20_21
pred_over18_20_21 = 0
num_20_21 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'20_21'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_20_21 = pred_over18_20_21 + 1
    if gmd_df_ps_F.at[row,'20_21'] == True:
        num_20_21 = num_20_21 + 1
   
percent_ps_20_21_pred_over18_F = pred_over18_20_21/num_20_21

# 22_23
pred_over18_22_23 = 0
num_22_23 = 0

for row in gmd_df_ps_F.index:
    if (gmd_df_ps_F.at[row,'22_23'] == True) & (gmd_df_ps_F.at[row,'pred_over18'] == True):
        pred_over18_22_23 = pred_over18_22_23 + 1
    if gmd_df_ps_F.at[row,'22_23'] == True:
        num_22_23 = num_22_23 + 1
   
percent_ps_22_23_pred_over18_F = pred_over18_22_23/num_22_23


trace1 = go.Bar(
    x=['8_9','10_11','12_13','14_15','16_17','18_19','20_21'],
    y=[percent_td_8_9_pred_over18_F, percent_td_10_11_pred_over18_F, percent_td_12_13_pred_over18_F,
       percent_td_14_15_pred_over18_F, percent_td_16_17_pred_over18_F, percent_td_18_19_pred_over18_F,
      percent_td_20_21_pred_over18_F],
    name='TD'
)

trace2 = go.Bar(
    x=['8_9','10_11','12_13','14_15','16_17','18_19','20_21'],
    y=[percent_ps_8_9_pred_over18_F, percent_ps_10_11_pred_over18_F, percent_ps_12_13_pred_over18_F,
       percent_ps_14_15_pred_over18_F, percent_ps_16_17_pred_over18_F, percent_ps_18_19_pred_over18_F,
      percent_ps_20_21_pred_over18_F],
    name='PS'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Percent of Females by Age Predicted to be Over 18'
)

F_barplot = go.Figure(data=data, layout=layout)
py.iplot(F_barplot, filename='grouped-bar')


######################### Males #########################
gmd_df_td_M = TD_data_M
gmd_df_td_M = gmd_df_td_M.dropna()
gmd_X_td_M = gmd_df_td_M.drop(columns=['ageAtScan1', 'goassessDxpmr7', 'sex'])
for column in gmd_X_td_M: 
    pd.to_numeric(gmd_X_td_M[column], errors='coerce')
gmd_y_td_M = gmd_df_td_M.ageAtScan1
pd.to_numeric(gmd_y_td_M, errors='coerce')

TDmodel_gmd_M = TPOTRegressor(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, memory='auto', scoring='r2')
TDmodel_gmd_M.fit(gmd_X_td_M, gmd_y_td_M)

# TD R2
print(TDmodel_gmd_M.score(gmd_X_td_M, gmd_y_td_M))

# PS R2
colvec = idcols + gmdcols
gmd_df_ps_M = PS_data_M[colvec]
gmd_df_ps_M = gmd_df_ps_M.dropna()
gmd_X_ps_M = gmd_df_ps_M.drop(['ageAtScan1', 'goassessDxpmr7', 'sex'], axis=1)
for column in gmd_X_ps_M: 
    pd.to_numeric(gmd_X_ps_M[column], errors='coerce')
gmd_y_ps_M = gmd_df_ps_M.ageAtScan1
print(TDmodel_gmd_M.score(gmd_X_ps_M, gmd_y_ps_M))

# Create new columns in dataframe
# --- TD
# 1) real and predicted
gmd_df_td_M['pred_age'] = TDmodel_gmd_M.predict(gmd_X_td_M)
real_age_td_M = gmd_df_td_M.ageAtScan1
pred_age_td_M = TDmodel_gmd_M.predict(gmd_X_td_M)
gmd_df_td_M['diff_real_pred_age'] = real_age_td_M - pred_age_td_M
gmd_df_td_M['real_over18'] = gmd_df_td_M.ageAtScan1 >= 216
gmd_df_td_M['pred_over18'] = gmd_df_td_M.pred_age >= 216
# 2) age group indicators
gmd_df_td_M['8_9'] = ((gmd_df_td_M.ageAtScan1 >= 96) & (gmd_df_td_M.ageAtScan1 < 120))
gmd_df_td_M['10_11'] = ((gmd_df_td_M.ageAtScan1 >= 120) & (gmd_df_td_M.ageAtScan1 < 144))
gmd_df_td_M['12_13'] = ((gmd_df_td_M.ageAtScan1 >= 144) & (gmd_df_td_M.ageAtScan1 < 168))
gmd_df_td_M['14_15'] = ((gmd_df_td_M.ageAtScan1 >= 168) & (gmd_df_td_M.ageAtScan1 < 192))
gmd_df_td_M['16_17'] = ((gmd_df_td_M.ageAtScan1 >= 192) & (gmd_df_td_M.ageAtScan1 < 216))
gmd_df_td_M['18_19'] = ((gmd_df_td_M.ageAtScan1 >= 216) & (gmd_df_td_M.ageAtScan1 < 240))
gmd_df_td_M['20_21'] = ((gmd_df_td_M.ageAtScan1 >= 240) & (gmd_df_td_M.ageAtScan1 < 264))
gmd_df_td_M['22_23'] = ((gmd_df_td_M.ageAtScan1 >= 264) & (gmd_df_td_M.ageAtScan1 < 388))

# --- PS
# 1) real and predicted
gmd_df_ps_M['pred_age'] = TDmodel_gmd_M.predict(gmd_X_ps_M)
real_age_ps_M = gmd_df_ps_M.ageAtScan1
pred_age_ps_M = TDmodel_gmd_M.predict(gmd_X_ps_M)
gmd_df_ps_M['diff_real_pred_age'] = real_age_ps_M - pred_age_ps_M
gmd_df_ps_M['real_over18'] = gmd_df_ps_M.ageAtScan1 >= 216
gmd_df_ps_M['pred_over18'] = gmd_df_ps_M.pred_age >= 216
# 2) age group indicators
gmd_df_ps_M['8_9'] = ((gmd_df_ps_M.ageAtScan1 >= 96) & (gmd_df_ps_M.ageAtScan1 < 120))
gmd_df_ps_M['10_11'] = ((gmd_df_ps_M.ageAtScan1 >= 120) & (gmd_df_ps_M.ageAtScan1 < 144))
gmd_df_ps_M['12_13'] = ((gmd_df_ps_M.ageAtScan1 >= 144) & (gmd_df_ps_M.ageAtScan1 < 168))
gmd_df_ps_M['14_15'] = ((gmd_df_ps_M.ageAtScan1 >= 168) & (gmd_df_ps_M.ageAtScan1 < 192))
gmd_df_ps_M['16_17'] = ((gmd_df_ps_M.ageAtScan1 >= 192) & (gmd_df_ps_M.ageAtScan1 < 216))
gmd_df_ps_M['18_19'] = ((gmd_df_ps_M.ageAtScan1 >= 216) & (gmd_df_ps_M.ageAtScan1 < 240))
gmd_df_ps_M['20_21'] = ((gmd_df_ps_M.ageAtScan1 >= 240) & (gmd_df_ps_M.ageAtScan1 < 264))
gmd_df_ps_M['22_23'] = ((gmd_df_ps_M.ageAtScan1 >= 264) & (gmd_df_ps_M.ageAtScan1 < 388))

# Scatterplot of Pred Age vs. Real Age: TD
plt.scatter(gmd_df_td_M['ageAtScan1'], gmd_df_td_M['pred_age'])
plt.title("TD Males")
plt.xlabel("Real Age (months)")
plt.ylabel("Predicted Age (months)")
plt.plot( [0,300],[0,300] )

# Scatterplot of Pred Age vs. Real Age: PS
plt.scatter(gmd_df_ps_M['ageAtScan1'], gmd_df_ps_M['pred_age'])
plt.title("PS Males")
plt.xlabel("Real Age (months)")
plt.ylabel("Predicted Age (months)")
plt.plot( [0,300],[0,300] )

# Barplot of % in age bin inaccurately predicted as being over 18: TD
# 8_9
pred_over18_8_9 = 0
num_8_9 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'8_9'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_8_9 = pred_over18_8_9 + 1
    if gmd_df_td_M.at[row,'8_9'] == True:
        num_8_9 = num_8_9 + 1
   
percent_td_8_9_pred_over18_M = pred_over18_8_9/num_8_9

# 10_11
pred_over18_10_11 = 0
num_10_11 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'10_11'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_10_11 = pred_over18_10_11 + 1
    if gmd_df_td_M.at[row,'10_11'] == True:
        num_10_11 = num_10_11 + 1
   
percent_td_10_11_pred_over18_M = pred_over18_10_11/num_10_11

# 12_13
pred_over18_12_13 = 0
num_12_13 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'12_13'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_12_13 = pred_over18_12_13 + 1
    if gmd_df_td_M.at[row,'12_13'] == True:
        num_12_13 = num_12_13 + 1
   
percent_td_12_13_pred_over18_M = pred_over18_12_13/num_12_13

# 14_15
pred_over18_14_15 = 0
num_14_15 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'14_15'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_14_15 = pred_over18_14_15 + 1
    if gmd_df_td_M.at[row,'14_15'] == True:
        num_14_15 = num_14_15 + 1
   
percent_td_14_15_pred_over18_M = pred_over18_14_15/num_14_15

# 16_17
pred_over18_16_17 = 0
num_16_17 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'16_17'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_16_17 = pred_over18_16_17 + 1
    if gmd_df_td_M.at[row,'16_17'] == True:
        num_16_17 = num_16_17 + 1
   
percent_td_16_17_pred_over18_M = pred_over18_16_17/num_16_17

# 18_19
pred_over18_18_19 = 0
num_18_19 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'18_19'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_18_19 = pred_over18_18_19 + 1
    if gmd_df_td_M.at[row,'18_19'] == True:
        num_18_19 = num_18_19 + 1
   
percent_td_18_19_pred_over18_M = pred_over18_18_19/num_18_19

# 20_21
pred_over18_20_21 = 0
num_20_21 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'20_21'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_20_21 = pred_over18_20_21 + 1
    if gmd_df_td_M.at[row,'20_21'] == True:
        num_20_21 = num_20_21 + 1
   
percent_td_20_21_pred_over18_M = pred_over18_20_21/num_20_21

# 22_23
pred_over18_22_23 = 0
num_22_23 = 0

for row in gmd_df_td_M.index:
    if (gmd_df_td_M.at[row,'22_23'] == True) & (gmd_df_td_M.at[row,'pred_over18'] == True):
        pred_over18_22_23 = pred_over18_22_23 + 1
    if gmd_df_td_M.at[row,'22_23'] == True:
        num_22_23 = num_22_23 + 1
   
percent_td_22_23_pred_over18_M = pred_over18_22_23/num_22_23

# Barplot of % in age bin inaccurately predicted as being over 18: PS
# 8_9
pred_over18_8_9 = 0
num_8_9 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'8_9'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_8_9 = pred_over18_8_9 + 1
    if gmd_df_ps_M.at[row,'8_9'] == True:
        num_8_9 = num_8_9 + 1
   
percent_ps_8_9_pred_over18_M = pred_over18_8_9/num_8_9

# 10_11
pred_over18_10_11 = 0
num_10_11 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'10_11'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_10_11 = pred_over18_10_11 + 1
    if gmd_df_ps_M.at[row,'10_11'] == True:
        num_10_11 = num_10_11 + 1
   
percent_ps_10_11_pred_over18_M = pred_over18_10_11/num_10_11

# 12_13
pred_over18_12_13 = 0
num_12_13 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'12_13'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_12_13 = pred_over18_12_13 + 1
    if gmd_df_ps_M.at[row,'12_13'] == True:
        num_12_13 = num_12_13 + 1
   
percent_ps_12_13_pred_over18_M = pred_over18_12_13/num_12_13

# 14_15
pred_over18_14_15 = 0
num_14_15 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'14_15'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_14_15 = pred_over18_14_15 + 1
    if gmd_df_ps_M.at[row,'14_15'] == True:
        num_14_15 = num_14_15 + 1
   
percent_ps_14_15_pred_over18_M = pred_over18_14_15/num_14_15

# 16_17
pred_over18_16_17 = 0
num_16_17 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'16_17'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_16_17 = pred_over18_16_17 + 1
    if gmd_df_ps_M.at[row,'16_17'] == True:
        num_16_17 = num_16_17 + 1
   
percent_ps_16_17_pred_over18_M = pred_over18_16_17/num_16_17

# 18_19
pred_over18_18_19 = 0
num_18_19 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'18_19'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_18_19 = pred_over18_18_19 + 1
    if gmd_df_ps_M.at[row,'18_19'] == True:
        num_18_19 = num_18_19 + 1
   
percent_ps_18_19_pred_over18_M = pred_over18_18_19/num_18_19

# 20_21
pred_over18_20_21 = 0
num_20_21 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'20_21'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_20_21 = pred_over18_20_21 + 1
    if gmd_df_ps_M.at[row,'20_21'] == True:
        num_20_21 = num_20_21 + 1
   
percent_ps_20_21_pred_over18_M = pred_over18_20_21/num_20_21

# 22_23
pred_over18_22_23 = 0
num_22_23 = 0

for row in gmd_df_ps_M.index:
    if (gmd_df_ps_M.at[row,'22_23'] == True) & (gmd_df_ps_M.at[row,'pred_over18'] == True):
        pred_over18_22_23 = pred_over18_22_23 + 1
    if gmd_df_ps_M.at[row,'22_23'] == True:
        num_22_23 = num_22_23 + 1
   
percent_ps_22_23_pred_over18_M = pred_over18_22_23/num_22_23


trace1 = go.Bar(
    x=['8_9','10_11','12_13','14_15','16_17','18_19','20_21'],
    y=[percent_td_8_9_pred_over18_M, percent_td_10_11_pred_over18_M, percent_td_12_13_pred_over18_M,
       percent_td_14_15_pred_over18_M, percent_td_16_17_pred_over18_M, percent_td_18_19_pred_over18_M,
      percent_td_20_21_pred_over18_M],
    name='TD'
)

trace2 = go.Bar(
    x=['8_9','10_11','12_13','14_15','16_17','18_19','20_21'],
    y=[percent_ps_8_9_pred_over18_M, percent_ps_10_11_pred_over18_M, percent_ps_12_13_pred_over18_M,
       percent_ps_14_15_pred_over18_M, percent_ps_16_17_pred_over18_M, percent_ps_18_19_pred_over18_M,
      percent_ps_20_21_pred_over18_M],
    name='PS'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Percent of Males by Age Predicted to be Over 18'
)

M_barplot = go.Figure(data=data, layout=layout)
py.iplot(M_barplot, filename='grouped-bar')

# export optimized pipelines
TDmodel_gmd_F.export('tpot_agepred_TD_gmd_F.py')
TDmodel_gmd_M.export('tpot_agepred_TD_gmd_M.py')































