### This script produces linear models, features for which have been selected with a grid search of
### alphas and lambdas for elastic net, for all modalities predicting age in two same-sex samples who
### did not have a mental illness (TD) at their final visit. All predictions are out-of-sample, such that
### TD's obtain their predicted age from a model built on the other 4/5's of TD's, and people with psychosis-
### spectrum symptoms (PS's) obtain their predicted age from a model built on all of the TD's.
###
### Ellyn Butler
### March 11, 2019 - present

args <- commandArgs(trailingOnly=TRUE)
type <- args[1]

# Load libraries
library('glmnet')
library('ggplot2')
library('reshape2')
library('gridExtra')
library('psych')
library('dplyr')
library('caret')
library('dfoptim')
library('Rmisc')

# Source my functions
source("/home/butellyn/ButlerPlotFuncs/plotFuncs_AdonPath.R")

# Load data #February 25: Change clinical variable to psTerminal
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

alffvars <- c('rest_jlf_alff_R_Accumbens_Area', 'rest_jlf_alff_L_Accumbens_Area', 'rest_jlf_alff_R_Amygdala', 'rest_jlf_alff_L_Amygdala', 'rest_jlf_alff_R_Caudate', 'rest_jlf_alff_L_Caudate', 'rest_jlf_alff_R_Cerebellum_Exterior', 'rest_jlf_alff_L_Cerebellum_Exterior', 'rest_jlf_alff_R_Hippocampus', 'rest_jlf_alff_L_Hippocampus', 'rest_jlf_alff_R_Pallidum', 'rest_jlf_alff_L_Pallidum', 'rest_jlf_alff_R_Putamen', 'rest_jlf_alff_L_Putamen', 'rest_jlf_alff_R_Thalamus_Proper', 'rest_jlf_alff_L_Thalamus_Proper', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_I.V', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_VI.VII', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_VIII.X', 'rest_jlf_alff_R_ACgG', 'rest_jlf_alff_L_ACgG', 'rest_jlf_alff_R_AIns', 'rest_jlf_alff_L_AIns', 'rest_jlf_alff_R_AOrG', 'rest_jlf_alff_L_AOrG', 'rest_jlf_alff_R_AnG', 'rest_jlf_alff_L_AnG', 'rest_jlf_alff_R_Calc', 'rest_jlf_alff_L_Calc', 'rest_jlf_alff_R_CO', 'rest_jlf_alff_L_CO', 'rest_jlf_alff_R_Cun', 'rest_jlf_alff_L_Cun', 'rest_jlf_alff_R_Ent', 'rest_jlf_alff_L_Ent', 'rest_jlf_alff_R_FO', 'rest_jlf_alff_L_FO', 'rest_jlf_alff_R_FRP', 'rest_jlf_alff_L_FRP', 'rest_jlf_alff_R_FuG', 'rest_jlf_alff_L_FuG', 'rest_jlf_alff_R_GRe', 'rest_jlf_alff_L_GRe', 'rest_jlf_alff_R_IOG', 'rest_jlf_alff_L_IOG', 'rest_jlf_alff_R_ITG', 'rest_jlf_alff_L_ITG', 'rest_jlf_alff_R_LiG', 'rest_jlf_alff_L_LiG', 'rest_jlf_alff_R_LOrG', 'rest_jlf_alff_L_LOrG', 'rest_jlf_alff_R_MCgG', 'rest_jlf_alff_L_MCgG', 'rest_jlf_alff_R_MFC', 'rest_jlf_alff_L_MFC', 'rest_jlf_alff_R_MFG', 'rest_jlf_alff_L_MFG', 'rest_jlf_alff_R_MOG', 'rest_jlf_alff_L_MOG', 'rest_jlf_alff_R_MOrG', 'rest_jlf_alff_L_MOrG', 'rest_jlf_alff_R_MPoG', 'rest_jlf_alff_L_MPoG', 'rest_jlf_alff_R_MPrG', 'rest_jlf_alff_L_MPrG', 'rest_jlf_alff_R_MSFG', 'rest_jlf_alff_L_MSFG', 'rest_jlf_alff_R_MTG', 'rest_jlf_alff_L_MTG', 'rest_jlf_alff_R_OCP', 'rest_jlf_alff_L_OCP', 'rest_jlf_alff_R_OFuG', 'rest_jlf_alff_L_OFuG', 'rest_jlf_alff_R_OpIFG', 'rest_jlf_alff_L_OpIFG', 'rest_jlf_alff_R_OrIFG', 'rest_jlf_alff_L_OrIFG', 'rest_jlf_alff_R_PCgG', 'rest_jlf_alff_L_PCgG', 'rest_jlf_alff_R_PCu', 'rest_jlf_alff_L_PCu', 'rest_jlf_alff_R_PHG', 'rest_jlf_alff_L_PHG', 'rest_jlf_alff_R_PIns', 'rest_jlf_alff_L_PIns', 'rest_jlf_alff_R_PO', 'rest_jlf_alff_L_PO', 'rest_jlf_alff_R_PoG', 'rest_jlf_alff_L_PoG', 'rest_jlf_alff_R_POrG', 'rest_jlf_alff_L_POrG', 'rest_jlf_alff_R_PP', 'rest_jlf_alff_L_PP', 'rest_jlf_alff_R_PrG', 'rest_jlf_alff_L_PrG', 'rest_jlf_alff_R_PT', 'rest_jlf_alff_L_PT', 'rest_jlf_alff_R_SCA', 'rest_jlf_alff_L_SCA', 'rest_jlf_alff_R_SFG', 'rest_jlf_alff_L_SFG', 'rest_jlf_alff_R_SMC', 'rest_jlf_alff_L_SMC', 'rest_jlf_alff_R_SMG', 'rest_jlf_alff_L_SMG', 'rest_jlf_alff_R_SOG', 'rest_jlf_alff_L_SOG', 'rest_jlf_alff_R_SPL', 'rest_jlf_alff_L_SPL', 'rest_jlf_alff_R_STG', 'rest_jlf_alff_L_STG', 'rest_jlf_alff_R_TMP', 'rest_jlf_alff_L_TMP', 'rest_jlf_alff_R_TrIFG', 'rest_jlf_alff_L_TrIFG', 'rest_jlf_alff_R_TTG', 'rest_jlf_alff_L_TTG')

cbfvars <- c('pcasl_jlf_cbf_R_Accumbens_Area', 'pcasl_jlf_cbf_L_Accumbens_Area', 'pcasl_jlf_cbf_R_Amygdala', 'pcasl_jlf_cbf_L_Amygdala', 'pcasl_jlf_cbf_R_Caudate', 'pcasl_jlf_cbf_L_Caudate', 'pcasl_jlf_cbf_R_Hippocampus', 'pcasl_jlf_cbf_L_Hippocampus', 'pcasl_jlf_cbf_R_Pallidum', 'pcasl_jlf_cbf_L_Pallidum', 'pcasl_jlf_cbf_R_Putamen', 'pcasl_jlf_cbf_L_Putamen', 'pcasl_jlf_cbf_R_Thalamus_Proper', 'pcasl_jlf_cbf_L_Thalamus_Proper', 'pcasl_jlf_cbf_R_ACgG', 'pcasl_jlf_cbf_L_ACgG', 'pcasl_jlf_cbf_R_AIns', 'pcasl_jlf_cbf_L_AIns', 'pcasl_jlf_cbf_R_AOrG', 'pcasl_jlf_cbf_L_AOrG', 'pcasl_jlf_cbf_R_AnG', 'pcasl_jlf_cbf_L_AnG', 'pcasl_jlf_cbf_R_Calc', 'pcasl_jlf_cbf_L_Calc', 'pcasl_jlf_cbf_R_CO', 'pcasl_jlf_cbf_L_CO', 'pcasl_jlf_cbf_R_Cun', 'pcasl_jlf_cbf_L_Cun', 'pcasl_jlf_cbf_R_Ent', 'pcasl_jlf_cbf_L_Ent', 'pcasl_jlf_cbf_R_FO', 'pcasl_jlf_cbf_L_FO', 'pcasl_jlf_cbf_R_FRP', 'pcasl_jlf_cbf_L_FRP', 'pcasl_jlf_cbf_R_FuG', 'pcasl_jlf_cbf_L_FuG', 'pcasl_jlf_cbf_R_GRe', 'pcasl_jlf_cbf_L_GRe', 'pcasl_jlf_cbf_R_IOG', 'pcasl_jlf_cbf_L_IOG', 'pcasl_jlf_cbf_R_ITG', 'pcasl_jlf_cbf_L_ITG', 'pcasl_jlf_cbf_R_LiG', 'pcasl_jlf_cbf_L_LiG', 'pcasl_jlf_cbf_R_LOrG', 'pcasl_jlf_cbf_L_LOrG', 'pcasl_jlf_cbf_R_MCgG', 'pcasl_jlf_cbf_L_MCgG', 'pcasl_jlf_cbf_R_MFC', 'pcasl_jlf_cbf_L_MFC', 'pcasl_jlf_cbf_R_MFG', 'pcasl_jlf_cbf_L_MFG', 'pcasl_jlf_cbf_R_MOG', 'pcasl_jlf_cbf_L_MOG', 'pcasl_jlf_cbf_R_MOrG',  'pcasl_jlf_cbf_L_MOrG', 'pcasl_jlf_cbf_R_MPoG', 'pcasl_jlf_cbf_L_MPoG', 'pcasl_jlf_cbf_R_MPrG', 'pcasl_jlf_cbf_L_MPrG', 'pcasl_jlf_cbf_R_MSFG', 'pcasl_jlf_cbf_L_MSFG', 'pcasl_jlf_cbf_R_MTG', 'pcasl_jlf_cbf_R_OCP', 'pcasl_jlf_cbf_L_OCP', 'pcasl_jlf_cbf_R_OFuG', 'pcasl_jlf_cbf_L_OFuG', 'pcasl_jlf_cbf_R_OpIFG', 'pcasl_jlf_cbf_L_OpIFG', 'pcasl_jlf_cbf_R_OrIFG', 'pcasl_jlf_cbf_L_OrIFG', 'pcasl_jlf_cbf_R_PCgG', 'pcasl_jlf_cbf_L_PCgG', 'pcasl_jlf_cbf_R_PCu', 'pcasl_jlf_cbf_L_PCu', 'pcasl_jlf_cbf_R_PHG', 'pcasl_jlf_cbf_L_PHG', 'pcasl_jlf_cbf_R_PIns', 'pcasl_jlf_cbf_L_PIns', 'pcasl_jlf_cbf_R_PO', 'pcasl_jlf_cbf_L_PO', 'pcasl_jlf_cbf_R_PoG', 'pcasl_jlf_cbf_L_PoG', 'pcasl_jlf_cbf_R_POrG', 'pcasl_jlf_cbf_L_POrG', 'pcasl_jlf_cbf_R_PP', 'pcasl_jlf_cbf_L_PP', 'pcasl_jlf_cbf_R_PrG', 'pcasl_jlf_cbf_L_PrG', 'pcasl_jlf_cbf_R_PT', 'pcasl_jlf_cbf_L_PT', 'pcasl_jlf_cbf_R_SCA', 'pcasl_jlf_cbf_L_SCA', 'pcasl_jlf_cbf_R_SFG', 'pcasl_jlf_cbf_L_SFG', 'pcasl_jlf_cbf_R_SMC', 'pcasl_jlf_cbf_L_SMC', 'pcasl_jlf_cbf_R_SMG', 'pcasl_jlf_cbf_L_SMG', 'pcasl_jlf_cbf_R_SOG', 'pcasl_jlf_cbf_L_SOG', 'pcasl_jlf_cbf_R_SPL', 'pcasl_jlf_cbf_L_SPL', 'pcasl_jlf_cbf_R_STG', 'pcasl_jlf_cbf_L_STG', 'pcasl_jlf_cbf_R_TMP', 'pcasl_jlf_cbf_L_TMP', 'pcasl_jlf_cbf_R_TrIFG', 'pcasl_jlf_cbf_L_TrIFG', 'pcasl_jlf_cbf_R_TTG', 'pcasl_jlf_cbf_L_TTG')

cortvars <- c('mprage_jlf_ct_R_ACgG', 'mprage_jlf_ct_L_ACgG', 'mprage_jlf_ct_R_AIns', 'mprage_jlf_ct_L_AIns', 'mprage_jlf_ct_R_AOrG', 'mprage_jlf_ct_L_AOrG', 'mprage_jlf_ct_R_AnG', 'mprage_jlf_ct_L_AnG', 'mprage_jlf_ct_R_Calc', 'mprage_jlf_ct_L_Calc', 'mprage_jlf_ct_R_CO', 'mprage_jlf_ct_L_CO', 'mprage_jlf_ct_R_Cun', 'mprage_jlf_ct_L_Cun', 'mprage_jlf_ct_R_Ent', 'mprage_jlf_ct_L_Ent', 'mprage_jlf_ct_R_FO', 'mprage_jlf_ct_L_FO', 'mprage_jlf_ct_R_FRP', 'mprage_jlf_ct_L_FRP', 'mprage_jlf_ct_R_FuG', 'mprage_jlf_ct_L_FuG', 'mprage_jlf_ct_R_GRe', 'mprage_jlf_ct_L_GRe', 'mprage_jlf_ct_R_IOG', 'mprage_jlf_ct_L_IOG', 'mprage_jlf_ct_R_ITG', 'mprage_jlf_ct_L_ITG', 'mprage_jlf_ct_R_LiG', 'mprage_jlf_ct_L_LiG', 'mprage_jlf_ct_R_LOrG', 'mprage_jlf_ct_L_LOrG', 'mprage_jlf_ct_R_MCgG', 'mprage_jlf_ct_L_MCgG', 'mprage_jlf_ct_R_MFC', 'mprage_jlf_ct_L_MFC', 'mprage_jlf_ct_R_MFG', 'mprage_jlf_ct_L_MFG', 'mprage_jlf_ct_R_MOG', 'mprage_jlf_ct_L_MOG', 'mprage_jlf_ct_R_MOrG', 'mprage_jlf_ct_L_MOrG', 'mprage_jlf_ct_R_MPoG', 'mprage_jlf_ct_L_MPoG', 'mprage_jlf_ct_R_MPrG', 'mprage_jlf_ct_L_MPrG', 'mprage_jlf_ct_R_MSFG', 'mprage_jlf_ct_L_MSFG', 'mprage_jlf_ct_R_MTG', 'mprage_jlf_ct_L_MTG', 'mprage_jlf_ct_R_OCP', 'mprage_jlf_ct_L_OCP', 'mprage_jlf_ct_R_OFuG', 'mprage_jlf_ct_L_OFuG', 'mprage_jlf_ct_R_OpIFG', 'mprage_jlf_ct_L_OpIFG', 'mprage_jlf_ct_R_OrIFG', 'mprage_jlf_ct_L_OrIFG', 'mprage_jlf_ct_R_PCgG', 'mprage_jlf_ct_L_PCgG', 'mprage_jlf_ct_R_PCu', 'mprage_jlf_ct_L_PCu', 'mprage_jlf_ct_R_PHG', 'mprage_jlf_ct_L_PHG', 'mprage_jlf_ct_R_PIns', 'mprage_jlf_ct_L_PIns', 'mprage_jlf_ct_R_PO', 'mprage_jlf_ct_L_PO', 'mprage_jlf_ct_R_PoG', 'mprage_jlf_ct_L_PoG', 'mprage_jlf_ct_R_POrG', 'mprage_jlf_ct_L_POrG', 'mprage_jlf_ct_R_PP', 'mprage_jlf_ct_L_PP', 'mprage_jlf_ct_R_PrG', 'mprage_jlf_ct_L_PrG', 'mprage_jlf_ct_R_PT', 'mprage_jlf_ct_L_PT', 'mprage_jlf_ct_R_SCA', 'mprage_jlf_ct_L_SCA', 'mprage_jlf_ct_R_SFG', 'mprage_jlf_ct_L_SFG', 'mprage_jlf_ct_R_SMC', 'mprage_jlf_ct_L_SMC', 'mprage_jlf_ct_R_SMG', 'mprage_jlf_ct_L_SMG', 'mprage_jlf_ct_R_SOG', 'mprage_jlf_ct_L_SOG', 'mprage_jlf_ct_R_SPL', 'mprage_jlf_ct_L_SPL', 'mprage_jlf_ct_R_STG', 'mprage_jlf_ct_L_STG', 'mprage_jlf_ct_R_TMP', 'mprage_jlf_ct_L_TMP') # March 18, 2019: Last one was missing until today

gmdvars <- c('mprage_jlf_gmd_R_Accumbens_Area', 'mprage_jlf_gmd_L_Accumbens_Area', 'mprage_jlf_gmd_R_Amygdala', 'mprage_jlf_gmd_L_Amygdala',  'mprage_jlf_gmd_R_Caudate', 'mprage_jlf_gmd_L_Caudate', 'mprage_jlf_gmd_R_Cerebellum_Exterior', 'mprage_jlf_gmd_L_Cerebellum_Exterior', 'mprage_jlf_gmd_R_Hippocampus', 'mprage_jlf_gmd_L_Hippocampus', 'mprage_jlf_gmd_R_Pallidum', 'mprage_jlf_gmd_L_Pallidum', 'mprage_jlf_gmd_R_Putamen', 'mprage_jlf_gmd_L_Putamen', 'mprage_jlf_gmd_R_Thalamus_Proper', 'mprage_jlf_gmd_L_Thalamus_Proper', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VI.VII', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_gmd_R_ACgG', 'mprage_jlf_gmd_L_ACgG', 'mprage_jlf_gmd_R_AIns', 'mprage_jlf_gmd_L_AIns', 'mprage_jlf_gmd_R_AOrG', 'mprage_jlf_gmd_L_AOrG', 'mprage_jlf_gmd_R_AnG', 'mprage_jlf_gmd_L_AnG', 'mprage_jlf_gmd_R_Calc', 'mprage_jlf_gmd_L_Calc', 'mprage_jlf_gmd_R_CO', 'mprage_jlf_gmd_L_CO', 'mprage_jlf_gmd_R_Cun', 'mprage_jlf_gmd_L_Cun', 'mprage_jlf_gmd_R_Ent', 'mprage_jlf_gmd_L_Ent', 'mprage_jlf_gmd_R_FO', 'mprage_jlf_gmd_L_FO', 'mprage_jlf_gmd_R_FRP', 'mprage_jlf_gmd_L_FRP', 'mprage_jlf_gmd_R_FuG', 'mprage_jlf_gmd_L_FuG', 'mprage_jlf_gmd_R_GRe', 'mprage_jlf_gmd_L_GRe', 'mprage_jlf_gmd_R_IOG', 'mprage_jlf_gmd_L_IOG', 'mprage_jlf_gmd_R_ITG', 'mprage_jlf_gmd_L_ITG', 'mprage_jlf_gmd_R_LiG', 'mprage_jlf_gmd_L_LiG', 'mprage_jlf_gmd_R_LOrG', 'mprage_jlf_gmd_L_LOrG', 'mprage_jlf_gmd_R_MCgG', 'mprage_jlf_gmd_L_MCgG', 'mprage_jlf_gmd_R_MFC', 'mprage_jlf_gmd_L_MFC', 'mprage_jlf_gmd_R_MFG', 'mprage_jlf_gmd_L_MFG', 'mprage_jlf_gmd_R_MOG', 'mprage_jlf_gmd_L_MOG', 'mprage_jlf_gmd_R_MOrG', 'mprage_jlf_gmd_L_MOrG', 'mprage_jlf_gmd_R_MPoG', 'mprage_jlf_gmd_L_MPoG', 'mprage_jlf_gmd_R_MPrG', 'mprage_jlf_gmd_L_MPrG', 'mprage_jlf_gmd_R_MSFG', 'mprage_jlf_gmd_L_MSFG', 'mprage_jlf_gmd_R_MTG', 'mprage_jlf_gmd_L_MTG', 'mprage_jlf_gmd_R_OCP', 'mprage_jlf_gmd_L_OCP', 'mprage_jlf_gmd_R_OFuG', 'mprage_jlf_gmd_L_OFuG', 'mprage_jlf_gmd_R_OpIFG', 'mprage_jlf_gmd_L_OpIFG', 'mprage_jlf_gmd_R_OrIFG', 'mprage_jlf_gmd_L_OrIFG', 'mprage_jlf_gmd_R_PCgG', 'mprage_jlf_gmd_L_PCgG', 'mprage_jlf_gmd_R_PCu', 'mprage_jlf_gmd_L_PCu', 'mprage_jlf_gmd_R_PHG', 'mprage_jlf_gmd_L_PHG', 'mprage_jlf_gmd_R_PIns', 'mprage_jlf_gmd_L_PIns', 'mprage_jlf_gmd_R_PO', 'mprage_jlf_gmd_L_PO', 'mprage_jlf_gmd_R_PoG', 'mprage_jlf_gmd_L_PoG', 'mprage_jlf_gmd_R_POrG', 'mprage_jlf_gmd_L_POrG', 'mprage_jlf_gmd_R_PP', 'mprage_jlf_gmd_L_PP', 'mprage_jlf_gmd_R_PrG', 'mprage_jlf_gmd_L_PrG', 'mprage_jlf_gmd_R_PT', 'mprage_jlf_gmd_L_PT', 'mprage_jlf_gmd_R_SCA', 'mprage_jlf_gmd_L_SCA', 'mprage_jlf_gmd_R_SFG', 'mprage_jlf_gmd_L_SFG', 'mprage_jlf_gmd_R_SMC', 'mprage_jlf_gmd_L_SMC', 'mprage_jlf_gmd_R_SMG', 'mprage_jlf_gmd_L_SMG', 'mprage_jlf_gmd_R_SOG', 'mprage_jlf_gmd_L_SOG', 'mprage_jlf_gmd_R_SPL', 'mprage_jlf_gmd_L_SPL', 'mprage_jlf_gmd_R_STG', 'mprage_jlf_gmd_L_STG', 'mprage_jlf_gmd_R_TMP', 'mprage_jlf_gmd_L_TMP', 'mprage_jlf_gmd_R_TrIFG', 'mprage_jlf_gmd_L_TrIFG', 'mprage_jlf_gmd_R_TTG', 'mprage_jlf_gmd_L_TTG')

gmvvars <- c('mprage_jlf_vol_R_Accumbens_Area', 'mprage_jlf_vol_L_Accumbens_Area', 'mprage_jlf_vol_R_Amygdala', 'mprage_jlf_vol_L_Amygdala', 'mprage_jlf_vol_R_Caudate', 'mprage_jlf_vol_L_Caudate', 'mprage_jlf_vol_R_Cerebellum_Exterior', 'mprage_jlf_vol_L_Cerebellum_Exterior', 'mprage_jlf_vol_R_Hippocampus', 'mprage_jlf_vol_L_Hippocampus','mprage_jlf_vol_R_Pallidum', 'mprage_jlf_vol_L_Pallidum', 'mprage_jlf_vol_R_Putamen', 'mprage_jlf_vol_L_Putamen', 'mprage_jlf_vol_R_Thalamus_Proper', 'mprage_jlf_vol_L_Thalamus_Proper', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VI.VII', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_vol_R_ACgG', 'mprage_jlf_vol_L_ACgG', 'mprage_jlf_vol_R_AIns', 'mprage_jlf_vol_L_AIns', 'mprage_jlf_vol_R_AOrG', 'mprage_jlf_vol_L_AOrG', 'mprage_jlf_vol_R_AnG', 'mprage_jlf_vol_L_AnG', 'mprage_jlf_vol_R_Calc', 'mprage_jlf_vol_L_Calc', 'mprage_jlf_vol_R_CO', 'mprage_jlf_vol_L_CO', 'mprage_jlf_vol_R_Cun', 'mprage_jlf_vol_L_Cun', 'mprage_jlf_vol_R_Ent', 'mprage_jlf_vol_L_Ent', 'mprage_jlf_vol_R_FO', 'mprage_jlf_vol_L_FO', 'mprage_jlf_vol_R_FRP', 'mprage_jlf_vol_L_FRP', 'mprage_jlf_vol_R_FuG', 'mprage_jlf_vol_L_FuG', 'mprage_jlf_vol_R_GRe', 'mprage_jlf_vol_L_GRe', 'mprage_jlf_vol_R_IOG', 'mprage_jlf_vol_L_IOG', 'mprage_jlf_vol_R_ITG', 'mprage_jlf_vol_L_ITG', 'mprage_jlf_vol_R_LiG', 'mprage_jlf_vol_L_LiG', 'mprage_jlf_vol_R_LOrG', 'mprage_jlf_vol_L_LOrG', 'mprage_jlf_vol_R_MCgG', 'mprage_jlf_vol_L_MCgG', 'mprage_jlf_vol_R_MFC', 'mprage_jlf_vol_L_MFC', 'mprage_jlf_vol_R_MFG', 'mprage_jlf_vol_L_MFG', 'mprage_jlf_vol_R_MOG', 'mprage_jlf_vol_L_MOG', 'mprage_jlf_vol_R_MOrG', 'mprage_jlf_vol_L_MOrG', 'mprage_jlf_vol_R_MPoG', 'mprage_jlf_vol_L_MPoG', 'mprage_jlf_vol_R_MPrG', 'mprage_jlf_vol_L_MPrG', 'mprage_jlf_vol_R_MSFG', 'mprage_jlf_vol_L_MSFG', 'mprage_jlf_vol_R_MTG', 'mprage_jlf_vol_L_MTG', 'mprage_jlf_vol_R_OCP', 'mprage_jlf_vol_L_OCP', 'mprage_jlf_vol_R_OFuG', 'mprage_jlf_vol_L_OFuG', 'mprage_jlf_vol_R_OpIFG', 'mprage_jlf_vol_L_OpIFG', 'mprage_jlf_vol_R_OrIFG', 'mprage_jlf_vol_L_OrIFG', 'mprage_jlf_vol_R_PCgG', 'mprage_jlf_vol_L_PCgG', 'mprage_jlf_vol_R_PCu', 'mprage_jlf_vol_L_PCu', 'mprage_jlf_vol_R_PHG', 'mprage_jlf_vol_L_PHG', 'mprage_jlf_vol_R_PIns', 'mprage_jlf_vol_L_PIns', 'mprage_jlf_vol_R_PO', 'mprage_jlf_vol_L_PO', 'mprage_jlf_vol_R_PoG', 'mprage_jlf_vol_L_PoG', 'mprage_jlf_vol_R_POrG', 'mprage_jlf_vol_L_POrG', 'mprage_jlf_vol_R_PP', 'mprage_jlf_vol_L_PP', 'mprage_jlf_vol_R_PrG', 'mprage_jlf_vol_L_PrG', 'mprage_jlf_vol_R_PT', 'mprage_jlf_vol_L_PT', 'mprage_jlf_vol_R_SCA', 'mprage_jlf_vol_L_SCA', 'mprage_jlf_vol_R_SFG', 'mprage_jlf_vol_L_SFG', 'mprage_jlf_vol_R_SMC', 'mprage_jlf_vol_L_SMC', 'mprage_jlf_vol_R_SMG', 'mprage_jlf_vol_L_SMG', 'mprage_jlf_vol_R_SOG', 'mprage_jlf_vol_L_SOG', 'mprage_jlf_vol_R_SPL', 'mprage_jlf_vol_L_SPL', 'mprage_jlf_vol_R_STG', 'mprage_jlf_vol_L_STG', 'mprage_jlf_vol_R_TMP', 'mprage_jlf_vol_L_TMP', 'mprage_jlf_vol_R_TrIFG', 'mprage_jlf_vol_L_TrIFG', 'mprage_jlf_vol_R_TTG', 'mprage_jlf_vol_L_TTG')

mdvars <- c('dti_jlf_tr_R_Accumbens_Area','dti_jlf_tr_L_Accumbens_Area', 'dti_jlf_tr_R_Amygdala', 'dti_jlf_tr_L_Amygdala', 'dti_jlf_tr_R_Caudate', 'dti_jlf_tr_L_Caudate', 'dti_jlf_tr_R_Cerebellum_Exterior', 'dti_jlf_tr_L_Cerebellum_Exterior', 'dti_jlf_tr_R_Hippocampus', 'dti_jlf_tr_L_Hippocampus', 'dti_jlf_tr_R_Pallidum', 'dti_jlf_tr_L_Pallidum', 'dti_jlf_tr_R_Putamen', 'dti_jlf_tr_L_Putamen', 'dti_jlf_tr_R_Thalamus_Proper', 'dti_jlf_tr_L_Thalamus_Proper', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_I.V', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VI.VII', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VIII.X', 'dti_jlf_tr_R_ACgG', 'dti_jlf_tr_L_ACgG', 'dti_jlf_tr_R_AIns', 'dti_jlf_tr_L_AIns', 'dti_jlf_tr_R_AOrG', 'dti_jlf_tr_L_AOrG', 'dti_jlf_tr_R_AnG', 'dti_jlf_tr_L_AnG', 'dti_jlf_tr_R_Calc', 'dti_jlf_tr_L_Calc', 'dti_jlf_tr_R_CO', 'dti_jlf_tr_L_CO', 'dti_jlf_tr_R_Cun', 'dti_jlf_tr_L_Cun', 'dti_jlf_tr_R_Ent', 'dti_jlf_tr_L_Ent', 'dti_jlf_tr_R_FO', 'dti_jlf_tr_L_FO', 'dti_jlf_tr_R_FRP', 'dti_jlf_tr_L_FRP', 'dti_jlf_tr_R_FuG', 'dti_jlf_tr_L_FuG', 'dti_jlf_tr_R_GRe', 'dti_jlf_tr_L_GRe', 'dti_jlf_tr_R_IOG', 'dti_jlf_tr_L_IOG', 'dti_jlf_tr_R_ITG', 'dti_jlf_tr_L_ITG', 'dti_jlf_tr_R_LiG', 'dti_jlf_tr_L_LiG', 'dti_jlf_tr_R_LOrG', 'dti_jlf_tr_L_LOrG', 'dti_jlf_tr_R_MCgG', 'dti_jlf_tr_L_MCgG', 'dti_jlf_tr_R_MFC', 'dti_jlf_tr_L_MFC', 'dti_jlf_tr_R_MFG', 'dti_jlf_tr_L_MFG', 'dti_jlf_tr_R_MOG', 'dti_jlf_tr_L_MOG', 'dti_jlf_tr_R_MOrG', 'dti_jlf_tr_L_MOrG', 'dti_jlf_tr_R_MPoG', 'dti_jlf_tr_L_MPoG', 'dti_jlf_tr_R_MPrG', 'dti_jlf_tr_L_MPrG', 'dti_jlf_tr_R_MSFG', 'dti_jlf_tr_L_MSFG', 'dti_jlf_tr_R_MTG', 'dti_jlf_tr_L_MTG', 'dti_jlf_tr_R_OCP', 'dti_jlf_tr_L_OCP', 'dti_jlf_tr_R_OFuG', 'dti_jlf_tr_L_OFuG', 'dti_jlf_tr_R_OpIFG', 'dti_jlf_tr_L_OpIFG', 'dti_jlf_tr_R_OrIFG', 'dti_jlf_tr_L_OrIFG', 'dti_jlf_tr_R_PCgG', 'dti_jlf_tr_L_PCgG', 'dti_jlf_tr_R_PCu', 'dti_jlf_tr_L_PCu', 'dti_jlf_tr_R_PHG', 'dti_jlf_tr_L_PHG', 'dti_jlf_tr_R_PIns', 'dti_jlf_tr_L_PIns', 'dti_jlf_tr_R_PO', 'dti_jlf_tr_L_PO', 'dti_jlf_tr_R_PoG', 'dti_jlf_tr_L_PoG', 'dti_jlf_tr_R_POrG', 'dti_jlf_tr_L_POrG', 'dti_jlf_tr_R_PP', 'dti_jlf_tr_L_PP', 'dti_jlf_tr_R_PrG', 'dti_jlf_tr_L_PrG', 'dti_jlf_tr_R_PT', 'dti_jlf_tr_L_PT', 'dti_jlf_tr_R_SCA', 'dti_jlf_tr_L_SCA', 'dti_jlf_tr_R_SFG', 'dti_jlf_tr_L_SFG', 'dti_jlf_tr_R_SMC', 'dti_jlf_tr_L_SMC', 'dti_jlf_tr_R_SMG', 'dti_jlf_tr_L_SMG', 'dti_jlf_tr_R_SOG', 'dti_jlf_tr_L_SOG', 'dti_jlf_tr_R_SPL', 'dti_jlf_tr_L_SPL', 'dti_jlf_tr_R_STG', 'dti_jlf_tr_L_STG', 'dti_jlf_tr_R_TMP', 'dti_jlf_tr_L_TMP', 'dti_jlf_tr_R_TrIFG', 'dti_jlf_tr_L_TrIFG')

rehovars <- c('rest_jlf_reho_R_Accumbens_Area', 'rest_jlf_reho_L_Accumbens_Area', 'rest_jlf_reho_R_Amygdala', 'rest_jlf_reho_L_Amygdala', 'rest_jlf_reho_R_Caudate', 'rest_jlf_reho_L_Caudate', 'rest_jlf_reho_R_Cerebellum_Exterior', 'rest_jlf_reho_L_Cerebellum_Exterior', 'rest_jlf_reho_R_Hippocampus', 'rest_jlf_reho_L_Hippocampus', 'rest_jlf_reho_R_Pallidum', 'rest_jlf_reho_L_Pallidum', 'rest_jlf_reho_R_Putamen', 'rest_jlf_reho_L_Putamen', 'rest_jlf_reho_R_Thalamus_Proper', 'rest_jlf_reho_L_Thalamus_Proper', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_I.V', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_VI.VII', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_VIII.X', 'rest_jlf_reho_R_ACgG', 'rest_jlf_reho_L_ACgG', 'rest_jlf_reho_R_AIns', 'rest_jlf_reho_L_AIns', 'rest_jlf_reho_R_AOrG', 'rest_jlf_reho_L_AOrG', 'rest_jlf_reho_R_AnG', 'rest_jlf_reho_L_AnG', 'rest_jlf_reho_R_Calc', 'rest_jlf_reho_L_Calc', 'rest_jlf_reho_R_CO', 'rest_jlf_reho_L_CO', 'rest_jlf_reho_R_Cun', 'rest_jlf_reho_L_Cun', 'rest_jlf_reho_R_Ent', 'rest_jlf_reho_L_Ent', 'rest_jlf_reho_R_FO', 'rest_jlf_reho_L_FO', 'rest_jlf_reho_R_FRP', 'rest_jlf_reho_L_FRP', 'rest_jlf_reho_R_FuG', 'rest_jlf_reho_L_FuG', 'rest_jlf_reho_R_GRe', 'rest_jlf_reho_L_GRe', 'rest_jlf_reho_R_IOG', 'rest_jlf_reho_L_IOG', 'rest_jlf_reho_R_ITG', 'rest_jlf_reho_L_ITG', 'rest_jlf_reho_R_LiG', 'rest_jlf_reho_L_LiG', 'rest_jlf_reho_R_LOrG', 'rest_jlf_reho_L_LOrG', 'rest_jlf_reho_R_MCgG', 'rest_jlf_reho_L_MCgG', 'rest_jlf_reho_R_MFC', 'rest_jlf_reho_L_MFC', 'rest_jlf_reho_R_MFG', 'rest_jlf_reho_L_MFG', 'rest_jlf_reho_R_MOG', 'rest_jlf_reho_L_MOG', 'rest_jlf_reho_R_MOrG', 'rest_jlf_reho_L_MOrG', 'rest_jlf_reho_R_MPoG', 'rest_jlf_reho_L_MPoG', 'rest_jlf_reho_R_MPrG', 'rest_jlf_reho_L_MPrG', 'rest_jlf_reho_R_MSFG', 'rest_jlf_reho_L_MSFG', 'rest_jlf_reho_R_MTG', 'rest_jlf_reho_L_MTG', 'rest_jlf_reho_R_OCP', 'rest_jlf_reho_L_OCP', 'rest_jlf_reho_R_OFuG', 'rest_jlf_reho_L_OFuG', 'rest_jlf_reho_R_OpIFG', 'rest_jlf_reho_L_OpIFG', 'rest_jlf_reho_R_OrIFG', 'rest_jlf_reho_L_OrIFG', 'rest_jlf_reho_R_PCgG', 'rest_jlf_reho_L_PCgG', 'rest_jlf_reho_R_PCu', 'rest_jlf_reho_L_PCu', 'rest_jlf_reho_R_PHG', 'rest_jlf_reho_L_PHG', 'rest_jlf_reho_R_PIns', 'rest_jlf_reho_L_PIns', 'rest_jlf_reho_R_PO', 'rest_jlf_reho_L_PO', 'rest_jlf_reho_R_PoG', 'rest_jlf_reho_L_PoG', 'rest_jlf_reho_R_POrG', 'rest_jlf_reho_L_POrG', 'rest_jlf_reho_R_PP', 'rest_jlf_reho_L_PP', 'rest_jlf_reho_R_PrG', 'rest_jlf_reho_L_PrG', 'rest_jlf_reho_R_PT', 'rest_jlf_reho_L_PT', 'rest_jlf_reho_R_SCA', 'rest_jlf_reho_L_SCA', 'rest_jlf_reho_R_SFG', 'rest_jlf_reho_L_SFG', 'rest_jlf_reho_R_SMC', 'rest_jlf_reho_L_SMC', 'rest_jlf_reho_R_SMG', 'rest_jlf_reho_L_SMG', 'rest_jlf_reho_R_SOG', 'rest_jlf_reho_L_SOG', 'rest_jlf_reho_R_SPL', 'rest_jlf_reho_L_SPL', 'rest_jlf_reho_R_STG', 'rest_jlf_reho_L_STG', 'rest_jlf_reho_R_TMP', 'rest_jlf_reho_L_TMP', 'rest_jlf_reho_R_TrIFG', 'rest_jlf_reho_L_TrIFG', 'rest_jlf_reho_R_TTG', 'rest_jlf_reho_L_TTG')

favars <- c('dti_dtitk_jhulabel_fa_mcp', 'dti_dtitk_jhulabel_fa_pct', 'dti_dtitk_jhulabel_fa_gcc', 'dti_dtitk_jhulabel_fa_bcc', 'dti_dtitk_jhulabel_fa_scc', 'dti_dtitk_jhulabel_fa_fnx', 'dti_dtitk_jhulabel_fa_cst_r', 'dti_dtitk_jhulabel_fa_cst_l', 'dti_dtitk_jhulabel_fa_mel_l', 'dti_dtitk_jhulabel_fa_mel_r', 'dti_dtitk_jhulabel_fa_icp_r', 'dti_dtitk_jhulabel_fa_icp_l', 'dti_dtitk_jhulabel_fa_scp_r', 'dti_dtitk_jhulabel_fa_scp_l', 'dti_dtitk_jhulabel_fa_cp_r', 'dti_dtitk_jhulabel_fa_cp_l', 'dti_dtitk_jhulabel_fa_alic_r', 'dti_dtitk_jhulabel_fa_alic_l', 'dti_dtitk_jhulabel_fa_plic_r', 'dti_dtitk_jhulabel_fa_plic_l', 'dti_dtitk_jhulabel_fa_rlic_r', 'dti_dtitk_jhulabel_fa_rlic_l', 'dti_dtitk_jhulabel_fa_acr_r', 'dti_dtitk_jhulabel_fa_acr_l', 'dti_dtitk_jhulabel_fa_scr_r', 'dti_dtitk_jhulabel_fa_scr_l', 'dti_dtitk_jhulabel_fa_pcr_r', 'dti_dtitk_jhulabel_fa_pcr_l', 'dti_dtitk_jhulabel_fa_ptr_r', 'dti_dtitk_jhulabel_fa_ptr_l', 'dti_dtitk_jhulabel_fa_ss_r', 'dti_dtitk_jhulabel_fa_ss_l', 'dti_dtitk_jhulabel_fa_ec_r', 'dti_dtitk_jhulabel_fa_ec_l', 'dti_dtitk_jhulabel_fa_cgc_r', 'dti_dtitk_jhulabel_fa_cgc_l', 'dti_dtitk_jhulabel_fa_cgh_r', 'dti_dtitk_jhulabel_fa_cgh_l', 'dti_dtitk_jhulabel_fa_fnx_st_r', 'dti_dtitk_jhulabel_fa_fnx_st_l', 'dti_dtitk_jhulabel_fa_slf_r', 'dti_dtitk_jhulabel_fa_slf_l', 'dti_dtitk_jhulabel_fa_sfo_r', 'dti_dtitk_jhulabel_fa_sfo_l', 'dti_dtitk_jhulabel_fa_uf_r', 'dti_dtitk_jhulabel_fa_uf_l', 'dti_dtitk_jhulabel_fa_tap_r', 'dti_dtitk_jhulabel_fa_tap_l')

yvar <- c('ageAtScan1')


####### psTerminal construction
clinical <- read.csv("/home/butellyn/longitudinal_psychosis/data/pnc_diagnosis_categorical_20170526.csv")
demo <- read.csv("/home/butellyn/longitudinal_psychosis/data/n2416_demographics_20170310.csv")
demo <- demo[,c("bblid", "scanid", "scanageMonths", "sex", "race", "DOSCAN")]
demo$DOSCAN <- as.Date(demo$DOSCAN, format = "%m/%d/%y")

clinical$bblid <- as.factor(clinical$bblid)
clinical$dodiagnosis_t1 <- as.Date(clinical$dodiagnosis_t1, format = "%m/%d/%y")
clinical$dodiagnosis_t2 <- as.Date(clinical$dodiagnosis_t2, format = "%m/%d/%y")
clinical$dodiagnosis_t3 <- as.Date(clinical$dodiagnosis_t3, format = "%m/%d/%y")
clinical$dodiagnosis_t4 <- as.Date(clinical$dodiagnosis_t4, format = "%m/%d/%y")

# PS vs. TD Terminal
demo$dx_ps <- NA
for (row in 1:nrow(demo)) {
	doscan <- demo[row, "DOSCAN"]
	bblid <- demo[row, "bblid"]
	dodiagnoses <- clinical[clinical$bblid == bblid, c("dodiagnosis_t1", "dodiagnosis_t2", "dodiagnosis_t3", "dodiagnosis_t4")]
	if (dim(dodiagnoses)[1] != 0) {
		if (!(is.na(dodiagnoses[["dodiagnosis_t1"]]))) { 
			doscan_to_dodiag_t1 <- abs(as.numeric(as.Date(doscan, format="%Y-%m-%d") - as.Date(dodiagnoses[["dodiagnosis_t1"]], format="%Y-%m-%d")))
		} else { doscan_to_dodiag_t1 <- NA }
		if (!(is.na(dodiagnoses[["dodiagnosis_t2"]]))) { 
			doscan_to_dodiag_t2 <- abs(as.numeric(as.Date(doscan, format="%Y-%m-%d") - as.Date(dodiagnoses[["dodiagnosis_t2"]], format="%Y-%m-%d")))
		} else { doscan_to_dodiag_t2 <- NA }
		if (!(is.na(dodiagnoses[["dodiagnosis_t3"]]))) { 
			doscan_to_dodiag_t3 <- abs(as.numeric(as.Date(doscan, format="%Y-%m-%d") - as.Date(dodiagnoses[["dodiagnosis_t3"]], format="%Y-%m-%d")))
		} else { doscan_to_dodiag_t3 <- NA }
		if (!(is.na(dodiagnoses[["dodiagnosis_t4"]]))) { 
			doscan_to_dodiag_t4 <- abs(as.numeric(as.Date(doscan, format="%Y-%m-%d") - as.Date(dodiagnoses[["dodiagnosis_t4"]], format="%Y-%m-%d")))
		} else { doscan_to_dodiag_t4 <- NA }
		
		# pt
		diagnosiscol <- "dx_t1_psychopathology"
		if (!(is.na(dodiagnoses[["dodiagnosis_t2"]]))) { if (doscan_to_dodiag_t1 > doscan_to_dodiag_t2) { diagnosiscol <- "dx_t2_psychopathology" }}
		if (!(is.na(dodiagnoses[["dodiagnosis_t3"]]))) { if (doscan_to_dodiag_t2 > doscan_to_dodiag_t3) { diagnosiscol <- "dx_t3_psychopathology" }}
		if (!(is.na(dodiagnoses[["dodiagnosis_t4"]]))) { if (doscan_to_dodiag_t3 > doscan_to_dodiag_t4) { diagnosiscol <- "dx_t4_psychopathology" }}
		
		diagnosis_pt <- as.character(clinical[clinical$bblid == bblid, diagnosiscol])

		# ps
		diagnosiscol <- "dx_t1_psychosis"
		if (!(is.na(dodiagnoses[["dodiagnosis_t2"]]))) { if (doscan_to_dodiag_t1 > doscan_to_dodiag_t2) { diagnosiscol <- "dx_t2_psychosis" }}
		if (!(is.na(dodiagnoses[["dodiagnosis_t3"]]))) { if (doscan_to_dodiag_t2 > doscan_to_dodiag_t3) { diagnosiscol <- "dx_t3_psychosis" }}
		if (!(is.na(dodiagnoses[["dodiagnosis_t4"]]))) { if (doscan_to_dodiag_t3 > doscan_to_dodiag_t4) { diagnosiscol <- "dx_t4_psychosis" }}
		
		diagnosis_ps <- as.character(clinical[clinical$bblid == bblid, diagnosiscol])

		# change PTs to PSs where appropriate ####### KOSHA SHOULD I DO THIS?
		if (diagnosis_pt == "PT" & diagnosis_ps == "PS") { diagnosis <- "PS" 
		} else if (diagnosis_pt == "PT" & diagnosis_ps != "PS") { diagnosis <- "OP"
		} else if (diagnosis_pt == "PT" & diagnosis_ps == "NONPS") { print(paste(bblid, "eek!"))
		} else if (diagnosis_pt == "TD" & diagnosis_ps == "NONPS") { diagnosis <- "TD" }

		demo[row, "dx_ps"] <- diagnosis
	}
}
demo$dx_ps <- as.factor(demo$dx_ps)

demo$Age <- demo$scanageMonths/12
demo$bblid <- as.factor(demo$bblid)
for (bblid in levels(demo$bblid)) {
	tmp <- demo[demo$bblid == bblid, ]
	lasttimerow <- rownames(tmp[tmp$Age == max(tmp$Age),])
	lastdiag <- as.character(tmp[lasttimerow, "dx_ps"])
	demo[demo$bblid == bblid, "psTerminal"] <- lastdiag
}
demo$psTerminal <- as.factor(demo$psTerminal)
#demo <- demo[unique(demo$bblid),c("bblid", "psTerminal")]

# now integrate with df
df$bblid <- as.factor(df$bblid)
df$psTerminal <- NA
for (i in 1:nrow(df)) { 
	bblid <- as.character(df[i, "bblid"])
	if (bblid %in% demo$bblid) { 
		if (!(is.na(as.character(demo[demo$bblid == bblid, "psTerminal"][1])))) {
			df[i, "psTerminal"] <- as.character(demo[demo$bblid == bblid, "psTerminal"][1])
		} else { 
			df[i, "psTerminal"] <- as.character(df[i, "goassessDxpmr7"][1]) 
		}
	} else { 
		df[i, "psTerminal"] <- as.character(df[i, "goassessDxpmr7"][1]) 
	}
}
df$psTerminal <- factor(df$psTerminal)



# Start analyses
set.seed(1)

######################################################## Vars ########################################################

if (type == "cort") {
	xvars <- cortvars
	fullname <- "Cortical Thickness"
} else if (type == "gmv") {
	xvars <- gmvvars
	fullname <- "Gray Matter Volume"
} else if (type == "gmd") {
	xvars <- gmdvars
	fullname <- "Gray Matter Density"
} else if (type == "fa") {
	xvars <- favars
	fullname <- "Fractional Anisotropy"
} else if (type == "md") {
	xvars <- mdvars
	fullname <- "Mean Diffusivity"
} else if (type == "reho") {
	xvars <- rehovars
	fullname <- "Regional Homogeneity"
} else if (type == "cbf") {
	xvars <- cbfvars
	fullname <- "Cerebral Blood Flow"
} else if (type == "alff") {
	xvars <- alffvars
	fullname <- "Amplitude of Low Frequency Fluctuations"
} 

############################ Females ############################

df_F <- df[df$sex == 2, c("bblid", "psTerminal", xvars, yvar)]
df_F$bblid <- as.factor(df_F$bblid)

for (i in 1:ncol(df_F)) {
	na_index <- is.na(df_F[,i])
	df_F <- df_F[!na_index,]
}

td_F <- df_F[df_F$psTerminal == "TD",]
rownames(td_F) <- 1:nrow(td_F)

folds5 <- createFolds(td_F$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5[[i]]
	trainfolds <- subset(folds5,!(grepl(i, names(folds5))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_td_train_F_", i), td_F[train, c("bblid", xvars)])
	assign(paste0("x_td_test_F_", i), td_F[test, c("bblid", xvars)])
	assign(paste0("y_td_train_F_", i), td_F[train, c("bblid", yvar)])
	assign(paste0("y_td_test_F_", i), td_F[test, c("bblid", yvar)])
}

#####Elastic Net: out of sample predictions for TD's
for (i in 1:5) {
	x_train_df <- get(paste0("x_td_train_F_", i))
	x_train_input <- x_train_df[,xvars]
	y_train_df <- get(paste0("y_td_train_F_", i))
	y_train_input <- y_train_df$ageAtScan1
	ageAtScan1 <- y_train_input
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = FALSE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_td_F_", i), elastic_net_model)
	x_test_df <- get(paste0("x_td_test_F_", i))
	y_test_df <- get(paste0("y_td_test_F_", i))
	y_test_predicted <- predict(elastic_net_model, x_test_df)
	if (i == 1) { 
		predicted_values <- merge(cbind(x_test_df, y_test_predicted), y_test_df)
		coefs <- coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)# Added March 11, 2019
	} else { 
		predicted_values <- rbind(predicted_values, merge(cbind(x_test_df, y_test_predicted), y_test_df))
		coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda))# Added March 11, 2019
	}
}
names(predicted_values)[names(predicted_values) == 'y_test_predicted'] <- 'predicted_ageAtScan1'

#####Elastic Net: predictions for PS's 
# train on full TD set
x_td_train_input <- td_F[,xvars]
ageAtScan1 <- td_F[,yvar]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = FALSE)
elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)) # Added March 11, 2019
coefs <- as.matrix(coefs) # Added March 11, 2019
coefs_F <- data.frame(coefs) # Added March 11, 2019
colnames(coefs_F) <- c("First_Fold", "Second_Fold", "Third_Fold", "Fourth_Fold", "Fifth_Fold", "Whole_Sample") # Added March 11, 2019

# predict on PS's
x_ps_F <- df_F[df_F$psTerminal == "PS", c("bblid", xvars)]
y_ps_F <- df_F[df_F$psTerminal == "PS", c("bblid", yvar)]
x_ps_input <- x_ps_F[,xvars]
y_ps_input <- y_ps_F[,yvar]
ageAtScan1 <- y_ps_input
y_ps_predicted <- predict(elastic_net_model, x_ps_input)
x_ps_F$predicted_ageAtScan1 <- y_ps_predicted ### March 12, 2019: THIS HAS TO BE HERE

# create final dfs
df_td_F <- predicted_values
df_ps_F <- merge(x_ps_F, y_ps_F) #####

# get predictions for PSs based on five models for TDs # Added March 11, 2019
for (i in 1:5) {
	elastic_net_model <- get(paste0("mod_td_F_", i))
	df_ps_F[,paste0("fold_", i, "_ageAtScan1")] <- predict(elastic_net_model, df_ps_F[,xvars]) ### March 12, 2019: THIS HAS TO BE df_ps_F, NOT x_ps_F
}

# Create years age variables
df_ps_F$real_age <- df_ps_F$ageAtScan1/12
df_ps_F$pred_age <- df_ps_F$predicted_ageAtScan1/12
df_td_F$real_age <- df_td_F$ageAtScan1/12
df_td_F$pred_age <- df_td_F$predicted_ageAtScan1/12

# Create the final dataframe
df_ps_F$diagnosis <- "PS"
df_td_F$diagnosis <- "TD"
df_both_F <- rbind(df_ps_F[,!(names(df_ps_F) %in% c("fold_1_ageAtScan1", "fold_2_ageAtScan1", "fold_3_ageAtScan1", "fold_4_ageAtScan1", "fold_5_ageAtScan1"))], df_td_F)
df_both_F$diagnosis <- as.factor(df_both_F$diagnosis)

### Perform LMEM
#df_long_F <- melt(df_both_F[, !(names(df_both_F) %in% c("predicted_ageAtScan1", "ageAtScan1", "pred_age", "age_cat", "pred_adult", "age_cat_3"))], #id.vars=c("bblid","diagnosis","real_age"))
#df_long_F <- transform(df_long_F, value=scale(value))
#lmem_mod <- lmer(real_age ~ (variable + value + diagnosis)^3 + (1 | bblid), data=df_long_F)
#lme_mod <- lme(real_age ~ (variable + value + diagnosis)^3, random=~1|bblid, data=df_long_F)
#lmem_mod.all <- allFit(lmem_mod)

### Look for interactions... Age ~ xvar*diagnosis
# all
siginter_all_F <- as.data.frame(matrix(0, ncol=4, nrow=length(xvars)))
colnames(siginter_all_F) <- c("ROI", "P_Value", "Sig", "InterEstPos")
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	f <- paste0("real_age ~ ", xvar, " + diagnosis + diagnosis:", xvar)
	mod <- lm(f, data=df_both_F)
	siginter_all_F[i,1] <- xvar
	siginter_all_F[i,2] <- round(anova(mod)[["Pr(>F)"]][[3]], digits=3)
	if (anova(mod)[["Pr(>F)"]][[3]] < .05) { siginter_all_F[i,3] <- "*" } else { siginter_all_F[i,3] <- "" }
	if (mod$coefficients[[4]] < 0) { siginter_all_F[i, "InterEstPos"] <- "Neg" } else { siginter_all_F[i, "InterEstPos"] <- "Pos" }
}
siginter_all_F$ROI <- as.factor(siginter_all_F$ROI)
numsig_all_F <- paste0(nrow(siginter_all_F[siginter_all_F$Sig == "*",]), " out of ", nrow(siginter_all_F), " interactions with p < .05\n", nrow(siginter_all_F[siginter_all_F$Sig == "*" & siginter_all_F$InterEstPos == "Pos",]), " of these have a positive coefficient estimate")

interaction_all_F <- ggplot(data=siginter_all_F, aes(P_Value, color=Sig, fill=Sig)) +
	geom_histogram(bins=20, alpha=0.5) + xlab("P-Value") +
	ggtitle("P-Values for Diagnosis:ROI (all subjects)") +
	labs(subtitle=numsig_all_F) + theme_minimal() +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 


# over 10
siginter_over10_F <- as.data.frame(matrix(0, ncol=4, nrow=length(xvars)))
colnames(siginter_over10_F) <- c("ROI", "P_Value", "Sig", "InterEstPos")
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	f <- paste0("real_age ~ ", xvar, " + diagnosis + diagnosis:", xvar)
	mod <- lm(f, data=df_both_F[df_both_F$real_age >= 10, ])
	siginter_over10_F[i,1] <- xvar
	siginter_over10_F[i,2] <- round(anova(mod)[["Pr(>F)"]][[3]], digits=3)
	if (anova(mod)[["Pr(>F)"]][[3]] < .05) { siginter_over10_F[i,3] <- "*" } else { siginter_over10_F[i,3] <- "" }
	if (mod$coefficients[[4]] < 0) { siginter_over10_F[i, "InterEstPos"] <- "Neg" } else { siginter_over10_F[i, "InterEstPos"] <- "Pos" }
}
siginter_over10_F$ROI <- as.factor(siginter_over10_F$ROI)
numsig_over10_F <- paste0(nrow(siginter_over10_F[siginter_over10_F$Sig == "*",]), " out of ", nrow(siginter_over10_F), " interactions with p < .05\n", nrow(siginter_over10_F[siginter_over10_F$Sig == "*" & siginter_over10_F$InterEstPos == "Pos",]), " of these have a positive coefficient estimate")

interaction_over10_F <- ggplot(data=siginter_over10_F, aes(P_Value, color=Sig, fill=Sig)) +
	geom_histogram(bins=20, alpha=0.5) + xlab("P-Value") +
	ggtitle("P-Values for Diagnosis:ROI (Age > 10)") +
	labs(subtitle=numsig_over10_F) + theme_minimal() +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

# Plot GAMMs
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	subtit <- paste0("ROI: ", xvar)
	gam_plot <- ggplot(data=df_both_F, aes_string("real_age", xvar, color="diagnosis")) +
		geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
		xlim(0,25) + labs(title = "Age~ROI", subtitle = subtit) + 
		xlab("Age") + ylab("ROI Value") + geom_smooth(se = TRUE, method= "gam", formula = y ~ log(x)) +
		theme(plot.title = element_text(family="Times", face="bold", size=18)) 
	assign(paste0("gamplot_", i, "_F"), gam_plot)
}

# Create age bins
df_both_F$age_cat <- 0
df_both_F$pred_adult <- 0

rownames(df_both_F) <- 1:nrow(df_both_F) # change row indices

for (row in 1:nrow(df_both_F)) {
	if (df_both_F[row,"real_age"] < 10) { df_both_F[row,"age_cat"] = "8_9" }
	else if (df_both_F[row,"real_age"] >= 10 & df_both_F[row,"real_age"] < 12) { df_both_F[row,"age_cat"] = "10_11" }
	else if (df_both_F[row,"real_age"] >= 12 & df_both_F[row,"real_age"] < 14) { df_both_F[row,"age_cat"] = "12_13" }
	else if (df_both_F[row,"real_age"] >= 14 & df_both_F[row,"real_age"] < 16) { df_both_F[row,"age_cat"] = "14_15" }
	else if (df_both_F[row,"real_age"] >= 16 & df_both_F[row,"real_age"] < 18) { df_both_F[row,"age_cat"] = "16_17" }
	else if (df_both_F[row,"real_age"] >= 18 & df_both_F[row,"real_age"] < 20) { df_both_F[row,"age_cat"] = "18_19" }
	else if (df_both_F[row,"real_age"] >= 20 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat"] = "20_23" }
	if (df_both_F[row,"pred_age"] >= 18) { df_both_F[row,"pred_adult"] = 1 }
}

df_both_F$age_cat <- as.factor(df_both_F$age_cat)

df_both_F$age_cat_3 <- 0

for (row in 1:nrow(df_both_F)) {
	if (df_both_F[row,"real_age"] < 12) { df_both_F[row,"age_cat_3"] = "8_11" }
	else if (df_both_F[row,"real_age"] >= 12 & df_both_F[row,"real_age"] < 15) { df_both_F[row,"age_cat_3"] = "12_14" }
	else if (df_both_F[row,"real_age"] >= 15 & df_both_F[row,"real_age"] < 18) { df_both_F[row,"age_cat_3"] = "15_17" }
	else if (df_both_F[row,"real_age"] >= 18 & df_both_F[row,"real_age"] < 21) { df_both_F[row,"age_cat_3"] = "18_20" }
	else if (df_both_F[row,"real_age"] >= 21 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat_3"] = "21_23" }
}

df_both_F$age_cat_3 <- as.factor(df_both_F$age_cat_3)

############# Summary Dataframes
### Two-Year Age Bins: % Adult
summary_2YBins_adultPer_F <- data.frame(matrix(0, nrow=14, ncol=7))
colnames(summary_2YBins_adultPer_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "Nadult_Ntotal", "Nadult", "Ntotal", "SE")
summary_2YBins_adultPer_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_2YBins_adultPer_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_2YBins_adultPer_F)) { #CIs from conf.level
	agebin <- summary_2YBins_adultPer_F[i, "Age_Category"]
	diag <- summary_2YBins_adultPer_F[i, "Diagnosis"]
	# Total number and number adult
	Nadult <- length(df_both_F[df_both_F$age_cat == agebin & df_both_F$diagnosis == diag & df_both_F$pred_adult == 1, "age_cat"])
	Ntotal <- length(df_both_F[df_both_F$age_cat == agebin & df_both_F$diagnosis == diag, "age_cat"])
	### For plot
	# Percentage adult
	PerAdult <- 100*(Nadult/Ntotal)
	# Nadult/Ntot
	Nadult_Ntotal <- paste0(Nadult, "/", Ntotal)
	# Put values in dataframe
	summary_2YBins_adultPer_F[i, "Nadult"] <- Nadult
	summary_2YBins_adultPer_F[i, "Ntotal"] <- Ntotal
	summary_2YBins_adultPer_F[i, "Percent_Pred_Adult"] <- PerAdult
	summary_2YBins_adultPer_F[i, "Nadult_Ntotal"] <- Nadult_Ntotal
	# SE
	summary_2YBins_adultPer_F[i, "SE"] <- 100*sqrt((Nadult/Ntotal)*(1-(Nadult/Ntotal))/Ntotal)
}
summary_2YBins_adultPer_F$Age_Category <- factor(summary_2YBins_adultPer_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))
 
### Three-year age bins: % Adult 
summary_3YBins_adultPer_F <- data.frame(matrix(0, nrow=10, ncol=7))
colnames(summary_3YBins_adultPer_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "Nadult_Ntotal", "Nadult", "Ntotal", "SE")
summary_3YBins_adultPer_F$Age_Category <- c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3YBins_adultPer_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_3YBins_adultPer_F)) { #CIs from conf.level
	agebin <- summary_3YBins_adultPer_F[i, "Age_Category"]
	diag <- summary_3YBins_adultPer_F[i, "Diagnosis"]
	# Total number and number adult
	Nadult <- length(df_both_F[df_both_F$age_cat_3 == agebin & df_both_F$diagnosis == diag & df_both_F$pred_adult == 1, "age_cat_3"])
	Ntotal <- length(df_both_F[df_both_F$age_cat_3 == agebin & df_both_F$diagnosis == diag, "age_cat_3"])
	### For plot
	# Percentage adult
	PerAdult <- 100*(Nadult/Ntotal)
	# Nadult/Ntot
	Nadult_Ntotal <- paste0(Nadult, "/", Ntotal)
	# Put values in dataframe
	summary_3YBins_adultPer_F[i, "Nadult"] <- Nadult
	summary_3YBins_adultPer_F[i, "Ntotal"] <- Ntotal
	summary_3YBins_adultPer_F[i, "Percent_Pred_Adult"] <- PerAdult
	summary_3YBins_adultPer_F[i, "Nadult_Ntotal"] <- Nadult_Ntotal
	# SE
	summary_3YBins_adultPer_F[i, "SE"] <- 100*sqrt((Nadult/Ntotal)*(1-(Nadult/Ntotal))/Ntotal)
}
summary_3YBins_adultPer_F$Age_Category <- factor(summary_3YBins_adultPer_F$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))

### Two-Year Age Bins: Mean Difference between Real and Predicted Ages 
summary_2YBins_diffRealPred_F <- data.frame(matrix(0, nrow=14, ncol=5))
colnames(summary_2YBins_diffRealPred_F) <- c("Age_Category", "Diagnosis", "Mean_Diff_Real_minus_Pred", "N", "SE")
summary_2YBins_diffRealPred_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_2YBins_diffRealPred_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_2YBins_diffRealPred_F)) { #CIs from conf.level
	agebin <- summary_2YBins_diffRealPred_F[i, "Age_Category"]
	diag <- summary_2YBins_diffRealPred_F[i, "Diagnosis"]
	# Total number 
	N <- length(df_both_F[df_both_F$age_cat == agebin & df_both_F$diagnosis == diag, "age_cat"])
	### For plot
	# Mean Difference between Real and Predicted Ages
	realpreddiff <- df_both_F[df_both_F$age_cat == agebin & df_both_F$diagnosis == diag, "real_age"] - df_both_F[df_both_F$age_cat == agebin & df_both_F$diagnosis == diag, "pred_age"]
	RealPredDiff <- mean(realpreddiff)
	# Put values in dataframe
	summary_2YBins_diffRealPred_F[i, "N"] <- N
	summary_2YBins_diffRealPred_F[i, "Mean_Diff_Real_minus_Pred"] <- RealPredDiff
	# SE
	summary_2YBins_diffRealPred_F[i, "SE"] <- sqrt(sd(realpreddiff)/N)
}
summary_2YBins_diffRealPred_F$Age_Category <- factor(summary_2YBins_diffRealPred_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

### Three-Year Age Bins: Mean Difference between Real and Predicted Ages #~~~~~~~~~~~~~~~~~~CORRECT SE~~~~~~~~~~~~~~~~~~
summary_3YBins_diffRealPred_F <- data.frame(matrix(0, nrow=10, ncol=5))
colnames(summary_3YBins_diffRealPred_F) <- c("Age_Category", "Diagnosis", "Mean_Diff_Real_minus_Pred", "N", "SE")
summary_3YBins_diffRealPred_F$Age_Category <-  c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3YBins_diffRealPred_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_3YBins_diffRealPred_F)) { #CIs from conf.level
	agebin <- summary_3YBins_diffRealPred_F[i, "Age_Category"]
	diag <- summary_3YBins_diffRealPred_F[i, "Diagnosis"]
	# Total number 
	N <- length(df_both_F[df_both_F$age_cat_3 == agebin & df_both_F$diagnosis == diag, "age_cat_3"])
	### For plot
	# Mean Difference between Real and Predicted Ages
	realpreddiff <- df_both_F[df_both_F$age_cat_3 == agebin & df_both_F$diagnosis == diag, "real_age"] - df_both_F[df_both_F$age_cat_3 == agebin & df_both_F$diagnosis == diag, "pred_age"]
	RealPredDiff <- mean(realpreddiff)
	# Put values in dataframe
	summary_3YBins_diffRealPred_F[i, "N"] <- N
	summary_3YBins_diffRealPred_F[i, "Mean_Diff_Real_minus_Pred"] <- RealPredDiff
	# SE
	summary_3YBins_diffRealPred_F[i, "SE"] <- sqrt(sd(realpreddiff)/N)
}
summary_3YBins_diffRealPred_F$Age_Category <- factor(summary_3YBins_diffRealPred_F$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))

######################
# FiveFoldWhole Plot Info
td_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age"])$p
if (td_p_F < .001) { td_p_F <- .001 } else if (td_p_F < .01) { td_p_F <- .01 } else if (td_p_F < .05) { td_p_F <- .05 } else { td_p_F <- round(td_p_F, digits=3) }
td_N_F <- nrow(df_both_F[df_both_F$diagnosis == "TD",])
ps_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age"])$p
if (ps_p_F < .001) { ps_p_F <- .001 } else if (ps_p_F < .01) { ps_p_F <- .01 } else if (ps_p_F < .05) { ps_p_F <- .05 } else { ps_p_F <- round(ps_p_F, digits=3) }
ps_N_F <- nrow(df_both_F[df_both_F$diagnosis == "PS",])

substring_F <- paste0("TD: r = ", td_corr_F, ", p < ", td_p_F,", N = ", td_N_F, "\nPS: r = ", ps_corr_F, ", p < ", ps_p_F,", N = ", ps_N_F)


# turn fold predicted ages into years
df_ps_F$pred_age_fold1 <- df_ps_F$fold_1_ageAtScan1/12
df_ps_F$pred_age_fold2 <- df_ps_F$fold_2_ageAtScan1/12
df_ps_F$pred_age_fold3 <- df_ps_F$fold_3_ageAtScan1/12
df_ps_F$pred_age_fold4 <- df_ps_F$fold_4_ageAtScan1/12
df_ps_F$pred_age_fold5 <- df_ps_F$fold_5_ageAtScan1/12

# Randomly choose one of the five folds for each PS 
folds5_PS <- createFolds(df_ps_F$ageAtScan1, k = 5, list = TRUE)
df_ps_F$pred_age_randFold <- NA
for (i in 1:nrow(df_ps_F)) {
	if (i %in% folds5_PS[[1]]) { df_ps_F[i, "pred_age_randFold"] <- df_ps_F[i, "pred_age_fold1"] 
	} else if (i %in% folds5_PS[[2]]) { df_ps_F[i, "pred_age_randFold"] <- df_ps_F[i, "pred_age_fold2"] 
	} else if (i %in% folds5_PS[[3]]) { df_ps_F[i, "pred_age_randFold"] <- df_ps_F[i, "pred_age_fold3"] 
	} else if (i %in% folds5_PS[[4]]) { df_ps_F[i, "pred_age_randFold"] <- df_ps_F[i, "pred_age_fold4"] 
	} else if (i %in% folds5_PS[[5]]) { df_ps_F[i, "pred_age_randFold"] <- df_ps_F[i, "pred_age_fold5"] 
	}
}
df_ps_F2 <- df_ps_F[,c("bblid", "real_age", "diagnosis", "pred_age_randFold")]
colnames(df_ps_F2)[colnames(df_ps_F2) == 'pred_age_randFold'] <- 'pred_age'
df_td_F2 <- df_td_F[,c("bblid", "real_age", "diagnosis", "pred_age")]
df_both_F2 <- rbind(df_td_F2, df_ps_F2)

# FiveFold Plot Info
df_both_F2$diagnosis <- factor(df_both_F2$diagnosis)
td_corr_F2 <- round(corr.test(df_both_F2[df_both_F2$diagnosis == "TD", "real_age"], df_both_F2[df_both_F2$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_F2 <- corr.test(df_both_F2[df_both_F2$diagnosis == "TD", "real_age"], df_both_F2[df_both_F2$diagnosis == "TD", "pred_age"])$p
if (td_p_F2 < .001) { td_p_F2 <- .001 } else if (td_p_F2 < .01) { td_p_F2 <- .01 } else if (td_p_F2 < .05) { td_p_F2 <- .05 } else { td_p_F2 <- round(td_p_F2, digits=3) }
td_N_F2 <- nrow(df_both_F2[df_both_F2$diagnosis == "TD",])
ps_corr_F2 <- round(corr.test(df_both_F2[df_both_F2$diagnosis == "PS", "real_age"], df_both_F2[df_both_F2$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_F2 <- corr.test(df_both_F2[df_both_F2$diagnosis == "PS", "real_age"], df_both_F2[df_both_F2$diagnosis == "PS", "pred_age"])$p
if (ps_p_F2 < .001) { ps_p_F2 <- .001 } else if (ps_p_F2 < .01) { ps_p_F2 <- .01 } else if (ps_p_F2 < .05) { ps_p_F2 <- .05 } else { ps_p_F2 <- round(ps_p_F2, digits=3) }
ps_N_F2 <- nrow(df_both_F2[df_both_F2$diagnosis == "PS",])

substring_F2 <- paste0("TD: r = ", td_corr_F2, ", p < ", td_p_F2,", N = ", td_N_F2, "\nPS: r = ", ps_corr_F2, ", p < ", ps_p_F2,", N = ", ps_N_F2)

# Plots
real_pred_fiveFoldWhole_F <- ggplot(data=df_both_F, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Age (TDs from Five; PSs from Whole)", subtitle = substring_F) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

real_pred_fiveFold_F <- ggplot(data=df_both_F2, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Age (TDs and PSs from Five)", subtitle = substring_F2) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 






minrow_2YBins_adultPer_F <- round(min(summary_2YBins_adultPer_F$Percent_Pred_Adult - 2*summary_2YBins_adultPer_F$SE, na.rm=TRUE)-10, digits=-1)
maxrow_2YBins_adultPer_F <- round(max(summary_2YBins_adultPer_F$Percent_Pred_Adult + 2*summary_2YBins_adultPer_F$SE, na.rm=TRUE)+10, digits=-1)
if (maxrow_2YBins_adultPer_F < 100) { maxrow_2YBins_adultPer_F <- 100 }
bar_2YBins_adultPer_F <- ggplot(data=summary_2YBins_adultPer_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Percent_Pred_Adult-2*SE, ymax=Percent_Pred_Adult+2*SE), position=position_dodge()) +
	geom_text(aes(label=Nadult_Ntotal), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Adult % (TDs from Five; PSs from Whole)") + xlab("Age Category") + ylab("% Predicted Adult (95% CI)") +
	scale_y_continuous(limits = c(minrow_2YBins_adultPer_F, maxrow_2YBins_adultPer_F), breaks = seq(minrow_2YBins_adultPer_F, maxrow_2YBins_adultPer_F, by=10)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_3YBins_adultPer_F <- round(min(summary_3YBins_adultPer_F$Percent_Pred_Adult - 2*summary_3YBins_adultPer_F$SE, na.rm=TRUE)-10, digits=-1)
maxrow_3YBins_adultPer_F <- round(max(summary_3YBins_adultPer_F$Percent_Pred_Adult + 2*summary_3YBins_adultPer_F$SE, na.rm=TRUE)+10, digits=-1)
if (maxrow_3YBins_adultPer_F < 100) { maxrow_3YBins_adultPer_F <- 100 }
bar_3YBins_adultPer_F <- ggplot(data=summary_3YBins_adultPer_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Percent_Pred_Adult-2*SE, ymax=Percent_Pred_Adult+2*SE), position=position_dodge()) +
	geom_text(aes(label=Nadult_Ntotal), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Adult % (TDs from Five; PSs from Whole)") + xlab("Age Category") + ylab("% Predicted Adult (95% CI)") +
	scale_y_continuous(limits = c(minrow_3YBins_adultPer_F, maxrow_3YBins_adultPer_F), breaks = seq(minrow_3YBins_adultPer_F, maxrow_3YBins_adultPer_F, by=10)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_2YBins_diffRealPred_F <- round(min(summary_2YBins_diffRealPred_F$Mean_Diff_Real_minus_Pred - 2*summary_2YBins_diffRealPred_F$SE, na.rm=TRUE)-1)
maxrow_2YBins_diffRealPred_F <- round(max(summary_2YBins_diffRealPred_F$Mean_Diff_Real_minus_Pred + 2*summary_2YBins_diffRealPred_F$SE, na.rm=TRUE)+1)
bar_2YBins_diffRealPred_F <- ggplot(data=summary_2YBins_diffRealPred_F, aes(x=Age_Category, y=Mean_Diff_Real_minus_Pred, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Mean_Diff_Real_minus_Pred+2*SE, ymax=Mean_Diff_Real_minus_Pred-2*SE), position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="black", position = position_dodge(0.9), size=4) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Real-Predicted Age by Age Bin and Diagnosis") + xlab("Age Category") + ylab("Real - Pred Age (95% CI)") +
	scale_y_continuous(limits = c(minrow_2YBins_diffRealPred_F, maxrow_2YBins_diffRealPred_F), breaks = seq(minrow_2YBins_diffRealPred_F, maxrow_2YBins_diffRealPred_F, by=1)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_3YBins_diffRealPred_F <- round(min(summary_3YBins_diffRealPred_F$Mean_Diff_Real_minus_Pred - 2*summary_3YBins_diffRealPred_F$SE, na.rm=TRUE)-1)
maxrow_3YBins_diffRealPred_F <- round(max(summary_3YBins_diffRealPred_F$Mean_Diff_Real_minus_Pred + 2*summary_3YBins_diffRealPred_F$SE, na.rm=TRUE)+1)
bar_3YBins_diffRealPred_F <- ggplot(data=summary_3YBins_diffRealPred_F, aes(x=Age_Category, y=Mean_Diff_Real_minus_Pred, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Mean_Diff_Real_minus_Pred+2*SE, ymax=Mean_Diff_Real_minus_Pred-2*SE), position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="black", position = position_dodge(0.9), size=4) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Real-Predicted Age by Age Bin and Diagnosis") + xlab("Age Category") + ylab("Real - Pred Age (95% CI)") +
	scale_y_continuous(limits = c(minrow_3YBins_diffRealPred_F, maxrow_3YBins_diffRealPred_F), breaks = seq(minrow_3YBins_diffRealPred_F, maxrow_3YBins_diffRealPred_F, by=1)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_RvsW_F <- ggplot(data=df_ps_F, aes(real_age, pred_age)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from All of the TDs ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs1_F <- ggplot(data=df_ps_F, aes(real_age, pred_age_fold1)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 1 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs2_F <- ggplot(data=df_ps_F, aes(real_age, pred_age_fold2)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 2 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs3_F <- ggplot(data=df_ps_F, aes(real_age, pred_age_fold3)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 3 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs4_F <- ggplot(data=df_ps_F, aes(real_age, pred_age_fold4)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 4 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs5_F <- ggplot(data=df_ps_F, aes(real_age, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 5 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_1vs5_F <- ggplot(data=df_ps_F, aes(pred_age_fold1, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 5 ~ Pred Age from Fold 1") + 
	xlab("Predicted Age: Fold 1") + ylab("Predicted Age: Fold 5") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs1_F <- ggplot(data=df_ps_F, aes(pred_age, pred_age_fold1)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 1 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 1") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs2_F <- ggplot(data=df_ps_F, aes(pred_age, pred_age_fold2)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 2 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 2") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs3_F <- ggplot(data=df_ps_F, aes(pred_age, pred_age_fold3)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 3 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 3") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs4_F <- ggplot(data=df_ps_F, aes(pred_age, pred_age_fold4)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 4 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 4") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs5_F <- ggplot(data=df_ps_F, aes(pred_age, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 5 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 5") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

betas_1vs5_F <- ggplot(data=coefs_F[-1,], aes(First_Fold, Fifth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: First 4/5ths vs. Fifth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using Fifth 4/5ths of TDs") + ylab("Betas using First 4/5ths of TDs")

betas_Wvs1_F <- ggplot(data=coefs_F[-1,], aes(Whole_Sample, First_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. First 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using First 4/5ths of TDs")

betas_Wvs2_F <- ggplot(data=coefs_F[-1,], aes(Whole_Sample, Second_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Second 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Second 4/5ths of TDs")

betas_Wvs3_F <- ggplot(data=coefs_F[-1,], aes(Whole_Sample, Third_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Third 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Third 4/5ths of TDs")

betas_Wvs4_F <- ggplot(data=coefs_F[-1,], aes(Whole_Sample, Fourth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Fourth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Fourth 4/5ths of TDs")

betas_Wvs5_F <- ggplot(data=coefs_F[-1,], aes(Whole_Sample, Fifth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Fifth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Fifth 4/5ths of TDs")


############################ Males ############################

df_M <- df[df$sex == 1, c("bblid", "psTerminal", xvars, yvar)]
df_M$bblid <- as.factor(df_M$bblid)

for (i in 1:ncol(df_M)) {
	na_index <- is.na(df_M[,i])
	df_M <- df_M[!na_index,]
}

td_M <- df_M[df_M$psTerminal == "TD",]
rownames(td_M) <- 1:nrow(td_M)

folds5 <- createFolds(td_M$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5[[i]]
	trainfolds <- subset(folds5,!(grepl(i, names(folds5))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_td_train_M_", i), td_M[train, c("bblid", xvars)])
	assign(paste0("x_td_test_M_", i), td_M[test, c("bblid", xvars)])
	assign(paste0("y_td_train_M_", i), td_M[train, c("bblid", yvar)])
	assign(paste0("y_td_test_M_", i), td_M[test, c("bblid", yvar)])
}

#####Elastic Net: out of sample predictions for TD's
for (i in 1:5) {
	x_train_df <- get(paste0("x_td_train_M_", i))
	x_train_input <- x_train_df[,xvars]
	y_train_df <- get(paste0("y_td_train_M_", i))
	y_train_input <- y_train_df$ageAtScan1
	ageAtScan1 <- y_train_input
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = FALSE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_td_M_", i), elastic_net_model)
	x_test_df <- get(paste0("x_td_test_M_", i))
	y_test_df <- get(paste0("y_td_test_M_", i))
	y_test_predicted <- predict(elastic_net_model, x_test_df)
	if (i == 1) { 
		predicted_values <- merge(cbind(x_test_df, y_test_predicted), y_test_df)
		coefs <- coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)# Added March 11, 2019
	} else { 
		predicted_values <- rbind(predicted_values, merge(cbind(x_test_df, y_test_predicted), y_test_df))
		coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda))# Added March 11, 2019
	}
}
names(predicted_values)[names(predicted_values) == 'y_test_predicted'] <- 'predicted_ageAtScan1'

#####Elastic Net: predictions for PS's 
# train on full TD set
x_td_train_input <- td_M[,xvars]
ageAtScan1 <- td_M[,yvar]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = FALSE)
elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)) # Added March 11, 2019
coefs <- as.matrix(coefs) # Added March 11, 2019
coefs_M <- data.frame(coefs) # Added March 11, 2019
colnames(coefs_M) <- c("First_Fold", "Second_Fold", "Third_Fold", "Fourth_Fold", "Fifth_Fold", "Whole_Sample") # Added March 11, 2019

# predict on PS's
x_ps_M <- df_M[df_M$psTerminal == "PS", c("bblid", xvars)]
y_ps_M <- df_M[df_M$psTerminal == "PS", c("bblid", yvar)]
x_ps_input <- x_ps_M[,xvars]
y_ps_input <- y_ps_M[,yvar]
ageAtScan1 <- y_ps_input
y_ps_predicted <- predict(elastic_net_model, x_ps_input)
x_ps_M$predicted_ageAtScan1 <- y_ps_predicted ### March 12, 2019: THIS HAS TO BE HERE

# create final dfs
df_td_M <- predicted_values
df_ps_M <- merge(x_ps_M, y_ps_M) #####

# get predictions for PSs based on five models for TDs # Added March 11, 2019
for (i in 1:5) {
	elastic_net_model <- get(paste0("mod_td_M_", i))
	df_ps_M[,paste0("fold_", i, "_ageAtScan1")] <- predict(elastic_net_model, df_ps_M[,xvars]) ### March 12, 2019: THIS HAS TO BE df_ps_M, NOT x_ps_M
}

# Create years age variables
df_ps_M$real_age <- df_ps_M$ageAtScan1/12
df_ps_M$pred_age <- df_ps_M$predicted_ageAtScan1/12
df_td_M$real_age <- df_td_M$ageAtScan1/12
df_td_M$pred_age <- df_td_M$predicted_ageAtScan1/12

# Create the final dataframe
df_ps_M$diagnosis <- "PS"
df_td_M$diagnosis <- "TD"
df_both_M <- rbind(df_ps_M[,!(names(df_ps_M) %in% c("fold_1_ageAtScan1", "fold_2_ageAtScan1", "fold_3_ageAtScan1", "fold_4_ageAtScan1", "fold_5_ageAtScan1"))], df_td_M)
df_both_M$diagnosis <- as.factor(df_both_M$diagnosis)

### Perform LMEM
#df_long_M <- melt(df_both_M[, !(names(df_both_M) %in% c("predicted_ageAtScan1", "ageAtScan1", "pred_age", "age_cat", "pred_adult", "age_cat_3"))], #id.vars=c("bblid","diagnosis","real_age"))
#df_long_M <- transform(df_long_M, value=scale(value))
#lmem_mod <- lmer(real_age ~ (variable + value + diagnosis)^3 + (1 | bblid), data=df_long_M)
#lme_mod <- lme(real_age ~ (variable + value + diagnosis)^3, random=~1|bblid, data=df_long_M)
#lmem_mod.all <- allFit(lmem_mod)

### Look for interactions... Age ~ xvar*diagnosis
# all
siginter_all_M <- as.data.frame(matrix(0, ncol=4, nrow=length(xvars)))
colnames(siginter_all_M) <- c("ROI", "P_Value", "Sig", "InterEstPos")
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	f <- paste0("real_age ~ ", xvar, " + diagnosis + diagnosis:", xvar)
	mod <- lm(f, data=df_both_M)
	siginter_all_M[i,1] <- xvar
	siginter_all_M[i,2] <- round(anova(mod)[["Pr(>F)"]][[3]], digits=3)
	if (anova(mod)[["Pr(>F)"]][[3]] < .05) { siginter_all_M[i,3] <- "*" } else { siginter_all_M[i,3] <- "" }
	if (mod$coefficients[[4]] < 0) { siginter_all_M[i, "InterEstPos"] <- "Neg" } else { siginter_all_M[i, "InterEstPos"] <- "Pos" }
}
siginter_all_M$ROI <- as.factor(siginter_all_M$ROI)
numsig_all_M <- paste0(nrow(siginter_all_M[siginter_all_M$Sig == "*",]), " out of ", nrow(siginter_all_M), " interactions with p < .05\n", nrow(siginter_all_M[siginter_all_M$Sig == "*" & siginter_all_M$InterEstPos == "Pos",]), " of these have a positive coefficient estimate")

interaction_all_M <- ggplot(data=siginter_all_M, aes(P_Value, color=Sig, fill=Sig)) +
	geom_histogram(bins=20, alpha=0.5) + xlab("P-Value") +
	ggtitle("P-Values for Diagnosis:ROI (all subjects)") +
	labs(subtitle=numsig_all_M) + theme_minimal() +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 


# over 10
siginter_over10_M <- as.data.frame(matrix(0, ncol=4, nrow=length(xvars)))
colnames(siginter_over10_M) <- c("ROI", "P_Value", "Sig", "InterEstPos")
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	f <- paste0("real_age ~ ", xvar, " + diagnosis + diagnosis:", xvar)
	mod <- lm(f, data=df_both_M[df_both_M$real_age >= 10, ])
	siginter_over10_M[i,1] <- xvar
	siginter_over10_M[i,2] <- round(anova(mod)[["Pr(>F)"]][[3]], digits=3)
	if (anova(mod)[["Pr(>F)"]][[3]] < .05) { siginter_over10_M[i,3] <- "*" } else { siginter_over10_M[i,3] <- "" }
	if (mod$coefficients[[4]] < 0) { siginter_over10_M[i, "InterEstPos"] <- "Neg" } else { siginter_over10_M[i, "InterEstPos"] <- "Pos" }
}
siginter_over10_M$ROI <- as.factor(siginter_over10_M$ROI)
numsig_over10_M <- paste0(nrow(siginter_over10_M[siginter_over10_M$Sig == "*",]), " out of ", nrow(siginter_over10_M), " interactions with p < .05\n", nrow(siginter_over10_M[siginter_over10_M$Sig == "*" & siginter_over10_M$InterEstPos == "Pos",]), " of these have a positive coefficient estimate")

interaction_over10_M <- ggplot(data=siginter_over10_M, aes(P_Value, color=Sig, fill=Sig)) +
	geom_histogram(bins=20, alpha=0.5) + xlab("P-Value") +
	ggtitle("P-Values for Diagnosis:ROI (Age > 10)") +
	labs(subtitle=numsig_over10_M) + theme_minimal() +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

# Plot GAMMs
for (i in 1:length(xvars)) {
	xvar <- xvars[[i]]
	subtit <- paste0("ROI: ", xvar)
	gam_plot <- ggplot(data=df_both_M, aes_string("real_age", xvar, color="diagnosis")) +
		geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
		xlim(0,25) + labs(title = "Age~ROI", subtitle = subtit) + 
		xlab("Age") + ylab("ROI Value") + geom_smooth(se = TRUE, method= "gam", formula = y ~ log(x)) +
		theme(plot.title = element_text(family="Times", face="bold", size=18)) 
	assign(paste0("gamplot_", i, "_M"), gam_plot)
}

# Create age bins
df_both_M$age_cat <- 0
df_both_M$pred_adult <- 0

rownames(df_both_M) <- 1:nrow(df_both_M) # change row indices

for (row in 1:nrow(df_both_M)) {
	if (df_both_M[row,"real_age"] < 10) { df_both_M[row,"age_cat"] = "8_9" }
	else if (df_both_M[row,"real_age"] >= 10 & df_both_M[row,"real_age"] < 12) { df_both_M[row,"age_cat"] = "10_11" }
	else if (df_both_M[row,"real_age"] >= 12 & df_both_M[row,"real_age"] < 14) { df_both_M[row,"age_cat"] = "12_13" }
	else if (df_both_M[row,"real_age"] >= 14 & df_both_M[row,"real_age"] < 16) { df_both_M[row,"age_cat"] = "14_15" }
	else if (df_both_M[row,"real_age"] >= 16 & df_both_M[row,"real_age"] < 18) { df_both_M[row,"age_cat"] = "16_17" }
	else if (df_both_M[row,"real_age"] >= 18 & df_both_M[row,"real_age"] < 20) { df_both_M[row,"age_cat"] = "18_19" }
	else if (df_both_M[row,"real_age"] >= 20 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat"] = "20_23" }
	if (df_both_M[row,"pred_age"] >= 18) { df_both_M[row,"pred_adult"] = 1 }
}

df_both_M$age_cat <- as.factor(df_both_M$age_cat)

df_both_M$age_cat_3 <- 0

for (row in 1:nrow(df_both_M)) {
	if (df_both_M[row,"real_age"] < 12) { df_both_M[row,"age_cat_3"] = "8_11" }
	else if (df_both_M[row,"real_age"] >= 12 & df_both_M[row,"real_age"] < 15) { df_both_M[row,"age_cat_3"] = "12_14" }
	else if (df_both_M[row,"real_age"] >= 15 & df_both_M[row,"real_age"] < 18) { df_both_M[row,"age_cat_3"] = "15_17" }
	else if (df_both_M[row,"real_age"] >= 18 & df_both_M[row,"real_age"] < 21) { df_both_M[row,"age_cat_3"] = "18_20" }
	else if (df_both_M[row,"real_age"] >= 21 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat_3"] = "21_23" }
}

df_both_M$age_cat_3 <- as.factor(df_both_M$age_cat_3)

############# Summary Dataframes
### Two-Year Age Bins: % Adult
summary_2YBins_adultPer_M <- data.frame(matrix(0, nrow=14, ncol=7))
colnames(summary_2YBins_adultPer_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "Nadult_Ntotal", "Nadult", "Ntotal", "SE")
summary_2YBins_adultPer_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_2YBins_adultPer_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_2YBins_adultPer_M)) { #CIs from conf.level
	agebin <- summary_2YBins_adultPer_M[i, "Age_Category"]
	diag <- summary_2YBins_adultPer_M[i, "Diagnosis"]
	# Total number and number adult
	Nadult <- length(df_both_M[df_both_M$age_cat == agebin & df_both_M$diagnosis == diag & df_both_M$pred_adult == 1, "age_cat"])
	Ntotal <- length(df_both_M[df_both_M$age_cat == agebin & df_both_M$diagnosis == diag, "age_cat"])
	### For plot
	# Percentage adult
	PerAdult <- 100*(Nadult/Ntotal)
	# Nadult/Ntot
	Nadult_Ntotal <- paste0(Nadult, "/", Ntotal)
	# Put values in dataframe
	summary_2YBins_adultPer_M[i, "Nadult"] <- Nadult
	summary_2YBins_adultPer_M[i, "Ntotal"] <- Ntotal
	summary_2YBins_adultPer_M[i, "Percent_Pred_Adult"] <- PerAdult
	summary_2YBins_adultPer_M[i, "Nadult_Ntotal"] <- Nadult_Ntotal
	# SE
	summary_2YBins_adultPer_M[i, "SE"] <- 100*sqrt((Nadult/Ntotal)*(1-(Nadult/Ntotal))/Ntotal)
}
summary_2YBins_adultPer_M$Age_Category <- factor(summary_2YBins_adultPer_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))
 
### Three-year age bins: % Adult 
summary_3YBins_adultPer_M <- data.frame(matrix(0, nrow=10, ncol=7))
colnames(summary_3YBins_adultPer_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "Nadult_Ntotal", "Nadult", "Ntotal", "SE")
summary_3YBins_adultPer_M$Age_Category <- c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3YBins_adultPer_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_3YBins_adultPer_M)) { #CIs from conf.level
	agebin <- summary_3YBins_adultPer_M[i, "Age_Category"]
	diag <- summary_3YBins_adultPer_M[i, "Diagnosis"]
	# Total number and number adult
	Nadult <- length(df_both_M[df_both_M$age_cat_3 == agebin & df_both_M$diagnosis == diag & df_both_M$pred_adult == 1, "age_cat_3"])
	Ntotal <- length(df_both_M[df_both_M$age_cat_3 == agebin & df_both_M$diagnosis == diag, "age_cat_3"])
	### For plot
	# Percentage adult
	PerAdult <- 100*(Nadult/Ntotal)
	# Nadult/Ntot
	Nadult_Ntotal <- paste0(Nadult, "/", Ntotal)
	# Put values in dataframe
	summary_3YBins_adultPer_M[i, "Nadult"] <- Nadult
	summary_3YBins_adultPer_M[i, "Ntotal"] <- Ntotal
	summary_3YBins_adultPer_M[i, "Percent_Pred_Adult"] <- PerAdult
	summary_3YBins_adultPer_M[i, "Nadult_Ntotal"] <- Nadult_Ntotal
	# SE
	summary_3YBins_adultPer_M[i, "SE"] <- 100*sqrt((Nadult/Ntotal)*(1-(Nadult/Ntotal))/Ntotal)
}
summary_3YBins_adultPer_M$Age_Category <- factor(summary_3YBins_adultPer_M$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))

### Two-Year Age Bins: Mean Difference between Real and Predicted Ages 
summary_2YBins_diffRealPred_M <- data.frame(matrix(0, nrow=14, ncol=5))
colnames(summary_2YBins_diffRealPred_M) <- c("Age_Category", "Diagnosis", "Mean_Diff_Real_minus_Pred", "N", "SE")
summary_2YBins_diffRealPred_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_2YBins_diffRealPred_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_2YBins_diffRealPred_M)) { #CIs from conf.level
	agebin <- summary_2YBins_diffRealPred_M[i, "Age_Category"]
	diag <- summary_2YBins_diffRealPred_M[i, "Diagnosis"]
	# Total number 
	N <- length(df_both_M[df_both_M$age_cat == agebin & df_both_M$diagnosis == diag, "age_cat"])
	### For plot
	# Mean Difference between Real and Predicted Ages
	realpreddiff <- df_both_M[df_both_M$age_cat == agebin & df_both_M$diagnosis == diag, "real_age"] - df_both_M[df_both_M$age_cat == agebin & df_both_M$diagnosis == diag, "pred_age"]
	RealPredDiff <- mean(realpreddiff)
	# Put values in dataframe
	summary_2YBins_diffRealPred_M[i, "N"] <- N
	summary_2YBins_diffRealPred_M[i, "Mean_Diff_Real_minus_Pred"] <- RealPredDiff
	# SE
	summary_2YBins_diffRealPred_M[i, "SE"] <- sqrt(sd(realpreddiff)/N)
}
summary_2YBins_diffRealPred_M$Age_Category <- factor(summary_2YBins_diffRealPred_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

### Three-Year Age Bins: Mean Difference between Real and Predicted Ages #~~~~~~~~~~~~~~~~~~CORRECT SE~~~~~~~~~~~~~~~~~~
summary_3YBins_diffRealPred_M <- data.frame(matrix(0, nrow=10, ncol=5))
colnames(summary_3YBins_diffRealPred_M) <- c("Age_Category", "Diagnosis", "Mean_Diff_Real_minus_Pred", "N", "SE")
summary_3YBins_diffRealPred_M$Age_Category <-  c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3YBins_diffRealPred_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

for (i in 1:nrow(summary_3YBins_diffRealPred_M)) { #CIs from conf.level
	agebin <- summary_3YBins_diffRealPred_M[i, "Age_Category"]
	diag <- summary_3YBins_diffRealPred_M[i, "Diagnosis"]
	# Total number 
	N <- length(df_both_M[df_both_M$age_cat_3 == agebin & df_both_M$diagnosis == diag, "age_cat_3"])
	### For plot
	# Mean Difference between Real and Predicted Ages
	realpreddiff <- df_both_M[df_both_M$age_cat_3 == agebin & df_both_M$diagnosis == diag, "real_age"] - df_both_M[df_both_M$age_cat_3 == agebin & df_both_M$diagnosis == diag, "pred_age"]
	RealPredDiff <- mean(realpreddiff)
	# Put values in dataframe
	summary_3YBins_diffRealPred_M[i, "N"] <- N
	summary_3YBins_diffRealPred_M[i, "Mean_Diff_Real_minus_Pred"] <- RealPredDiff
	# SE
	summary_3YBins_diffRealPred_M[i, "SE"] <- sqrt(sd(realpreddiff)/N)
}
summary_3YBins_diffRealPred_M$Age_Category <- factor(summary_3YBins_diffRealPred_M$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))

######################
# FiveFoldWhole Plot Info
td_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age"])$p
if (td_p_M < .001) { td_p_M <- .001 } else if (td_p_M < .01) { td_p_M <- .01 } else if (td_p_M < .05) { td_p_M <- .05 } else { td_p_M <- round(td_p_M, digits=3) }
td_N_M <- nrow(df_both_M[df_both_M$diagnosis == "TD",])
ps_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age"])$p
if (ps_p_M < .001) { ps_p_M <- .001 } else if (ps_p_M < .01) { ps_p_M <- .01 } else if (ps_p_M < .05) { ps_p_M <- .05 } else { ps_p_M <- round(ps_p_M, digits=3) }
ps_N_M <- nrow(df_both_M[df_both_M$diagnosis == "PS",])

substring_M <- paste0("TD: r = ", td_corr_M, ", p < ", td_p_M,", N = ", td_N_M, "\nPS: r = ", ps_corr_M, ", p < ", ps_p_M,", N = ", ps_N_M)


# turn fold predicted ages into years
df_ps_M$pred_age_fold1 <- df_ps_M$fold_1_ageAtScan1/12
df_ps_M$pred_age_fold2 <- df_ps_M$fold_2_ageAtScan1/12
df_ps_M$pred_age_fold3 <- df_ps_M$fold_3_ageAtScan1/12
df_ps_M$pred_age_fold4 <- df_ps_M$fold_4_ageAtScan1/12
df_ps_M$pred_age_fold5 <- df_ps_M$fold_5_ageAtScan1/12

# Randomly choose one of the five folds for each PS 
folds5_PS <- createFolds(df_ps_M$ageAtScan1, k = 5, list = TRUE)
df_ps_M$pred_age_randFold <- NA
for (i in 1:nrow(df_ps_M)) {
	if (i %in% folds5_PS[[1]]) { df_ps_M[i, "pred_age_randFold"] <- df_ps_M[i, "pred_age_fold1"] 
	} else if (i %in% folds5_PS[[2]]) { df_ps_M[i, "pred_age_randFold"] <- df_ps_M[i, "pred_age_fold2"] 
	} else if (i %in% folds5_PS[[3]]) { df_ps_M[i, "pred_age_randFold"] <- df_ps_M[i, "pred_age_fold3"] 
	} else if (i %in% folds5_PS[[4]]) { df_ps_M[i, "pred_age_randFold"] <- df_ps_M[i, "pred_age_fold4"] 
	} else if (i %in% folds5_PS[[5]]) { df_ps_M[i, "pred_age_randFold"] <- df_ps_M[i, "pred_age_fold5"] 
	}
}
df_ps_M2 <- df_ps_M[,c("bblid", "real_age", "diagnosis", "pred_age_randFold")]
colnames(df_ps_M2)[colnames(df_ps_M2) == 'pred_age_randFold'] <- 'pred_age'
df_td_M2 <- df_td_M[,c("bblid", "real_age", "diagnosis", "pred_age")]
df_both_M2 <- rbind(df_td_M2, df_ps_M2)

# FiveFold Plot Info
df_both_M2$diagnosis <- factor(df_both_M2$diagnosis)
td_corr_M2 <- round(corr.test(df_both_M2[df_both_M2$diagnosis == "TD", "real_age"], df_both_M2[df_both_M2$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_M2 <- corr.test(df_both_M2[df_both_M2$diagnosis == "TD", "real_age"], df_both_M2[df_both_M2$diagnosis == "TD", "pred_age"])$p
if (td_p_M2 < .001) { td_p_M2 <- .001 } else if (td_p_M2 < .01) { td_p_M2 <- .01 } else if (td_p_M2 < .05) { td_p_M2 <- .05 } else { td_p_M2 <- round(td_p_M2, digits=3) }
td_N_M2 <- nrow(df_both_M2[df_both_M2$diagnosis == "TD",])
ps_corr_M2 <- round(corr.test(df_both_M2[df_both_M2$diagnosis == "PS", "real_age"], df_both_M2[df_both_M2$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_M2 <- corr.test(df_both_M2[df_both_M2$diagnosis == "PS", "real_age"], df_both_M2[df_both_M2$diagnosis == "PS", "pred_age"])$p
if (ps_p_M2 < .001) { ps_p_M2 <- .001 } else if (ps_p_M2 < .01) { ps_p_M2 <- .01 } else if (ps_p_M2 < .05) { ps_p_M2 <- .05 } else { ps_p_M2 <- round(ps_p_M2, digits=3) }
ps_N_M2 <- nrow(df_both_M2[df_both_M2$diagnosis == "PS",])

substring_M2 <- paste0("TD: r = ", td_corr_M2, ", p < ", td_p_M2,", N = ", td_N_M2, "\nPS: r = ", ps_corr_M2, ", p < ", ps_p_M2,", N = ", ps_N_M2)

# Plots
real_pred_fiveFoldWhole_M <- ggplot(data=df_both_M, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Age (TDs from Five; PSs from Whole)", subtitle = substring_M) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

real_pred_fiveFold_M <- ggplot(data=df_both_M2, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Age (TDs and PSs from Five)", subtitle = substring_M2) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 






minrow_2YBins_adultPer_M <- round(min(summary_2YBins_adultPer_M$Percent_Pred_Adult - 2*summary_2YBins_adultPer_M$SE, na.rm=TRUE)-10, digits=-1)
maxrow_2YBins_adultPer_M <- round(max(summary_2YBins_adultPer_M$Percent_Pred_Adult + 2*summary_2YBins_adultPer_M$SE, na.rm=TRUE)+10, digits=-1)
if (maxrow_2YBins_adultPer_M < 100) { maxrow_2YBins_adultPer_M <- 100 }
bar_2YBins_adultPer_M <- ggplot(data=summary_2YBins_adultPer_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Percent_Pred_Adult-2*SE, ymax=Percent_Pred_Adult+2*SE), position=position_dodge()) +
	geom_text(aes(label=Nadult_Ntotal), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Adult % (TDs from Five; PSs from Whole)") + xlab("Age Category") + ylab("% Predicted Adult (95% CI)") +
	scale_y_continuous(limits = c(minrow_2YBins_adultPer_M, maxrow_2YBins_adultPer_M), breaks = seq(minrow_2YBins_adultPer_M, maxrow_2YBins_adultPer_M, by=10)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_3YBins_adultPer_M <- round(min(summary_3YBins_adultPer_M$Percent_Pred_Adult - 2*summary_3YBins_adultPer_M$SE, na.rm=TRUE)-10, digits=-1)
maxrow_3YBins_adultPer_M <- round(max(summary_3YBins_adultPer_M$Percent_Pred_Adult + 2*summary_3YBins_adultPer_M$SE, na.rm=TRUE)+10, digits=-1)
if (maxrow_3YBins_adultPer_M < 100) { maxrow_3YBins_adultPer_M <- 100 }
bar_3YBins_adultPer_M <- ggplot(data=summary_3YBins_adultPer_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Percent_Pred_Adult-2*SE, ymax=Percent_Pred_Adult+2*SE), position=position_dodge()) +
	geom_text(aes(label=Nadult_Ntotal), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Adult % (TDs from Five; PSs from Whole)") + xlab("Age Category") + ylab("% Predicted Adult (95% CI)") +
	scale_y_continuous(limits = c(minrow_3YBins_adultPer_M, maxrow_3YBins_adultPer_M), breaks = seq(minrow_3YBins_adultPer_M, maxrow_3YBins_adultPer_M, by=10)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_2YBins_diffRealPred_M <- round(min(summary_2YBins_diffRealPred_M$Mean_Diff_Real_minus_Pred - 2*summary_2YBins_diffRealPred_M$SE, na.rm=TRUE)-1)
maxrow_2YBins_diffRealPred_M <- round(max(summary_2YBins_diffRealPred_M$Mean_Diff_Real_minus_Pred + 2*summary_2YBins_diffRealPred_M$SE, na.rm=TRUE)+1)
bar_2YBins_diffRealPred_M <- ggplot(data=summary_2YBins_diffRealPred_M, aes(x=Age_Category, y=Mean_Diff_Real_minus_Pred, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Mean_Diff_Real_minus_Pred+2*SE, ymax=Mean_Diff_Real_minus_Pred-2*SE), position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="black", position = position_dodge(0.9), size=4) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Real-Predicted Age by Age Bin and Diagnosis") + xlab("Age Category") + ylab("Real - Pred Age (95% CI)") +
	scale_y_continuous(limits = c(minrow_2YBins_diffRealPred_M, maxrow_2YBins_diffRealPred_M), breaks = seq(minrow_2YBins_diffRealPred_M, maxrow_2YBins_diffRealPred_M, by=1)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

minrow_3YBins_diffRealPred_M <- round(min(summary_3YBins_diffRealPred_M$Mean_Diff_Real_minus_Pred - 2*summary_3YBins_diffRealPred_M$SE, na.rm=TRUE)-1)
maxrow_3YBins_diffRealPred_M <- round(max(summary_3YBins_diffRealPred_M$Mean_Diff_Real_minus_Pred + 2*summary_3YBins_diffRealPred_M$SE, na.rm=TRUE)+1)
bar_3YBins_diffRealPred_M <- ggplot(data=summary_3YBins_diffRealPred_M, aes(x=Age_Category, y=Mean_Diff_Real_minus_Pred, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_errorbar(mapping=aes(ymin=Mean_Diff_Real_minus_Pred+2*SE, ymax=Mean_Diff_Real_minus_Pred-2*SE), position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="black", position = position_dodge(0.9), size=4) +
	scale_fill_brewer(palette="Paired") + 
	ggtitle("Real-Predicted Age by Age Bin and Diagnosis") + xlab("Age Category") + ylab("Real - Pred Age (95% CI)") +
	scale_y_continuous(limits = c(minrow_3YBins_diffRealPred_M, maxrow_3YBins_diffRealPred_M), breaks = seq(minrow_3YBins_diffRealPred_M, maxrow_3YBins_diffRealPred_M, by=1)) +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_RvsW_M <- ggplot(data=df_ps_M, aes(real_age, pred_age)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from All of the TDs ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs1_M <- ggplot(data=df_ps_M, aes(real_age, pred_age_fold1)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 1 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs2_M <- ggplot(data=df_ps_M, aes(real_age, pred_age_fold2)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 2 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs3_M <- ggplot(data=df_ps_M, aes(real_age, pred_age_fold3)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 3 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs4_M <- ggplot(data=df_ps_M, aes(real_age, pred_age_fold4)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 4 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Rvs5_M <- ggplot(data=df_ps_M, aes(real_age, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Predicted Age from Fold 5 ~ Real Age") + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_1vs5_M <- ggplot(data=df_ps_M, aes(pred_age_fold1, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 5 ~ Pred Age from Fold 1") + 
	xlab("Predicted Age: Fold 1") + ylab("Predicted Age: Fold 5") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs1_M <- ggplot(data=df_ps_M, aes(pred_age, pred_age_fold1)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 1 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 1") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs2_M <- ggplot(data=df_ps_M, aes(pred_age, pred_age_fold2)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 2 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 2") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs3_M <- ggplot(data=df_ps_M, aes(pred_age, pred_age_fold3)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 3 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 3") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs4_M <- ggplot(data=df_ps_M, aes(pred_age, pred_age_fold4)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 4 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 4") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

predage_Wvs5_M <- ggplot(data=df_ps_M, aes(pred_age, pred_age_fold5)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "PS: Pred Age from Fold 5 ~ Pred Age from All TDs") + 
	xlab("Predicted Age: All TDs") + ylab("Predicted Age: Fold 5") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

betas_1vs5_M <- ggplot(data=coefs_M[-1,], aes(First_Fold, Fifth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: First 4/5ths vs. Fifth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using Fifth 4/5ths of TDs") + ylab("Betas using First 4/5ths of TDs")

betas_Wvs1_M <- ggplot(data=coefs_M[-1,], aes(Whole_Sample, First_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. First 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using First 4/5ths of TDs")

betas_Wvs2_M <- ggplot(data=coefs_M[-1,], aes(Whole_Sample, Second_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Second 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Second 4/5ths of TDs")

betas_Wvs3_M <- ggplot(data=coefs_M[-1,], aes(Whole_Sample, Third_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Third 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Third 4/5ths of TDs")

betas_Wvs4_M <- ggplot(data=coefs_M[-1,], aes(Whole_Sample, Fourth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Fourth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Fourth 4/5ths of TDs")

betas_Wvs5_M <- ggplot(data=coefs_M[-1,], aes(Whole_Sample, Fifth_Fold)) +
	geom_point(shape = 16, size = 4, show.legend = TRUE, color="slateblue") + theme_minimal() +
	labs(title = "Betas: All of the TDs vs. Fifth 4/5ths of the TDs") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
	xlab("Betas using All TDs") + ylab("Betas using Fifth 4/5ths of TDs")
##### Save plots
#Elastic Net Plots
pdf(file=paste0("/home/butellyn/age_prediction/plots/", type, "_fiveFoldWholeTDvsFiveFoldTD_ElasticNet.pdf"), width=12, height=6)
plot(0:10, type = "n", xaxt="n", yaxt="n", bty="n", xlab = "", ylab = "")
text(5, 8, paste0("Predicted Age for Females Using ", fullname, " ROIs in an Elastic Net Model"))
grid.arrange(real_pred_fiveFoldWhole_F, real_pred_fiveFold_F, ncol=2)
grid.arrange(bar_2YBins_adultPer_F, bar_3YBins_adultPer_F, ncol=2)
grid.arrange(bar_2YBins_diffRealPred_F, bar_3YBins_diffRealPred_F, ncol=2)
grid.arrange(interaction_all_F, interaction_over10_F, ncol=2)
grid.arrange(predage_RvsW_F, predage_Rvs1_F, ncol=2)
grid.arrange(predage_Rvs2_F, predage_Rvs3_F, ncol=2)
grid.arrange(predage_Rvs4_F, predage_Rvs5_F, ncol=2)
grid.arrange(predage_1vs5_F, predage_Wvs1_F, ncol=2)
grid.arrange(predage_Wvs2_F, predage_Wvs3_F, ncol=2)
grid.arrange(predage_Wvs4_F, predage_Wvs5_F, ncol=2)
grid.arrange(betas_1vs5_F, betas_Wvs1_F, ncol=2)
grid.arrange(betas_Wvs2_F, betas_Wvs3_F, ncol=2)
grid.arrange(betas_Wvs4_F, betas_Wvs5_F, ncol=2)
plot(0:10, type = "n", xaxt="n", yaxt="n", bty="n", xlab = "", ylab = "")
text(5, 8, paste0("Predicted Age for Males Using ", fullname, " ROIs in an Elastic Net Model"))
grid.arrange(real_pred_fiveFoldWhole_M, real_pred_fiveFold_M, ncol=2)
grid.arrange(bar_2YBins_adultPer_M, bar_3YBins_adultPer_M, ncol=2)
grid.arrange(bar_2YBins_diffRealPred_M, bar_3YBins_diffRealPred_M, ncol=2)
grid.arrange(interaction_all_M, interaction_over10_M, ncol=2)
grid.arrange(predage_RvsW_M, predage_Rvs1_M, ncol=2)
grid.arrange(predage_Rvs2_M, predage_Rvs3_M, ncol=2)
grid.arrange(predage_Rvs4_M, predage_Rvs5_M, ncol=2)
grid.arrange(predage_1vs5_M, predage_Wvs1_M, ncol=2)
grid.arrange(predage_Wvs2_M, predage_Wvs3_M, ncol=2)
grid.arrange(predage_Wvs4_M, predage_Wvs5_M, ncol=2)
grid.arrange(betas_1vs5_M, betas_Wvs1_M, ncol=2)
grid.arrange(betas_Wvs2_M, betas_Wvs3_M, ncol=2)
grid.arrange(betas_Wvs4_M, betas_Wvs5_M, ncol=2)
dev.off()

#GAM plots
pdf(file=paste0("/home/butellyn/age_prediction/plots/", type, "_gamplot.pdf"), width=6, height=6)
plot(0:10, type = "n", xaxt="n", yaxt="n", bty="n", xlab = "", ylab = "")
text(5, 8, "Females Age~ROI")
for (i in 1:length(xvars)) { 
	print(get(paste0("gamplot_", i, "_F")))
}
plot(0:10, type = "n", xaxt="n", yaxt="n", bty="n", xlab = "", ylab = "")
text(5, 8, "Males Age~ROI")
for (i in 1:length(xvars)) { 
	print(get(paste0("gamplot_", i, "_M")))
}
dev.off()













