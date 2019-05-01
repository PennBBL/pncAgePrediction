### This script produces two histograms, one for each sex, of the differences in correlations between PSs and TDs
### for each across for every modality. It then identifies the ROIs in the top 5% of these histograms, and identifies
### the percentage that come from each modality.
###
### Ellyn Butler
### March 1, 2019 - present

# Load libraries
library('ggplot2')
library('reshape2')
library('gridExtra')
library('corrplot')

# Source my functions
source("/home/butellyn/ButlerPlotFuncs/plotFuncs_AdonPath.R")

# Load data
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

alffrois <- c('rest_jlf_alff_R_Accumbens_Area', 'rest_jlf_alff_L_Accumbens_Area', 'rest_jlf_alff_R_Amygdala', 'rest_jlf_alff_L_Amygdala', 'rest_jlf_alff_R_Caudate', 'rest_jlf_alff_L_Caudate', 'rest_jlf_alff_R_Cerebellum_Exterior', 'rest_jlf_alff_L_Cerebellum_Exterior', 'rest_jlf_alff_R_Hippocampus', 'rest_jlf_alff_L_Hippocampus', 'rest_jlf_alff_R_Pallidum', 'rest_jlf_alff_L_Pallidum', 'rest_jlf_alff_R_Putamen', 'rest_jlf_alff_L_Putamen', 'rest_jlf_alff_R_Thalamus_Proper', 'rest_jlf_alff_L_Thalamus_Proper', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_I.V', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_VI.VII', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_VIII.X', 'rest_jlf_alff_R_ACgG', 'rest_jlf_alff_L_ACgG', 'rest_jlf_alff_R_AIns', 'rest_jlf_alff_L_AIns', 'rest_jlf_alff_R_AOrG', 'rest_jlf_alff_L_AOrG', 'rest_jlf_alff_R_AnG', 'rest_jlf_alff_L_AnG', 'rest_jlf_alff_R_Calc', 'rest_jlf_alff_L_Calc', 'rest_jlf_alff_R_CO', 'rest_jlf_alff_L_CO', 'rest_jlf_alff_R_Cun', 'rest_jlf_alff_L_Cun', 'rest_jlf_alff_R_Ent', 'rest_jlf_alff_L_Ent', 'rest_jlf_alff_R_FO', 'rest_jlf_alff_L_FO', 'rest_jlf_alff_R_FRP', 'rest_jlf_alff_L_FRP', 'rest_jlf_alff_R_FuG', 'rest_jlf_alff_L_FuG', 'rest_jlf_alff_R_GRe', 'rest_jlf_alff_L_GRe', 'rest_jlf_alff_R_IOG', 'rest_jlf_alff_L_IOG', 'rest_jlf_alff_R_ITG', 'rest_jlf_alff_L_ITG', 'rest_jlf_alff_R_LiG', 'rest_jlf_alff_L_LiG', 'rest_jlf_alff_R_LOrG', 'rest_jlf_alff_L_LOrG', 'rest_jlf_alff_R_MCgG', 'rest_jlf_alff_L_MCgG', 'rest_jlf_alff_R_MFC', 'rest_jlf_alff_L_MFC', 'rest_jlf_alff_R_MFG', 'rest_jlf_alff_L_MFG', 'rest_jlf_alff_R_MOG', 'rest_jlf_alff_L_MOG', 'rest_jlf_alff_R_MOrG', 'rest_jlf_alff_L_MOrG', 'rest_jlf_alff_R_MPoG', 'rest_jlf_alff_L_MPoG', 'rest_jlf_alff_R_MPrG', 'rest_jlf_alff_L_MPrG', 'rest_jlf_alff_R_MSFG', 'rest_jlf_alff_L_MSFG', 'rest_jlf_alff_R_MTG', 'rest_jlf_alff_L_MTG', 'rest_jlf_alff_R_OCP', 'rest_jlf_alff_L_OCP', 'rest_jlf_alff_R_OFuG', 'rest_jlf_alff_L_OFuG', 'rest_jlf_alff_R_OpIFG', 'rest_jlf_alff_L_OpIFG', 'rest_jlf_alff_R_OrIFG', 'rest_jlf_alff_L_OrIFG', 'rest_jlf_alff_R_PCgG', 'rest_jlf_alff_L_PCgG', 'rest_jlf_alff_R_PCu', 'rest_jlf_alff_L_PCu', 'rest_jlf_alff_R_PHG', 'rest_jlf_alff_L_PHG', 'rest_jlf_alff_R_PIns', 'rest_jlf_alff_L_PIns', 'rest_jlf_alff_R_PO', 'rest_jlf_alff_L_PO', 'rest_jlf_alff_R_PoG', 'rest_jlf_alff_L_PoG', 'rest_jlf_alff_R_POrG', 'rest_jlf_alff_L_POrG', 'rest_jlf_alff_R_PP', 'rest_jlf_alff_L_PP', 'rest_jlf_alff_R_PrG', 'rest_jlf_alff_L_PrG', 'rest_jlf_alff_R_PT', 'rest_jlf_alff_L_PT', 'rest_jlf_alff_R_SCA', 'rest_jlf_alff_L_SCA', 'rest_jlf_alff_R_SFG', 'rest_jlf_alff_L_SFG', 'rest_jlf_alff_R_SMC', 'rest_jlf_alff_L_SMC', 'rest_jlf_alff_R_SMG', 'rest_jlf_alff_L_SMG', 'rest_jlf_alff_R_SOG', 'rest_jlf_alff_L_SOG', 'rest_jlf_alff_R_SPL', 'rest_jlf_alff_L_SPL', 'rest_jlf_alff_R_STG', 'rest_jlf_alff_L_STG', 'rest_jlf_alff_R_TMP', 'rest_jlf_alff_L_TMP', 'rest_jlf_alff_R_TrIFG', 'rest_jlf_alff_L_TrIFG', 'rest_jlf_alff_R_TTG', 'rest_jlf_alff_L_TTG')

cbfrois <- c('pcasl_jlf_cbf_R_Accumbens_Area', 'pcasl_jlf_cbf_L_Accumbens_Area', 'pcasl_jlf_cbf_R_Amygdala', 'pcasl_jlf_cbf_L_Amygdala', 'pcasl_jlf_cbf_R_Caudate', 'pcasl_jlf_cbf_L_Caudate', 'pcasl_jlf_cbf_R_Hippocampus', 'pcasl_jlf_cbf_L_Hippocampus', 'pcasl_jlf_cbf_R_Pallidum', 'pcasl_jlf_cbf_L_Pallidum', 'pcasl_jlf_cbf_R_Putamen', 'pcasl_jlf_cbf_L_Putamen', 'pcasl_jlf_cbf_R_Thalamus_Proper', 'pcasl_jlf_cbf_L_Thalamus_Proper', 'pcasl_jlf_cbf_R_ACgG', 'pcasl_jlf_cbf_L_ACgG', 'pcasl_jlf_cbf_R_AIns', 'pcasl_jlf_cbf_L_AIns', 'pcasl_jlf_cbf_R_AOrG', 'pcasl_jlf_cbf_L_AOrG', 'pcasl_jlf_cbf_R_AnG', 'pcasl_jlf_cbf_L_AnG', 'pcasl_jlf_cbf_R_Calc', 'pcasl_jlf_cbf_L_Calc', 'pcasl_jlf_cbf_R_CO', 'pcasl_jlf_cbf_L_CO', 'pcasl_jlf_cbf_R_Cun', 'pcasl_jlf_cbf_L_Cun', 'pcasl_jlf_cbf_R_Ent', 'pcasl_jlf_cbf_L_Ent', 'pcasl_jlf_cbf_R_FO', 'pcasl_jlf_cbf_L_FO', 'pcasl_jlf_cbf_R_FRP', 'pcasl_jlf_cbf_L_FRP', 'pcasl_jlf_cbf_R_FuG', 'pcasl_jlf_cbf_L_FuG', 'pcasl_jlf_cbf_R_GRe', 'pcasl_jlf_cbf_L_GRe', 'pcasl_jlf_cbf_R_IOG', 'pcasl_jlf_cbf_L_IOG', 'pcasl_jlf_cbf_R_ITG', 'pcasl_jlf_cbf_L_ITG', 'pcasl_jlf_cbf_R_LiG', 'pcasl_jlf_cbf_L_LiG', 'pcasl_jlf_cbf_R_LOrG', 'pcasl_jlf_cbf_L_LOrG', 'pcasl_jlf_cbf_R_MCgG', 'pcasl_jlf_cbf_L_MCgG', 'pcasl_jlf_cbf_R_MFC', 'pcasl_jlf_cbf_L_MFC', 'pcasl_jlf_cbf_R_MFG', 'pcasl_jlf_cbf_L_MFG', 'pcasl_jlf_cbf_R_MOG', 'pcasl_jlf_cbf_L_MOG', 'pcasl_jlf_cbf_R_MOrG',  'pcasl_jlf_cbf_L_MOrG', 'pcasl_jlf_cbf_R_MPoG', 'pcasl_jlf_cbf_L_MPoG', 'pcasl_jlf_cbf_R_MPrG', 'pcasl_jlf_cbf_L_MPrG', 'pcasl_jlf_cbf_R_MSFG', 'pcasl_jlf_cbf_L_MSFG', 'pcasl_jlf_cbf_R_MTG', 'pcasl_jlf_cbf_R_OCP', 'pcasl_jlf_cbf_L_OCP', 'pcasl_jlf_cbf_R_OFuG', 'pcasl_jlf_cbf_L_OFuG', 'pcasl_jlf_cbf_R_OpIFG', 'pcasl_jlf_cbf_L_OpIFG', 'pcasl_jlf_cbf_R_OrIFG', 'pcasl_jlf_cbf_L_OrIFG', 'pcasl_jlf_cbf_R_PCgG', 'pcasl_jlf_cbf_L_PCgG', 'pcasl_jlf_cbf_R_PCu', 'pcasl_jlf_cbf_L_PCu', 'pcasl_jlf_cbf_R_PHG', 'pcasl_jlf_cbf_L_PHG', 'pcasl_jlf_cbf_R_PIns', 'pcasl_jlf_cbf_L_PIns', 'pcasl_jlf_cbf_R_PO', 'pcasl_jlf_cbf_L_PO', 'pcasl_jlf_cbf_R_PoG', 'pcasl_jlf_cbf_L_PoG', 'pcasl_jlf_cbf_R_POrG', 'pcasl_jlf_cbf_L_POrG', 'pcasl_jlf_cbf_R_PP', 'pcasl_jlf_cbf_L_PP', 'pcasl_jlf_cbf_R_PrG', 'pcasl_jlf_cbf_L_PrG', 'pcasl_jlf_cbf_R_PT', 'pcasl_jlf_cbf_L_PT', 'pcasl_jlf_cbf_R_SCA', 'pcasl_jlf_cbf_L_SCA', 'pcasl_jlf_cbf_R_SFG', 'pcasl_jlf_cbf_L_SFG', 'pcasl_jlf_cbf_R_SMC', 'pcasl_jlf_cbf_L_SMC', 'pcasl_jlf_cbf_R_SMG', 'pcasl_jlf_cbf_L_SMG', 'pcasl_jlf_cbf_R_SOG', 'pcasl_jlf_cbf_L_SOG', 'pcasl_jlf_cbf_R_SPL', 'pcasl_jlf_cbf_L_SPL', 'pcasl_jlf_cbf_R_STG', 'pcasl_jlf_cbf_L_STG', 'pcasl_jlf_cbf_R_TMP', 'pcasl_jlf_cbf_L_TMP', 'pcasl_jlf_cbf_R_TrIFG', 'pcasl_jlf_cbf_L_TrIFG', 'pcasl_jlf_cbf_R_TTG', 'pcasl_jlf_cbf_L_TTG')

cortrois <- c('mprage_jlf_ct_R_ACgG', 'mprage_jlf_ct_L_ACgG', 'mprage_jlf_ct_R_AIns', 'mprage_jlf_ct_L_AIns', 'mprage_jlf_ct_R_AOrG', 'mprage_jlf_ct_L_AOrG', 'mprage_jlf_ct_R_AnG', 'mprage_jlf_ct_L_AnG', 'mprage_jlf_ct_R_Calc', 'mprage_jlf_ct_L_Calc', 'mprage_jlf_ct_R_CO', 'mprage_jlf_ct_L_CO', 'mprage_jlf_ct_R_Cun', 'mprage_jlf_ct_L_Cun', 'mprage_jlf_ct_R_Ent', 'mprage_jlf_ct_L_Ent', 'mprage_jlf_ct_R_FO', 'mprage_jlf_ct_L_FO', 'mprage_jlf_ct_R_FRP', 'mprage_jlf_ct_L_FRP', 'mprage_jlf_ct_R_FuG', 'mprage_jlf_ct_L_FuG', 'mprage_jlf_ct_R_GRe', 'mprage_jlf_ct_L_GRe', 'mprage_jlf_ct_R_IOG', 'mprage_jlf_ct_L_IOG', 'mprage_jlf_ct_R_ITG', 'mprage_jlf_ct_L_ITG', 'mprage_jlf_ct_R_LiG', 'mprage_jlf_ct_L_LiG', 'mprage_jlf_ct_R_LOrG', 'mprage_jlf_ct_L_LOrG', 'mprage_jlf_ct_R_MCgG', 'mprage_jlf_ct_L_MCgG', 'mprage_jlf_ct_R_MFC', 'mprage_jlf_ct_L_MFC', 'mprage_jlf_ct_R_MFG', 'mprage_jlf_ct_L_MFG', 'mprage_jlf_ct_R_MOG', 'mprage_jlf_ct_L_MOG', 'mprage_jlf_ct_R_MOrG', 'mprage_jlf_ct_L_MOrG', 'mprage_jlf_ct_R_MPoG', 'mprage_jlf_ct_L_MPoG', 'mprage_jlf_ct_R_MPrG', 'mprage_jlf_ct_L_MPrG', 'mprage_jlf_ct_R_MSFG', 'mprage_jlf_ct_L_MSFG', 'mprage_jlf_ct_R_MTG', 'mprage_jlf_ct_L_MTG', 'mprage_jlf_ct_R_OCP', 'mprage_jlf_ct_L_OCP', 'mprage_jlf_ct_R_OFuG', 'mprage_jlf_ct_L_OFuG', 'mprage_jlf_ct_R_OpIFG', 'mprage_jlf_ct_L_OpIFG', 'mprage_jlf_ct_R_OrIFG', 'mprage_jlf_ct_L_OrIFG', 'mprage_jlf_ct_R_PCgG', 'mprage_jlf_ct_L_PCgG', 'mprage_jlf_ct_R_PCu', 'mprage_jlf_ct_L_PCu', 'mprage_jlf_ct_R_PHG', 'mprage_jlf_ct_L_PHG', 'mprage_jlf_ct_R_PIns', 'mprage_jlf_ct_L_PIns', 'mprage_jlf_ct_R_PO', 'mprage_jlf_ct_L_PO', 'mprage_jlf_ct_R_PoG', 'mprage_jlf_ct_L_PoG', 'mprage_jlf_ct_R_POrG', 'mprage_jlf_ct_L_POrG', 'mprage_jlf_ct_R_PP', 'mprage_jlf_ct_L_PP', 'mprage_jlf_ct_R_PrG', 'mprage_jlf_ct_L_PrG', 'mprage_jlf_ct_R_PT', 'mprage_jlf_ct_L_PT', 'mprage_jlf_ct_R_SCA', 'mprage_jlf_ct_L_SCA', 'mprage_jlf_ct_R_SFG', 'mprage_jlf_ct_L_SFG', 'mprage_jlf_ct_R_SMC', 'mprage_jlf_ct_L_SMC', 'mprage_jlf_ct_R_SMG', 'mprage_jlf_ct_L_SMG', 'mprage_jlf_ct_R_SOG', 'mprage_jlf_ct_L_SOG', 'mprage_jlf_ct_R_SPL', 'mprage_jlf_ct_L_SPL', 'mprage_jlf_ct_R_STG', 'mprage_jlf_ct_L_STG', 'mprage_jlf_ct_R_TMP', 'mprage_jlf_ct_L_TMP')

farois <- c('dti_dtitk_jhulabel_fa_mcp', 'dti_dtitk_jhulabel_fa_pct', 'dti_dtitk_jhulabel_fa_gcc', 'dti_dtitk_jhulabel_fa_bcc', 'dti_dtitk_jhulabel_fa_scc', 'dti_dtitk_jhulabel_fa_fnx', 'dti_dtitk_jhulabel_fa_cst_r', 'dti_dtitk_jhulabel_fa_cst_l', 'dti_dtitk_jhulabel_fa_mel_l', 'dti_dtitk_jhulabel_fa_mel_r', 'dti_dtitk_jhulabel_fa_icp_r', 'dti_dtitk_jhulabel_fa_icp_l', 'dti_dtitk_jhulabel_fa_scp_r', 'dti_dtitk_jhulabel_fa_scp_l', 'dti_dtitk_jhulabel_fa_cp_r', 'dti_dtitk_jhulabel_fa_cp_l', 'dti_dtitk_jhulabel_fa_alic_r', 'dti_dtitk_jhulabel_fa_alic_l', 'dti_dtitk_jhulabel_fa_plic_r', 'dti_dtitk_jhulabel_fa_plic_l', 'dti_dtitk_jhulabel_fa_rlic_r', 'dti_dtitk_jhulabel_fa_rlic_l', 'dti_dtitk_jhulabel_fa_acr_r', 'dti_dtitk_jhulabel_fa_acr_l', 'dti_dtitk_jhulabel_fa_scr_r', 'dti_dtitk_jhulabel_fa_scr_l', 'dti_dtitk_jhulabel_fa_pcr_r', 'dti_dtitk_jhulabel_fa_pcr_l', 'dti_dtitk_jhulabel_fa_ptr_r', 'dti_dtitk_jhulabel_fa_ptr_l', 'dti_dtitk_jhulabel_fa_ss_r', 'dti_dtitk_jhulabel_fa_ss_l', 'dti_dtitk_jhulabel_fa_ec_r', 'dti_dtitk_jhulabel_fa_ec_l', 'dti_dtitk_jhulabel_fa_cgc_r', 'dti_dtitk_jhulabel_fa_cgc_l', 'dti_dtitk_jhulabel_fa_cgh_r', 'dti_dtitk_jhulabel_fa_cgh_l', 'dti_dtitk_jhulabel_fa_fnx_st_r', 'dti_dtitk_jhulabel_fa_fnx_st_l', 'dti_dtitk_jhulabel_fa_slf_r', 'dti_dtitk_jhulabel_fa_slf_l', 'dti_dtitk_jhulabel_fa_sfo_r', 'dti_dtitk_jhulabel_fa_sfo_l', 'dti_dtitk_jhulabel_fa_uf_r', 'dti_dtitk_jhulabel_fa_uf_l', 'dti_dtitk_jhulabel_fa_tap_r', 'dti_dtitk_jhulabel_fa_tap_l')

gmdrois <- c('mprage_jlf_gmd_R_Accumbens_Area', 'mprage_jlf_gmd_L_Accumbens_Area', 'mprage_jlf_gmd_R_Amygdala', 'mprage_jlf_gmd_L_Amygdala',  'mprage_jlf_gmd_R_Caudate', 'mprage_jlf_gmd_L_Caudate', 'mprage_jlf_gmd_R_Cerebellum_Exterior', 'mprage_jlf_gmd_L_Cerebellum_Exterior', 'mprage_jlf_gmd_R_Hippocampus', 'mprage_jlf_gmd_L_Hippocampus', 'mprage_jlf_gmd_R_Pallidum', 'mprage_jlf_gmd_L_Pallidum', 'mprage_jlf_gmd_R_Putamen', 'mprage_jlf_gmd_L_Putamen', 'mprage_jlf_gmd_R_Thalamus_Proper', 'mprage_jlf_gmd_L_Thalamus_Proper', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VI.VII', 'mprage_jlf_gmd_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_gmd_R_ACgG', 'mprage_jlf_gmd_L_ACgG', 'mprage_jlf_gmd_R_AIns', 'mprage_jlf_gmd_L_AIns', 'mprage_jlf_gmd_R_AOrG', 'mprage_jlf_gmd_L_AOrG', 'mprage_jlf_gmd_R_AnG', 'mprage_jlf_gmd_L_AnG', 'mprage_jlf_gmd_R_Calc', 'mprage_jlf_gmd_L_Calc', 'mprage_jlf_gmd_R_CO', 'mprage_jlf_gmd_L_CO', 'mprage_jlf_gmd_R_Cun', 'mprage_jlf_gmd_L_Cun', 'mprage_jlf_gmd_R_Ent', 'mprage_jlf_gmd_L_Ent', 'mprage_jlf_gmd_R_FO', 'mprage_jlf_gmd_L_FO', 'mprage_jlf_gmd_R_FRP', 'mprage_jlf_gmd_L_FRP', 'mprage_jlf_gmd_R_FuG', 'mprage_jlf_gmd_L_FuG', 'mprage_jlf_gmd_R_GRe', 'mprage_jlf_gmd_L_GRe', 'mprage_jlf_gmd_R_IOG', 'mprage_jlf_gmd_L_IOG', 'mprage_jlf_gmd_R_ITG', 'mprage_jlf_gmd_L_ITG', 'mprage_jlf_gmd_R_LiG', 'mprage_jlf_gmd_L_LiG', 'mprage_jlf_gmd_R_LOrG', 'mprage_jlf_gmd_L_LOrG', 'mprage_jlf_gmd_R_MCgG', 'mprage_jlf_gmd_L_MCgG', 'mprage_jlf_gmd_R_MFC', 'mprage_jlf_gmd_L_MFC', 'mprage_jlf_gmd_R_MFG', 'mprage_jlf_gmd_L_MFG', 'mprage_jlf_gmd_R_MOG', 'mprage_jlf_gmd_L_MOG', 'mprage_jlf_gmd_R_MOrG', 'mprage_jlf_gmd_L_MOrG', 'mprage_jlf_gmd_R_MPoG', 'mprage_jlf_gmd_L_MPoG', 'mprage_jlf_gmd_R_MPrG', 'mprage_jlf_gmd_L_MPrG', 'mprage_jlf_gmd_R_MSFG', 'mprage_jlf_gmd_L_MSFG', 'mprage_jlf_gmd_R_MTG', 'mprage_jlf_gmd_L_MTG', 'mprage_jlf_gmd_R_OCP', 'mprage_jlf_gmd_L_OCP', 'mprage_jlf_gmd_R_OFuG', 'mprage_jlf_gmd_L_OFuG', 'mprage_jlf_gmd_R_OpIFG', 'mprage_jlf_gmd_L_OpIFG', 'mprage_jlf_gmd_R_OrIFG', 'mprage_jlf_gmd_L_OrIFG', 'mprage_jlf_gmd_R_PCgG', 'mprage_jlf_gmd_L_PCgG', 'mprage_jlf_gmd_R_PCu', 'mprage_jlf_gmd_L_PCu', 'mprage_jlf_gmd_R_PHG', 'mprage_jlf_gmd_L_PHG', 'mprage_jlf_gmd_R_PIns', 'mprage_jlf_gmd_L_PIns', 'mprage_jlf_gmd_R_PO', 'mprage_jlf_gmd_L_PO', 'mprage_jlf_gmd_R_PoG', 'mprage_jlf_gmd_L_PoG', 'mprage_jlf_gmd_R_POrG', 'mprage_jlf_gmd_L_POrG', 'mprage_jlf_gmd_R_PP', 'mprage_jlf_gmd_L_PP', 'mprage_jlf_gmd_R_PrG', 'mprage_jlf_gmd_L_PrG', 'mprage_jlf_gmd_R_PT', 'mprage_jlf_gmd_L_PT', 'mprage_jlf_gmd_R_SCA', 'mprage_jlf_gmd_L_SCA', 'mprage_jlf_gmd_R_SFG', 'mprage_jlf_gmd_L_SFG', 'mprage_jlf_gmd_R_SMC', 'mprage_jlf_gmd_L_SMC', 'mprage_jlf_gmd_R_SMG', 'mprage_jlf_gmd_L_SMG', 'mprage_jlf_gmd_R_SOG', 'mprage_jlf_gmd_L_SOG', 'mprage_jlf_gmd_R_SPL', 'mprage_jlf_gmd_L_SPL', 'mprage_jlf_gmd_R_STG', 'mprage_jlf_gmd_L_STG', 'mprage_jlf_gmd_R_TMP', 'mprage_jlf_gmd_L_TMP', 'mprage_jlf_gmd_R_TrIFG', 'mprage_jlf_gmd_L_TrIFG', 'mprage_jlf_gmd_R_TTG', 'mprage_jlf_gmd_L_TTG')

gmvrois <- c('mprage_jlf_vol_R_Accumbens_Area', 'mprage_jlf_vol_L_Accumbens_Area', 'mprage_jlf_vol_R_Amygdala', 'mprage_jlf_vol_L_Amygdala', 'mprage_jlf_vol_R_Caudate', 'mprage_jlf_vol_L_Caudate', 'mprage_jlf_vol_R_Cerebellum_Exterior', 'mprage_jlf_vol_L_Cerebellum_Exterior', 'mprage_jlf_vol_R_Hippocampus', 'mprage_jlf_vol_L_Hippocampus','mprage_jlf_vol_R_Pallidum', 'mprage_jlf_vol_L_Pallidum', 'mprage_jlf_vol_R_Putamen', 'mprage_jlf_vol_L_Putamen', 'mprage_jlf_vol_R_Thalamus_Proper', 'mprage_jlf_vol_L_Thalamus_Proper', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VI.VII', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_vol_R_ACgG', 'mprage_jlf_vol_L_ACgG', 'mprage_jlf_vol_R_AIns', 'mprage_jlf_vol_L_AIns', 'mprage_jlf_vol_R_AOrG', 'mprage_jlf_vol_L_AOrG', 'mprage_jlf_vol_R_AnG', 'mprage_jlf_vol_L_AnG', 'mprage_jlf_vol_R_Calc', 'mprage_jlf_vol_L_Calc', 'mprage_jlf_vol_R_CO', 'mprage_jlf_vol_L_CO', 'mprage_jlf_vol_R_Cun', 'mprage_jlf_vol_L_Cun', 'mprage_jlf_vol_R_Ent', 'mprage_jlf_vol_L_Ent', 'mprage_jlf_vol_R_FO', 'mprage_jlf_vol_L_FO', 'mprage_jlf_vol_R_FRP', 'mprage_jlf_vol_L_FRP', 'mprage_jlf_vol_R_FuG', 'mprage_jlf_vol_L_FuG', 'mprage_jlf_vol_R_GRe', 'mprage_jlf_vol_L_GRe', 'mprage_jlf_vol_R_IOG', 'mprage_jlf_vol_L_IOG', 'mprage_jlf_vol_R_ITG', 'mprage_jlf_vol_L_ITG', 'mprage_jlf_vol_R_LiG', 'mprage_jlf_vol_L_LiG', 'mprage_jlf_vol_R_LOrG', 'mprage_jlf_vol_L_LOrG', 'mprage_jlf_vol_R_MCgG', 'mprage_jlf_vol_L_MCgG', 'mprage_jlf_vol_R_MFC', 'mprage_jlf_vol_L_MFC', 'mprage_jlf_vol_R_MFG', 'mprage_jlf_vol_L_MFG', 'mprage_jlf_vol_R_MOG', 'mprage_jlf_vol_L_MOG', 'mprage_jlf_vol_R_MOrG', 'mprage_jlf_vol_L_MOrG', 'mprage_jlf_vol_R_MPoG', 'mprage_jlf_vol_L_MPoG', 'mprage_jlf_vol_R_MPrG', 'mprage_jlf_vol_L_MPrG', 'mprage_jlf_vol_R_MSFG', 'mprage_jlf_vol_L_MSFG', 'mprage_jlf_vol_R_MTG', 'mprage_jlf_vol_L_MTG', 'mprage_jlf_vol_R_OCP', 'mprage_jlf_vol_L_OCP', 'mprage_jlf_vol_R_OFuG', 'mprage_jlf_vol_L_OFuG', 'mprage_jlf_vol_R_OpIFG', 'mprage_jlf_vol_L_OpIFG', 'mprage_jlf_vol_R_OrIFG', 'mprage_jlf_vol_L_OrIFG', 'mprage_jlf_vol_R_PCgG', 'mprage_jlf_vol_L_PCgG', 'mprage_jlf_vol_R_PCu', 'mprage_jlf_vol_L_PCu', 'mprage_jlf_vol_R_PHG', 'mprage_jlf_vol_L_PHG', 'mprage_jlf_vol_R_PIns', 'mprage_jlf_vol_L_PIns', 'mprage_jlf_vol_R_PO', 'mprage_jlf_vol_L_PO', 'mprage_jlf_vol_R_PoG', 'mprage_jlf_vol_L_PoG', 'mprage_jlf_vol_R_POrG', 'mprage_jlf_vol_L_POrG', 'mprage_jlf_vol_R_PP', 'mprage_jlf_vol_L_PP', 'mprage_jlf_vol_R_PrG', 'mprage_jlf_vol_L_PrG', 'mprage_jlf_vol_R_PT', 'mprage_jlf_vol_L_PT', 'mprage_jlf_vol_R_SCA', 'mprage_jlf_vol_L_SCA', 'mprage_jlf_vol_R_SFG', 'mprage_jlf_vol_L_SFG', 'mprage_jlf_vol_R_SMC', 'mprage_jlf_vol_L_SMC', 'mprage_jlf_vol_R_SMG', 'mprage_jlf_vol_L_SMG', 'mprage_jlf_vol_R_SOG', 'mprage_jlf_vol_L_SOG', 'mprage_jlf_vol_R_SPL', 'mprage_jlf_vol_L_SPL', 'mprage_jlf_vol_R_STG', 'mprage_jlf_vol_L_STG', 'mprage_jlf_vol_R_TMP', 'mprage_jlf_vol_L_TMP', 'mprage_jlf_vol_R_TrIFG', 'mprage_jlf_vol_L_TrIFG', 'mprage_jlf_vol_R_TTG', 'mprage_jlf_vol_L_TTG')

mdrois <- c('dti_jlf_tr_R_Accumbens_Area','dti_jlf_tr_L_Accumbens_Area', 'dti_jlf_tr_R_Amygdala', 'dti_jlf_tr_L_Amygdala', 'dti_jlf_tr_R_Caudate', 'dti_jlf_tr_L_Caudate', 'dti_jlf_tr_R_Cerebellum_Exterior', 'dti_jlf_tr_L_Cerebellum_Exterior', 'dti_jlf_tr_R_Hippocampus', 'dti_jlf_tr_L_Hippocampus', 'dti_jlf_tr_R_Pallidum', 'dti_jlf_tr_L_Pallidum', 'dti_jlf_tr_R_Putamen', 'dti_jlf_tr_L_Putamen', 'dti_jlf_tr_R_Thalamus_Proper', 'dti_jlf_tr_L_Thalamus_Proper', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_I.V', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VI.VII', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VIII.X', 'dti_jlf_tr_R_ACgG', 'dti_jlf_tr_L_ACgG', 'dti_jlf_tr_R_AIns', 'dti_jlf_tr_L_AIns', 'dti_jlf_tr_R_AOrG', 'dti_jlf_tr_L_AOrG', 'dti_jlf_tr_R_AnG', 'dti_jlf_tr_L_AnG', 'dti_jlf_tr_R_Calc', 'dti_jlf_tr_L_Calc', 'dti_jlf_tr_R_CO', 'dti_jlf_tr_L_CO', 'dti_jlf_tr_R_Cun', 'dti_jlf_tr_L_Cun', 'dti_jlf_tr_R_Ent', 'dti_jlf_tr_L_Ent', 'dti_jlf_tr_R_FO', 'dti_jlf_tr_L_FO', 'dti_jlf_tr_R_FRP', 'dti_jlf_tr_L_FRP', 'dti_jlf_tr_R_FuG', 'dti_jlf_tr_L_FuG', 'dti_jlf_tr_R_GRe', 'dti_jlf_tr_L_GRe', 'dti_jlf_tr_R_IOG', 'dti_jlf_tr_L_IOG', 'dti_jlf_tr_R_ITG', 'dti_jlf_tr_L_ITG', 'dti_jlf_tr_R_LiG', 'dti_jlf_tr_L_LiG', 'dti_jlf_tr_R_LOrG', 'dti_jlf_tr_L_LOrG', 'dti_jlf_tr_R_MCgG', 'dti_jlf_tr_L_MCgG', 'dti_jlf_tr_R_MFC', 'dti_jlf_tr_L_MFC', 'dti_jlf_tr_R_MFG', 'dti_jlf_tr_L_MFG', 'dti_jlf_tr_R_MOG', 'dti_jlf_tr_L_MOG', 'dti_jlf_tr_R_MOrG', 'dti_jlf_tr_L_MOrG', 'dti_jlf_tr_R_MPoG', 'dti_jlf_tr_L_MPoG', 'dti_jlf_tr_R_MPrG', 'dti_jlf_tr_L_MPrG', 'dti_jlf_tr_R_MSFG', 'dti_jlf_tr_L_MSFG', 'dti_jlf_tr_R_MTG', 'dti_jlf_tr_L_MTG', 'dti_jlf_tr_R_OCP', 'dti_jlf_tr_L_OCP', 'dti_jlf_tr_R_OFuG', 'dti_jlf_tr_L_OFuG', 'dti_jlf_tr_R_OpIFG', 'dti_jlf_tr_L_OpIFG', 'dti_jlf_tr_R_OrIFG', 'dti_jlf_tr_L_OrIFG', 'dti_jlf_tr_R_PCgG', 'dti_jlf_tr_L_PCgG', 'dti_jlf_tr_R_PCu', 'dti_jlf_tr_L_PCu', 'dti_jlf_tr_R_PHG', 'dti_jlf_tr_L_PHG', 'dti_jlf_tr_R_PIns', 'dti_jlf_tr_L_PIns', 'dti_jlf_tr_R_PO', 'dti_jlf_tr_L_PO', 'dti_jlf_tr_R_PoG', 'dti_jlf_tr_L_PoG', 'dti_jlf_tr_R_POrG', 'dti_jlf_tr_L_POrG', 'dti_jlf_tr_R_PP', 'dti_jlf_tr_L_PP', 'dti_jlf_tr_R_PrG', 'dti_jlf_tr_L_PrG', 'dti_jlf_tr_R_PT', 'dti_jlf_tr_L_PT', 'dti_jlf_tr_R_SCA', 'dti_jlf_tr_L_SCA', 'dti_jlf_tr_R_SFG', 'dti_jlf_tr_L_SFG', 'dti_jlf_tr_R_SMC', 'dti_jlf_tr_L_SMC', 'dti_jlf_tr_R_SMG', 'dti_jlf_tr_L_SMG', 'dti_jlf_tr_R_SOG', 'dti_jlf_tr_L_SOG', 'dti_jlf_tr_R_SPL', 'dti_jlf_tr_L_SPL', 'dti_jlf_tr_R_STG', 'dti_jlf_tr_L_STG', 'dti_jlf_tr_R_TMP', 'dti_jlf_tr_L_TMP', 'dti_jlf_tr_R_TrIFG', 'dti_jlf_tr_L_TrIFG')

rehorois <- c('rest_jlf_reho_R_Accumbens_Area', 'rest_jlf_reho_L_Accumbens_Area', 'rest_jlf_reho_R_Amygdala', 'rest_jlf_reho_L_Amygdala', 'rest_jlf_reho_R_Caudate', 'rest_jlf_reho_L_Caudate', 'rest_jlf_reho_R_Cerebellum_Exterior', 'rest_jlf_reho_L_Cerebellum_Exterior', 'rest_jlf_reho_R_Hippocampus', 'rest_jlf_reho_L_Hippocampus', 'rest_jlf_reho_R_Pallidum', 'rest_jlf_reho_L_Pallidum', 'rest_jlf_reho_R_Putamen', 'rest_jlf_reho_L_Putamen', 'rest_jlf_reho_R_Thalamus_Proper', 'rest_jlf_reho_L_Thalamus_Proper', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_I.V', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_VI.VII', 'rest_jlf_reho_Cerebellar_Vermal_Lobules_VIII.X', 'rest_jlf_reho_R_ACgG', 'rest_jlf_reho_L_ACgG', 'rest_jlf_reho_R_AIns', 'rest_jlf_reho_L_AIns', 'rest_jlf_reho_R_AOrG', 'rest_jlf_reho_L_AOrG', 'rest_jlf_reho_R_AnG', 'rest_jlf_reho_L_AnG', 'rest_jlf_reho_R_Calc', 'rest_jlf_reho_L_Calc', 'rest_jlf_reho_R_CO', 'rest_jlf_reho_L_CO', 'rest_jlf_reho_R_Cun', 'rest_jlf_reho_L_Cun', 'rest_jlf_reho_R_Ent', 'rest_jlf_reho_L_Ent', 'rest_jlf_reho_R_FO', 'rest_jlf_reho_L_FO', 'rest_jlf_reho_R_FRP', 'rest_jlf_reho_L_FRP', 'rest_jlf_reho_R_FuG', 'rest_jlf_reho_L_FuG', 'rest_jlf_reho_R_GRe', 'rest_jlf_reho_L_GRe', 'rest_jlf_reho_R_IOG', 'rest_jlf_reho_L_IOG', 'rest_jlf_reho_R_ITG', 'rest_jlf_reho_L_ITG', 'rest_jlf_reho_R_LiG', 'rest_jlf_reho_L_LiG', 'rest_jlf_reho_R_LOrG', 'rest_jlf_reho_L_LOrG', 'rest_jlf_reho_R_MCgG', 'rest_jlf_reho_L_MCgG', 'rest_jlf_reho_R_MFC', 'rest_jlf_reho_L_MFC', 'rest_jlf_reho_R_MFG', 'rest_jlf_reho_L_MFG', 'rest_jlf_reho_R_MOG', 'rest_jlf_reho_L_MOG', 'rest_jlf_reho_R_MOrG', 'rest_jlf_reho_L_MOrG', 'rest_jlf_reho_R_MPoG', 'rest_jlf_reho_L_MPoG', 'rest_jlf_reho_R_MPrG', 'rest_jlf_reho_L_MPrG', 'rest_jlf_reho_R_MSFG', 'rest_jlf_reho_L_MSFG', 'rest_jlf_reho_R_MTG', 'rest_jlf_reho_L_MTG', 'rest_jlf_reho_R_OCP', 'rest_jlf_reho_L_OCP', 'rest_jlf_reho_R_OFuG', 'rest_jlf_reho_L_OFuG', 'rest_jlf_reho_R_OpIFG', 'rest_jlf_reho_L_OpIFG', 'rest_jlf_reho_R_OrIFG', 'rest_jlf_reho_L_OrIFG', 'rest_jlf_reho_R_PCgG', 'rest_jlf_reho_L_PCgG', 'rest_jlf_reho_R_PCu', 'rest_jlf_reho_L_PCu', 'rest_jlf_reho_R_PHG', 'rest_jlf_reho_L_PHG', 'rest_jlf_reho_R_PIns', 'rest_jlf_reho_L_PIns', 'rest_jlf_reho_R_PO', 'rest_jlf_reho_L_PO', 'rest_jlf_reho_R_PoG', 'rest_jlf_reho_L_PoG', 'rest_jlf_reho_R_POrG', 'rest_jlf_reho_L_POrG', 'rest_jlf_reho_R_PP', 'rest_jlf_reho_L_PP', 'rest_jlf_reho_R_PrG', 'rest_jlf_reho_L_PrG', 'rest_jlf_reho_R_PT', 'rest_jlf_reho_L_PT', 'rest_jlf_reho_R_SCA', 'rest_jlf_reho_L_SCA', 'rest_jlf_reho_R_SFG', 'rest_jlf_reho_L_SFG', 'rest_jlf_reho_R_SMC', 'rest_jlf_reho_L_SMC', 'rest_jlf_reho_R_SMG', 'rest_jlf_reho_L_SMG', 'rest_jlf_reho_R_SOG', 'rest_jlf_reho_L_SOG', 'rest_jlf_reho_R_SPL', 'rest_jlf_reho_L_SPL', 'rest_jlf_reho_R_STG', 'rest_jlf_reho_L_STG', 'rest_jlf_reho_R_TMP', 'rest_jlf_reho_L_TMP', 'rest_jlf_reho_R_TrIFG', 'rest_jlf_reho_L_TrIFG', 'rest_jlf_reho_R_TTG', 'rest_jlf_reho_L_TTG')

rois <- c(alffrois, cbfrois, cortrois, farois, gmdrois, gmvrois, mdrois, rehorois)
df <- df[,c('bblid','goassessDxpmr7','sex','ageAtScan1',rois)]
	    

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

for (i in 1:ncol(df)) {
	na_index <- is.na(df[,i])
	df <- df[!na_index,]
}


######### ======================================================================== #########

td_F <- df[df$psTerminal == "TD" & df$sex == "2",]
ps_F <- df[df$psTerminal == "PS" & df$sex == "2",]
td_M <- df[df$psTerminal == "TD" & df$sex == "1",]
ps_M <- df[df$psTerminal == "PS" & df$sex == "1",]

corr_df <- as.data.frame(matrix(0, nrow=835, ncol=6))
colnames(corr_df) <- c("ROI", "Modality", "TD_F_Corr", "PS_F_Corr", "TD_M_Corr", "PS_M_Corr")
corr_df$ROI <- colnames(td_F)[5:839]

for (i in 1:nrow(corr_df)) {
	modal <- strsplit(corr_df[i, "ROI"], split="_")[[1]][[3]]
	if (modal == "alff") { corr_df[i, "Modality"] <- "ALFF" 
	} else if (modal == "cbf") { corr_df[i, "Modality"] <- "CBF" 
	} else if (modal == "ct") { corr_df[i, "Modality"] <- "Cort" 
	} else if (modal == "jhulabel") { corr_df[i, "Modality"] <- "FA" 
	} else if (modal == "gmd") { corr_df[i, "Modality"] <- "GMD" 
	} else if (modal == "vol") { corr_df[i, "Modality"] <- "GMV" 
	} else if (modal == "tr") { corr_df[i, "Modality"] <- "MD" 
	} else if (modal == "reho") { corr_df[i, "Modality"] <- "REHO" }
}
corr_df$Modality <- factor(corr_df$Modality)

for (i in 1:835) {
	corr_df[i,3] <- cor(td_F[,i+4],td_F$ageAtScan1, method="spearman")
	corr_df[i,4] <- cor(ps_F[,i+4],ps_F$ageAtScan1, method="spearman")
	corr_df[i,5] <- cor(td_M[,i+4],td_M$ageAtScan1, method="spearman")
	corr_df[i,6] <- cor(ps_M[,i+4],ps_M$ageAtScan1, method="spearman")
}

# TD F	
p_gm_td_F <- ggplot(corr_df[corr_df$Modality == "Cort" | corr_df$Modality == "GMD" | 
	corr_df$Modality == "GMV" | corr_df$Modality == "MD",], aes(x=TD_F_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Struc. Gray Matter Correlation Age~ROI (TD F)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

p_funcw_td_F <- ggplot(corr_df[corr_df$Modality == "ALFF" | corr_df$Modality == "CBF" | 
	corr_df$Modality == "REHO" | corr_df$Modality == "FA",], aes(x=TD_F_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Func. & White Correlation Age~ROI (TD F)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

# PS F
p_gm_ps_F <- ggplot(corr_df[corr_df$Modality == "Cort" | corr_df$Modality == "GMD" | 
	corr_df$Modality == "GMV" | corr_df$Modality == "MD",], aes(x=PS_F_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Struc. Gray Matter Correlation Age~ROI (PS F)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

p_funcw_ps_F <- ggplot(corr_df[corr_df$Modality == "ALFF" | corr_df$Modality == "CBF" | 
	corr_df$Modality == "REHO" | corr_df$Modality == "FA",], aes(x=PS_F_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) +
	labs(title="Func. & White Correlation Age~ROI (PS F)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

# TD M	
p_gm_td_M <- ggplot(corr_df[corr_df$Modality == "Cort" | corr_df$Modality == "GMD" | 
	corr_df$Modality == "GMV" | corr_df$Modality == "MD",], aes(x=TD_M_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Struc. Gray Matter Correlation Age~ROI (TD M)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

p_funcw_td_M <- ggplot(corr_df[corr_df$Modality == "ALFF" | corr_df$Modality == "CBF" | 
	corr_df$Modality == "REHO" | corr_df$Modality == "FA",], aes(x=TD_M_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Func. & White Correlation Age~ROI (TD M)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

# PS M
p_gm_ps_M <- ggplot(corr_df[corr_df$Modality == "Cort" | corr_df$Modality == "GMD" | 
	corr_df$Modality == "GMV" | corr_df$Modality == "MD",], aes(x=PS_M_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Struc. Gray Matter Correlation Age~ROI (PS M)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))

p_funcw_ps_M <- ggplot(corr_df[corr_df$Modality == "ALFF" | corr_df$Modality == "CBF" | 
	corr_df$Modality == "REHO" | corr_df$Modality == "FA",], aes(x=PS_M_Corr, fill=Modality, color=Modality)) + theme_minimal() +
	geom_histogram(position="identity", alpha=0.5) + 
	labs(title="Func. & White Correlation Age~ROI (PS M)",x="Correlation", y = "# of ROIs") +
	theme(plot.title = element_text(size=12), axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
	scale_x_continuous(breaks=c(-.6, -.4, -.2, 0, .2, .4, .6), limits=c(-.7, .7)) +
	scale_y_continuous(breaks=c(0, 10, 20, 30, 40, 50), limits=c(0, 55))



pdf(file="/home/butellyn/age_prediction/plots/corr_SexDiag_AgeMod.pdf", width=9, height=9)
grid.arrange(p_gm_td_F, p_gm_ps_F, p_funcw_td_F, p_funcw_ps_F, nrow=2, ncol=2)
grid.arrange(p_gm_td_M, p_gm_ps_M, p_funcw_td_M, p_funcw_ps_M, nrow=2, ncol=2)
dev.off()






