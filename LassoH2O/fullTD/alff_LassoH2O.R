### This script produces linear models, features for which have been selected with CV Lasso (to choose lambda),
### of a given modality predicting age in two same-sex samples with no history of mental illness.
### In addition, it produces plots of lambda vs. features' beta values
###
### Ellyn Butler
### February 5, 2019


# Load libraries
library('glmnet')
library('ggplot2')
library('reshape2')
library('gridExtra')
library('psych')
library('h2o')
h2o.init(max_mem_size = "4G")

# Source my functions
source("/home/butellyn/ButlerPlotFuncs/plotFuncs_AdonPath.R")

# Load data
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

alffvars <- c('rest_jlf_alff_R_Accumbens_Area', 'rest_jlf_alff_L_Accumbens_Area', 'rest_jlf_alff_R_Amygdala', 
            'rest_jlf_alff_L_Amygdala', 'rest_jlf_alff_R_Caudate', 'rest_jlf_alff_L_Caudate', 'rest_jlf_alff_R_Cerebellum_Exterior', 
            'rest_jlf_alff_L_Cerebellum_Exterior', 'rest_jlf_alff_R_Hippocampus', 'rest_jlf_alff_L_Hippocampus', 
            'rest_jlf_alff_R_Pallidum', 'rest_jlf_alff_L_Pallidum', 'rest_jlf_alff_R_Putamen', 'rest_jlf_alff_L_Putamen', 
            'rest_jlf_alff_R_Thalamus_Proper', 'rest_jlf_alff_L_Thalamus_Proper', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_I.V', 
            'rest_jlf_alff_Cerebellar_Vermal_Lobules_VI.VII', 'rest_jlf_alff_Cerebellar_Vermal_Lobules_VIII.X', 
            'rest_jlf_alff_R_ACgG', 'rest_jlf_alff_L_ACgG', 'rest_jlf_alff_R_AIns', 'rest_jlf_alff_L_AIns', 'rest_jlf_alff_R_AOrG', 
            'rest_jlf_alff_L_AOrG', 'rest_jlf_alff_R_AnG', 'rest_jlf_alff_L_AnG', 'rest_jlf_alff_R_Calc', 'rest_jlf_alff_L_Calc', 
            'rest_jlf_alff_R_CO', 'rest_jlf_alff_L_CO', 'rest_jlf_alff_R_Cun', 'rest_jlf_alff_L_Cun', 'rest_jlf_alff_R_Ent', 
            'rest_jlf_alff_L_Ent', 'rest_jlf_alff_R_FO', 'rest_jlf_alff_L_FO', 'rest_jlf_alff_R_FRP', 'rest_jlf_alff_L_FRP', 
            'rest_jlf_alff_R_FuG', 'rest_jlf_alff_L_FuG', 'rest_jlf_alff_R_GRe', 'rest_jlf_alff_L_GRe', 'rest_jlf_alff_R_IOG', 
            'rest_jlf_alff_L_IOG', 'rest_jlf_alff_R_ITG', 'rest_jlf_alff_L_ITG', 'rest_jlf_alff_R_LiG', 'rest_jlf_alff_L_LiG', 
            'rest_jlf_alff_R_LOrG', 'rest_jlf_alff_L_LOrG', 'rest_jlf_alff_R_MCgG', 'rest_jlf_alff_L_MCgG', 'rest_jlf_alff_R_MFC', 
            'rest_jlf_alff_L_MFC', 'rest_jlf_alff_R_MFG', 'rest_jlf_alff_L_MFG', 'rest_jlf_alff_R_MOG', 'rest_jlf_alff_L_MOG', 
            'rest_jlf_alff_R_MOrG', 'rest_jlf_alff_L_MOrG', 'rest_jlf_alff_R_MPoG', 'rest_jlf_alff_L_MPoG', 'rest_jlf_alff_R_MPrG', 
            'rest_jlf_alff_L_MPrG', 'rest_jlf_alff_R_MSFG', 'rest_jlf_alff_L_MSFG', 'rest_jlf_alff_R_MTG', 'rest_jlf_alff_L_MTG', 
            'rest_jlf_alff_R_OCP', 'rest_jlf_alff_L_OCP', 'rest_jlf_alff_R_OFuG', 'rest_jlf_alff_L_OFuG', 'rest_jlf_alff_R_OpIFG', 
            'rest_jlf_alff_L_OpIFG', 'rest_jlf_alff_R_OrIFG', 'rest_jlf_alff_L_OrIFG', 'rest_jlf_alff_R_PCgG', 
            'rest_jlf_alff_L_PCgG', 'rest_jlf_alff_R_PCu', 'rest_jlf_alff_L_PCu', 'rest_jlf_alff_R_PHG', 'rest_jlf_alff_L_PHG', 
            'rest_jlf_alff_R_PIns', 'rest_jlf_alff_L_PIns', 'rest_jlf_alff_R_PO', 'rest_jlf_alff_L_PO', 'rest_jlf_alff_R_PoG', 
            'rest_jlf_alff_L_PoG', 'rest_jlf_alff_R_POrG', 'rest_jlf_alff_L_POrG', 'rest_jlf_alff_R_PP', 'rest_jlf_alff_L_PP', 
            'rest_jlf_alff_R_PrG', 'rest_jlf_alff_L_PrG', 'rest_jlf_alff_R_PT', 'rest_jlf_alff_L_PT', 'rest_jlf_alff_R_SCA', 
            'rest_jlf_alff_L_SCA', 'rest_jlf_alff_R_SFG', 'rest_jlf_alff_L_SFG', 'rest_jlf_alff_R_SMC', 'rest_jlf_alff_L_SMC', 
            'rest_jlf_alff_R_SMG', 'rest_jlf_alff_L_SMG', 'rest_jlf_alff_R_SOG', 'rest_jlf_alff_L_SOG', 'rest_jlf_alff_R_SPL', 
            'rest_jlf_alff_L_SPL', 'rest_jlf_alff_R_STG', 'rest_jlf_alff_L_STG', 'rest_jlf_alff_R_TMP', 'rest_jlf_alff_L_TMP', 
            'rest_jlf_alff_R_TrIFG', 'rest_jlf_alff_L_TrIFG', 'rest_jlf_alff_R_TTG', 'rest_jlf_alff_L_TTG')


yvar <- c('ageAtScan1')


# Start analyses
set.seed(1)

######################################################## Vars ########################################################

xvars <- alffvars

############################ Females ############################

df_F <- df[df$sex == 2, c("goassessDxpmr7", xvars, yvar)]

for (i in 1:ncol(df_F)) {
	na_index <- is.na(df_F[,i])
	df_F <- df_F[!na_index,]
}

x_td_F <- df_F[df_F$goassessDxpmr7 == "TD", xvars]
x_td_F <- as.matrix(x_td_F)
y_td_F <- df_F[df_F$goassessDxpmr7 == "TD", yvar]
x_ps_F <- df_F[df_F$goassessDxpmr7 == "PS", xvars]
x_ps_F <- as.matrix(x_ps_F)
y_ps_F <- df_F[df_F$goassessDxpmr7 == "PS", yvar]

# Create heatmap of correlations between potential predictors
cormat <- round(cor(x_td_F),2)

get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}

upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
heatmap_td_F <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
 geom_tile(color = "white")+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Pearson\nCorrelation") +
  theme_minimal()+ 
 theme(axis.text.x = element_text(angle = 60, vjust = 1, 
    size = 10, hjust = 1))+
 coord_fixed()

# Build models
#H20
xy_td_F <- cbind(x_td_F, y_td_F)
xy_ps_F <- cbind(x_ps_F, y_ps_F)
h2o_td_F <- as.h2o(xy_td_F)
h2o_ps_F <- as.h2o(xy_ps_F)
mod <- h2o.automl(x = xvars, y = "y_td_F", training_frame = h2o_td_F, max_runtime_secs = 30)
ML_td_F <- as.matrix(h2o.predict(mod@leader,h2o_td_F))
ML_ps_F <- as.matrix(h2o.predict(mod@leader,h2o_ps_F))

#Lasso
mod_td_F <- glmnet(x_td_F, y_td_F, family=c("gaussian"))
pdf(NULL)
dev.control(displaylist="enable")
plot(mod_td_F)
p_mod_td_F <- recordPlot()
invisible(dev.off())

cv.out_F <- cv.glmnet(x_td_F, y_td_F, alpha=1)
pdf(NULL)
dev.control(displaylist="enable")
plot(cv.out_F)
p_cv.out_F <- recordPlot()
invisible(dev.off())

# get the 18 variables
bestlam_min=cv.out_F$lambda.min
bestlam_1se=cv.out_F$lambda.1se
coef_min_td_F <- predict(cv.out_F,type="coefficients",s=bestlam_min)
coef_1se_td_F <- predict(cv.out_F,type="coefficients",s=bestlam_1se) ## way fewer predictors... I like this one

# predict ps
pred_ps_F <- predict(mod_td_F,s=bestlam_1se,newx=x_ps_F)

# predict td
pred_td_F <- predict(mod_td_F,s=bestlam_1se,newx=x_td_F)

# Put in Predicted Age
# ps
df_ps_F <- data.frame(x_ps_F)
df_ps_F$real_age <- y_ps_F/12
df_ps_F$pred_age_lasso <- pred_ps_F/12
df_ps_F$pred_age_h2o <- ML_ps_F/12
df_td_F <- data.frame(x_td_F)
df_td_F$real_age <- y_td_F/12
df_td_F$pred_age_lasso <- pred_td_F/12
df_td_F$pred_age_h2o <- ML_td_F/12

# Plot Percent Predicted to be over 18 in each two-year age bins, split by diagnosis
df_ps_F$diagnosis <- "PS"
df_td_F$diagnosis <- "TD"
df_both_F <- rbind(df_ps_F, df_td_F)
df_both_F$diagnosis <- as.factor(df_both_F$diagnosis)

df_both_F$age_cat <- 0
df_both_F$pred_adult_lasso <- 0
df_both_F$pred_adult_h2o <- 0

rownames(df_both_F) <- 1:nrow(df_both_F) # change row indices

for (row in 1:nrow(df_both_F)) {
	if (df_both_F[row,"real_age"] < 10) { df_both_F[row,"age_cat"] = "8_9" }
	else if (df_both_F[row,"real_age"] >= 10 & df_both_F[row,"real_age"] < 12) { df_both_F[row,"age_cat"] = "10_11" }
	else if (df_both_F[row,"real_age"] >= 12 & df_both_F[row,"real_age"] < 14) { df_both_F[row,"age_cat"] = "12_13" }
	else if (df_both_F[row,"real_age"] >= 14 & df_both_F[row,"real_age"] < 16) { df_both_F[row,"age_cat"] = "14_15" }
	else if (df_both_F[row,"real_age"] >= 16 & df_both_F[row,"real_age"] < 18) { df_both_F[row,"age_cat"] = "16_17" }
	else if (df_both_F[row,"real_age"] >= 18 & df_both_F[row,"real_age"] < 20) { df_both_F[row,"age_cat"] = "18_19" }
	else if (df_both_F[row,"real_age"] >= 20 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat"] = "20_23" }
	if (df_both_F[row,"pred_age_lasso"] >= 18) { df_both_F[row,"pred_adult_lasso"] = 1 }
	if (df_both_F[row,"pred_age_h2o"] >= 18) { df_both_F[row,"pred_adult_h2o"] = 1 }
}

df_both_F$age_cat <- as.factor(df_both_F$age_cat)

# Summary Dataframes
# lasso
summary_lasso_F <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_lasso_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_lasso_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_lasso_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_lasso_F$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

N_TD_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_lasso_F$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)


summary_lasso_F$Age_Category <- factor(summary_lasso_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

#h20
summary_h2o_F <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_h2o_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_h2o_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_h2o_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_h2o_F$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

N_TD_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_h2o_F$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)

summary_h2o_F$Age_Category <- factor(summary_h2o_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

# Plot Info
#lasso
td_lasso_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age_lasso"])$r, digits=3)
td_lasso_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age_lasso"])$p
if (td_lasso_p_F < .001) { td_lasso_p_F <- .001 } else if (td_lasso_p_F < .01) { td_lasso_p_F <- .01 } else if (td_lasso_p_F < .05) { td_lasso_p_F <- .05 } else { td_lasso_p_F <- round(td_lasso_p_F, digits=3) }
td_lasso_N_F <- nrow(df_both_F[df_both_F$diagnosis == "TD",])
ps_lasso_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age_lasso"])$r, digits=3)
ps_lasso_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age_lasso"])$p
if (ps_lasso_p_F < .001) { ps_lasso_p_F <- .001 } else if (ps_lasso_p_F < .01) { ps_lasso_p_F <- .01 } else if (ps_lasso_p_F < .05) { ps_lasso_p_F <- .05 } else { ps_lasso_p_F <- round(ps_lasso_p_F, digits=3) }
ps_lasso_N_F <- nrow(df_both_F[df_both_F$diagnosis == "PS",])

substring_lasso_F <- paste0("TD: r = ", td_lasso_corr_F, ", p < ", td_lasso_p_F,", N = ", td_lasso_N_F, "\nPS: r = ", ps_lasso_corr_F, ", p < ", ps_lasso_p_F,", N = ", ps_lasso_N_F)

#h2o
td_h2o_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age_h2o"])$r, digits=3)
td_h2o_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age_h2o"])$p
if (td_h2o_p_F < .001) { td_h2o_p_F <- .001 } else if (td_h2o_p_F < .01) { td_h2o_p_F <- .01 } else if (td_h2o_p_F < .05) { td_h2o_p_F <- .05 } else { td_h2o_p_F <- round(td_h2o_p_F, digits=3) }
td_h2o_N_F <- nrow(df_both_F[df_both_F$diagnosis == "TD",])
ps_h2o_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age_h2o"])$r, digits=3)
ps_h2o_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age_h2o"])$p
if (ps_h2o_p_F < .001) { ps_h2o_p_F <- .001 } else if (ps_h2o_p_F < .01) { ps_h2o_p_F <- .01 } else if (ps_h2o_p_F < .05) { ps_h2o_p_F <- .05 } else { ps_h2o_p_F <- round(ps_h2o_p_F, digits=3) }
ps_h2o_N_F <- nrow(df_both_F[df_both_F$diagnosis == "PS",])

substring_h2o_F <- paste0("TD: r = ", td_h2o_corr_F, ", p < ", td_h2o_p_F,", N = ", td_h2o_N_F, "\nPS: r = ", ps_h2o_corr_F, ", p < ", ps_h2o_p_F,", N = ", ps_h2o_N_F)

# Plots
real_pred_lasso_F <- ggplot(data=df_both_F, aes(real_age, pred_age_lasso, color=diagnosis)) +
	geom_point(shape = 16, size = 3, show.legend = TRUE) + theme_minimal() +
	xlim(4,25) + ylim(4,25) + geom_abline() +
	labs(title = "Female (Lasso): ALFF ROIs", subtitle = substring_lasso_F) + 
	xlab("Real Age") + ylab("Predicted Age") +
	geom_smooth(se = TRUE, method = "gam", formula = y ~ s(log(x))) + 
	theme(plot.title = element_text(family="Times", face="bold", size=20))

real_pred_h2o_F <- ggplot(data=df_both_F, aes(real_age, pred_age_h2o, color=diagnosis)) +
	geom_point(shape = 16, size = 3, show.legend = TRUE) + theme_minimal() +
	xlim(4,25) + ylim(4,25) + geom_abline() +
	labs(title = "Female (H2O): ALFF ROIs", subtitle = substring_h2o_F) + 
	xlab("Real Age") + ylab("Predicted Age") +
	geom_smooth(se = TRUE, method = "gam", formula = y ~ s(log(x))) + 
	theme(plot.title = element_text(family="Times", face="bold", size=20))

bar_lasso_F <- ggplot(data=summary_lasso_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female (Lasso): ALFF ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")

bar_h2o_F <- ggplot(data=summary_h2o_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female (H2O): ALFF ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")




############################ Males ############################

df_M <- df[df$sex == 1, c("goassessDxpmr7", xvars, yvar)]

for (i in 1:ncol(df_M)) {
	na_index <- is.na(df_M[,i])
	df_M <- df_M[!na_index,]
}

x_td_M <- df_M[df_M$goassessDxpmr7 == "TD", xvars]
x_td_M <- as.matrix(x_td_M)
y_td_M <- df_M[df_M$goassessDxpmr7 == "TD", yvar]
x_ps_M <- df_M[df_M$goassessDxpmr7 == "PS", xvars]
x_ps_M <- as.matrix(x_ps_M)
y_ps_M <- df_M[df_M$goassessDxpmr7 == "PS", yvar]

# Create heatmap of correlations between potential predictors
cormat <- round(cor(x_td_M),2)

get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
}

upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
heatmap_td_M <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
 geom_tile(color = "white")+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Pearson\nCorrelation") +
  theme_minimal()+ 
 theme(axis.text.x = element_text(angle = 60, vjust = 1, 
    size = 10, hjust = 1))+
 coord_fixed()

# Build models
#H20
xy_td_M <- cbind(x_td_M, y_td_M)
xy_ps_M <- cbind(x_ps_M, y_ps_M)
h2o_td_M <- as.h2o(xy_td_M)
h2o_ps_M <- as.h2o(xy_ps_M)
mod <- h2o.automl(x = xvars, y = "y_td_M", training_frame = h2o_td_M, max_runtime_secs = 30)
ML_td_M <- as.matrix(h2o.predict(mod@leader,h2o_td_M))
ML_ps_M <- as.matrix(h2o.predict(mod@leader,h2o_ps_M))

#Lasso
mod_td_M <- glmnet(x_td_M, y_td_M, family=c("gaussian"))
pdf(NULL)
dev.control(displaylist="enable")
plot(mod_td_M)
p_mod_td_M <- recordPlot()
invisible(dev.off())

cv.out_M <- cv.glmnet(x_td_M, y_td_M, alpha=1)
pdf(NULL)
dev.control(displaylist="enable")
plot(cv.out_M)
p_cv.out_M <- recordPlot()
invisible(dev.off())

# get the 18 variables
bestlam_min=cv.out_M$lambda.min
bestlam_1se=cv.out_M$lambda.1se
coef_min_td_M <- predict(cv.out_M,type="coefficients",s=bestlam_min)
coef_1se_td_M <- predict(cv.out_M,type="coefficients",s=bestlam_1se) ## way fewer predictors... I like this one

# predict ps
pred_ps_M <- predict(mod_td_M,s=bestlam_1se,newx=x_ps_M)

# predict td
pred_td_M <- predict(mod_td_M,s=bestlam_1se,newx=x_td_M)

# Put in Predicted Age
# ps
df_ps_M <- data.frame(x_ps_M)
df_ps_M$real_age <- y_ps_M/12
df_ps_M$pred_age_lasso <- pred_ps_M/12
df_ps_M$pred_age_h2o <- ML_ps_M/12
df_td_M <- data.frame(x_td_M)
df_td_M$real_age <- y_td_M/12
df_td_M$pred_age_lasso <- pred_td_M/12
df_td_M$pred_age_h2o <- ML_td_M/12

# Plot Percent Predicted to be over 18 in each two-year age bins, split by diagnosis
df_ps_M$diagnosis <- "PS"
df_td_M$diagnosis <- "TD"
df_both_M <- rbind(df_ps_M, df_td_M)
df_both_M$diagnosis <- as.factor(df_both_M$diagnosis)

df_both_M$age_cat <- 0
df_both_M$pred_adult_lasso <- 0
df_both_M$pred_adult_h2o <- 0

rownames(df_both_M) <- 1:nrow(df_both_M) # change row indices

for (row in 1:nrow(df_both_M)) {
	if (df_both_M[row,"real_age"] < 10) { df_both_M[row,"age_cat"] = "8_9" }
	else if (df_both_M[row,"real_age"] >= 10 & df_both_M[row,"real_age"] < 12) { df_both_M[row,"age_cat"] = "10_11" }
	else if (df_both_M[row,"real_age"] >= 12 & df_both_M[row,"real_age"] < 14) { df_both_M[row,"age_cat"] = "12_13" }
	else if (df_both_M[row,"real_age"] >= 14 & df_both_M[row,"real_age"] < 16) { df_both_M[row,"age_cat"] = "14_15" }
	else if (df_both_M[row,"real_age"] >= 16 & df_both_M[row,"real_age"] < 18) { df_both_M[row,"age_cat"] = "16_17" }
	else if (df_both_M[row,"real_age"] >= 18 & df_both_M[row,"real_age"] < 20) { df_both_M[row,"age_cat"] = "18_19" }
	else if (df_both_M[row,"real_age"] >= 20 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat"] = "20_23" }
	if (df_both_M[row,"pred_age_lasso"] >= 18) { df_both_M[row,"pred_adult_lasso"] = 1 }
	if (df_both_M[row,"pred_age_h2o"] >= 18) { df_both_M[row,"pred_adult_h2o"] = 1 }
}

df_both_M$age_cat <- as.factor(df_both_M$age_cat)

# Summary Dataframes
# lasso
summary_lasso_M <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_lasso_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_lasso_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_lasso_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_lasso_M$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

N_TD_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_lasso == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_lasso_M$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)


summary_lasso_M$Age_Category <- factor(summary_lasso_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

#h20
summary_h2o_M <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_h2o_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_h2o_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_h2o_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_h2o_M$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

N_TD_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult_h2o == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_h2o_M$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)

summary_h2o_M$Age_Category <- factor(summary_h2o_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

# Plot Info
#lasso
td_lasso_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age_lasso"])$r, digits=3)
td_lasso_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age_lasso"])$p
if (td_lasso_p_M < .001) { td_lasso_p_M <- .001 } else if (td_lasso_p_M < .01) { td_lasso_p_M <- .01 } else if (td_lasso_p_M < .05) { td_lasso_p_M <- .05 } else { td_lasso_p_M <- round(td_lasso_p_M, digits=3) }
td_lasso_N_M <- nrow(df_both_M[df_both_M$diagnosis == "TD",])
ps_lasso_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age_lasso"])$r, digits=3)
ps_lasso_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age_lasso"])$p
if (ps_lasso_p_M < .001) { ps_lasso_p_M <- .001 } else if (ps_lasso_p_M < .01) { ps_lasso_p_M <- .01 } else if (ps_lasso_p_M < .05) { ps_lasso_p_M <- .05 } else { ps_lasso_p_M <- round(ps_lasso_p_M, digits=3) }
ps_lasso_N_M <- nrow(df_both_M[df_both_M$diagnosis == "PS",])

substring_lasso_M <- paste0("TD: r = ", td_lasso_corr_M, ", p < ", td_lasso_p_M,", N = ", td_lasso_N_M, "\nPS: r = ", ps_lasso_corr_M, ", p < ", ps_lasso_p_M,", N = ", ps_lasso_N_M)

#h2o
td_h2o_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age_h2o"])$r, digits=3)
td_h2o_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age_h2o"])$p
if (td_h2o_p_M < .001) { td_h2o_p_M <- .001 } else if (td_h2o_p_M < .01) { td_h2o_p_M <- .01 } else if (td_h2o_p_M < .05) { td_h2o_p_M <- .05 } else { td_h2o_p_M <- round(td_h2o_p_M, digits=3) }
td_h2o_N_M <- nrow(df_both_M[df_both_M$diagnosis == "TD",])
ps_h2o_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age_h2o"])$r, digits=3)
ps_h2o_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age_h2o"])$p
if (ps_h2o_p_M < .001) { ps_h2o_p_M <- .001 } else if (ps_h2o_p_M < .01) { ps_h2o_p_M <- .01 } else if (ps_h2o_p_M < .05) { ps_h2o_p_M <- .05 } else { ps_h2o_p_M <- round(ps_h2o_p_M, digits=3) }
ps_h2o_N_M <- nrow(df_both_M[df_both_M$diagnosis == "PS",])

substring_h2o_M <- paste0("TD: r = ", td_h2o_corr_M, ", p < ", td_h2o_p_M,", N = ", td_h2o_N_M, "\nPS: r = ", ps_h2o_corr_M, ", p < ", ps_h2o_p_M,", N = ", ps_h2o_N_M)

# Plots
real_pred_lasso_M <- ggplot(data=df_both_M, aes(real_age, pred_age_lasso, color=diagnosis)) +
	geom_point(shape = 16, size = 3, show.legend = TRUE) + theme_minimal() +
	xlim(4,25) + ylim(4,25) + geom_abline() +
	labs(title = "Male (Lasso): ALFF ROIs", subtitle = substring_lasso_M) + 
	xlab("Real Age") + ylab("Predicted Age") +
	geom_smooth(se = TRUE, method = "gam", formula = y ~ s(log(x))) + 
	theme(plot.title = element_text(family="Times", face="bold", size=20))

real_pred_h2o_M <- ggplot(data=df_both_M, aes(real_age, pred_age_h2o, color=diagnosis)) +
	geom_point(shape = 16, size = 3, show.legend = TRUE) + theme_minimal() +
	xlim(4,25) + ylim(4,25) + geom_abline() +
	labs(title = "Male (H2O): ALFF ROIs", subtitle = substring_h2o_M) + 
	xlab("Real Age") + ylab("Predicted Age") +
	geom_smooth(se = TRUE, method = "gam", formula = y ~ s(log(x))) + 
	theme(plot.title = element_text(family="Times", face="bold", size=20))

bar_lasso_M <- ggplot(data=summary_lasso_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male (Lasso): ALFF ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")

bar_h2o_M <- ggplot(data=summary_h2o_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male (H2O): ALFF ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")



# Save plots
pdf(file="/home/butellyn/age_prediction/plots/alff_LassoH2O.pdf", width=12, height=6)
grid.arrange(real_pred_lasso_F, real_pred_lasso_M, ncol=2)
grid.arrange(bar_lasso_F, bar_lasso_M, ncol=2)
p_cv.out_F
grid.arrange(real_pred_h2o_F, real_pred_h2o_M, ncol=2)
grid.arrange(bar_h2o_F, bar_h2o_M, ncol=2)
p_cv.out_M
dev.off()















