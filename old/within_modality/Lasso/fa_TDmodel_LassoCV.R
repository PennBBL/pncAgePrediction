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

# Load data
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

yvar <- c('ageAtScan1')


# Start analyses
set.seed(1)

######################################################## Fractional Anisotropy ########################################################

xvars <- c('dti_dtitk_jhulabel_fa_mcp', 'dti_dtitk_jhulabel_fa_pct', 
          'dti_dtitk_jhulabel_fa_gcc', 'dti_dtitk_jhulabel_fa_bcc', 'dti_dtitk_jhulabel_fa_scc', 
          'dti_dtitk_jhulabel_fa_fnx', 'dti_dtitk_jhulabel_fa_cst_r', 'dti_dtitk_jhulabel_fa_cst_l', 
          'dti_dtitk_jhulabel_fa_mel_l', 'dti_dtitk_jhulabel_fa_mel_r', 'dti_dtitk_jhulabel_fa_icp_r', 
          'dti_dtitk_jhulabel_fa_icp_l', 'dti_dtitk_jhulabel_fa_scp_r', 'dti_dtitk_jhulabel_fa_scp_l', 
          'dti_dtitk_jhulabel_fa_cp_r', 'dti_dtitk_jhulabel_fa_cp_l', 'dti_dtitk_jhulabel_fa_alic_r', 
          'dti_dtitk_jhulabel_fa_alic_l', 'dti_dtitk_jhulabel_fa_plic_r', 'dti_dtitk_jhulabel_fa_plic_l', 
          'dti_dtitk_jhulabel_fa_rlic_r', 'dti_dtitk_jhulabel_fa_rlic_l', 'dti_dtitk_jhulabel_fa_acr_r', 
          'dti_dtitk_jhulabel_fa_acr_l', 'dti_dtitk_jhulabel_fa_scr_r', 'dti_dtitk_jhulabel_fa_scr_l', 
          'dti_dtitk_jhulabel_fa_pcr_r', 'dti_dtitk_jhulabel_fa_pcr_l', 'dti_dtitk_jhulabel_fa_ptr_r', 
          'dti_dtitk_jhulabel_fa_ptr_l', 'dti_dtitk_jhulabel_fa_ss_r', 'dti_dtitk_jhulabel_fa_ss_l', 
          'dti_dtitk_jhulabel_fa_ec_r', 'dti_dtitk_jhulabel_fa_ec_l', 'dti_dtitk_jhulabel_fa_cgc_r', 
          'dti_dtitk_jhulabel_fa_cgc_l', 'dti_dtitk_jhulabel_fa_cgh_r', 'dti_dtitk_jhulabel_fa_cgh_l', 
          'dti_dtitk_jhulabel_fa_fnx_st_r', 'dti_dtitk_jhulabel_fa_fnx_st_l', 'dti_dtitk_jhulabel_fa_slf_r', 
          'dti_dtitk_jhulabel_fa_slf_l', 'dti_dtitk_jhulabel_fa_sfo_r', 'dti_dtitk_jhulabel_fa_sfo_l', 
          'dti_dtitk_jhulabel_fa_uf_r', 'dti_dtitk_jhulabel_fa_uf_l', 'dti_dtitk_jhulabel_fa_tap_r', 
          'dti_dtitk_jhulabel_fa_tap_l')

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

# Plot Real vs. Predicted Age
# ps
df_ps_F <- data.frame(x_ps_F)
df_ps_F$real_age <- y_ps_F/12
df_ps_F$pred_age <- pred_ps_F/12

df_ps_F$pc <- predict(prcomp(~real_age+pred_age, df_ps_F))[,1]

ps_real_pred_F <- ggplot(data=df_ps_F, aes(real_age, pred_age, color=pc)) +
	geom_point(shape = 16, size = 5, show.legend = FALSE) + theme_minimal() +
	scale_color_gradient(low = "#0091ff", high = "#f0650e") + xlim(0,24) + ylim(0,24) + geom_abline() +
	ggtitle("PS Female: Fractional Anisotropy Brain Age") + 
	xlab("Real Age") + ylab("Predicted Age")

# td
df_td_F <- data.frame(x_td_F)
df_td_F$real_age <- y_td_F/12
df_td_F$pred_age <- pred_td_F/12

df_td_F$pc <- predict(prcomp(~real_age+pred_age, df_td_F))[,1]

td_real_pred_F <- ggplot(data=df_td_F, aes(real_age, pred_age, color=pc)) +
	geom_point(shape = 16, size = 5, show.legend = FALSE) + theme_minimal() +
	scale_color_gradient(low = "#0091ff", high = "#f0650e") + xlim(0,24) + ylim(0,24) + geom_abline() +
	ggtitle("TD Female: Fractional Anisotropy Brain Age") + 
	xlab("Real Age") + ylab("Predicted Age")

# Plot Percent Predicted to be over 18 in each two-year age bins, split by diagnosis
df_ps_F$diagnosis <- "PS"
df_td_F$diagnosis <- "TD"
df_both_F <- rbind(df_ps_F, df_td_F)
df_both_F$diagnosis <- as.factor(df_both_F$diagnosis)

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
	else if (df_both_F[row,"real_age"] >= 20 & df_both_F[row,"real_age"] < 22) { df_both_F[row,"age_cat"] = "20_21" }
	else if (df_both_F[row,"real_age"] >= 22 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat"] = "22_23" }
	if (df_both_F[row,"pred_age"] >= 18) { df_both_F[row,"pred_adult"] = 1 }
}

df_both_F$age_cat <- as.factor(df_both_F$age_cat)

summary_both_F <- data.frame(matrix(0, nrow=16, ncol=4))
colnames(summary_both_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_both_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", 
	"20_21", "20_21", "22_23", "22_23")
summary_both_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_20_21 <- 100*(length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_20_21 <- 100*(length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "PS", "age_cat"]))
TD_22_23 <- 100*(length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_22_23 <- 100*(length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_both_F$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_21, PS_20_21, TD_22_23, PS_22_23)

N_TD_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "8_9" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "10_11" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "12_13" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "14_15" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "16_17" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "18_19" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_20_21 <- paste0(length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_20_21 <- paste0(length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_21" & df_both_F$diagnosis == "PS", "age_cat"]))
N_TD_22_23 <- paste0(length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_22_23 <- paste0(length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "22_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_both_F$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_21, N_PS_20_21, N_TD_22_23, N_PS_22_23)


summary_both_F$Age_Category <- factor(summary_both_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_21", "22_23"))

bar_F <- ggplot(data=summary_both_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=3.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female Brain Age: Fractional Anisotropy") + xlab("Age Category") + ylab("Predicted Adult (%)")

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

# put ageAtScan1 and predicted age back into x_ps_M
# ps
df_ps_M <- data.frame(x_ps_M)
df_ps_M$real_age <- y_ps_M/12
df_ps_M$pred_age <- pred_ps_M/12

df_ps_M$pc <- predict(prcomp(~real_age+pred_age, df_ps_M))[,1]

ps_real_pred_M <- ggplot(data=df_ps_M, aes(real_age, pred_age, color=pc)) +
	geom_point(shape = 16, size = 5, show.legend = FALSE) + theme_minimal() +
	scale_color_gradient(low = "#0091ff", high = "#f0650e") + xlim(0,24) + ylim(0,24) + geom_abline() +
	ggtitle("PS Male: Fractional Anisotropy Brain Age") + 
	xlab("Real Age") + ylab("Predicted Age")

# td
df_td_M <- data.frame(x_td_M)
df_td_M$real_age <- y_td_M/12
df_td_M$pred_age <- pred_td_M/12

df_td_M$pc <- predict(prcomp(~real_age+pred_age, df_td_M))[,1]

td_real_pred_M <- ggplot(data=df_td_M, aes(real_age, pred_age, color=pc)) +
	geom_point(shape = 16, size = 5, show.legend = FALSE) + theme_minimal() +
	scale_color_gradient(low = "#0091ff", high = "#f0650e") + xlim(0,24) + ylim(0,24) + geom_abline() +
	ggtitle("TD Male: Fractional Anisotropy Brain Age") + 
	xlab("Real Age") + ylab("Predicted Age")

# Plot Percent Predicted to be over 18 in each two-year age bins, split by diagnosis
df_ps_M$diagnosis <- "PS"
df_td_M$diagnosis <- "TD"
df_both_M <- rbind(df_ps_M, df_td_M)
df_both_M$diagnosis <- as.factor(df_both_M$diagnosis)

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
	else if (df_both_M[row,"real_age"] >= 20 & df_both_M[row,"real_age"] < 22) { df_both_M[row,"age_cat"] = "20_21" }
	else if (df_both_M[row,"real_age"] >= 22 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat"] = "22_23" }
	if (df_both_M[row,"pred_age"] >= 18) { df_both_M[row,"pred_adult"] = 1 }
}

df_both_M$age_cat <- as.factor(df_both_M$age_cat)

summary_both_M <- data.frame(matrix(0, nrow=16, ncol=4))
colnames(summary_both_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_both_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", 
	"20_21", "20_21", "22_23", "22_23")
summary_both_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_8_9 <- 100*(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_10_11 <- 100*(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_12_13 <- 100*(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_14_15 <- 100*(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_16_17 <- 100*(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_18_19 <- 100*(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_20_21 <- 100*(length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_20_21 <- 100*(length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "PS", "age_cat"]))
TD_22_23 <- 100*(length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_22_23 <- 100*(length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_both_M$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_21, PS_20_21, TD_22_23, PS_22_23)

N_TD_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_8_9 <- paste0(length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "8_9" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_10_11 <- paste0(length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "10_11" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_12_13 <- paste0(length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "12_13" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_14_15 <- paste0(length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "14_15" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_16_17 <- paste0(length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "16_17" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_18_19 <- paste0(length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "18_19" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_20_21 <- paste0(length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_20_21 <- paste0(length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_21" & df_both_M$diagnosis == "PS", "age_cat"]))
N_TD_22_23 <- paste0(length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_22_23 <- paste0(length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "22_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_both_M$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_21, N_PS_20_21, N_TD_22_23, N_PS_22_23)


summary_both_M$Age_Category <- factor(summary_both_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_21", "22_23"))

bar_M <- ggplot(data=summary_both_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=3.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male Brain Age: Fractional Anisotropy") + xlab("Age Category") + ylab("Predicted Adult (%)")



# Save plots
pdf(file="/home/butellyn/age_prediction/plots/fa_TDmodel_LassoCV.pdf", width=8, height=6)
grid.arrange(td_real_pred_F, ps_real_pred_F, ncol=2)
p_mod_td_F
p_cv.out_F
grid.arrange(bar_F, ncol=1)
grid.arrange(td_real_pred_M, ps_real_pred_M, ncol=2)
p_mod_td_M
p_cv.out_M
grid.arrange(bar_M, ncol=1)
dev.off()















