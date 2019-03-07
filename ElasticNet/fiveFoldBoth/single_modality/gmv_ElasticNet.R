### This script produces linear models, features for which have been selected with CV Lasso (to choose lambda),
### of a given modality predicting age in two same-sex samples with no history of mental illness.
### In addition, it produces plots of lambda vs. features' beta values
###
### Ellyn Butler
### March 7, 2019


# Load libraries
library('glmnet')
library('ggplot2')
library('reshape2')
library('gridExtra')
library('psych')
library('dplyr')
library('caret')

# Source my functions
source("/home/butellyn/ButlerPlotFuncs/plotFuncs_AdonPath.R")

# Load data
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

gmvvars <- c('mprage_jlf_vol_R_Accumbens_Area', 'mprage_jlf_vol_L_Accumbens_Area', 'mprage_jlf_vol_R_Amygdala', 'mprage_jlf_vol_L_Amygdala', 'mprage_jlf_vol_R_Caudate', 'mprage_jlf_vol_L_Caudate', 'mprage_jlf_vol_R_Cerebellum_Exterior', 'mprage_jlf_vol_L_Cerebellum_Exterior', 'mprage_jlf_vol_R_Hippocampus', 'mprage_jlf_vol_L_Hippocampus','mprage_jlf_vol_R_Pallidum', 'mprage_jlf_vol_L_Pallidum', 'mprage_jlf_vol_R_Putamen', 'mprage_jlf_vol_L_Putamen', 'mprage_jlf_vol_R_Thalamus_Proper', 'mprage_jlf_vol_L_Thalamus_Proper', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_I.V', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VI.VII', 'mprage_jlf_vol_Cerebellar_Vermal_Lobules_VIII.X', 'mprage_jlf_vol_R_ACgG', 'mprage_jlf_vol_L_ACgG', 'mprage_jlf_vol_R_AIns', 'mprage_jlf_vol_L_AIns', 'mprage_jlf_vol_R_AOrG', 'mprage_jlf_vol_L_AOrG', 'mprage_jlf_vol_R_AnG', 'mprage_jlf_vol_L_AnG', 'mprage_jlf_vol_R_Calc', 'mprage_jlf_vol_L_Calc', 'mprage_jlf_vol_R_CO', 'mprage_jlf_vol_L_CO', 'mprage_jlf_vol_R_Cun', 'mprage_jlf_vol_L_Cun', 'mprage_jlf_vol_R_Ent', 'mprage_jlf_vol_L_Ent', 'mprage_jlf_vol_R_FO', 'mprage_jlf_vol_L_FO', 'mprage_jlf_vol_R_FRP', 'mprage_jlf_vol_L_FRP', 'mprage_jlf_vol_R_FuG', 'mprage_jlf_vol_L_FuG', 'mprage_jlf_vol_R_GRe', 'mprage_jlf_vol_L_GRe', 'mprage_jlf_vol_R_IOG', 'mprage_jlf_vol_L_IOG', 'mprage_jlf_vol_R_ITG', 'mprage_jlf_vol_L_ITG', 'mprage_jlf_vol_R_LiG', 'mprage_jlf_vol_L_LiG', 'mprage_jlf_vol_R_LOrG', 'mprage_jlf_vol_L_LOrG', 'mprage_jlf_vol_R_MCgG', 'mprage_jlf_vol_L_MCgG', 'mprage_jlf_vol_R_MFC', 'mprage_jlf_vol_L_MFC', 'mprage_jlf_vol_R_MFG', 'mprage_jlf_vol_L_MFG', 'mprage_jlf_vol_R_MOG', 'mprage_jlf_vol_L_MOG', 'mprage_jlf_vol_R_MOrG', 'mprage_jlf_vol_L_MOrG', 'mprage_jlf_vol_R_MPoG', 'mprage_jlf_vol_L_MPoG', 'mprage_jlf_vol_R_MPrG', 'mprage_jlf_vol_L_MPrG', 'mprage_jlf_vol_R_MSFG', 'mprage_jlf_vol_L_MSFG', 'mprage_jlf_vol_R_MTG', 'mprage_jlf_vol_L_MTG', 'mprage_jlf_vol_R_OCP', 'mprage_jlf_vol_L_OCP', 'mprage_jlf_vol_R_OFuG', 'mprage_jlf_vol_L_OFuG', 'mprage_jlf_vol_R_OpIFG', 'mprage_jlf_vol_L_OpIFG', 'mprage_jlf_vol_R_OrIFG', 'mprage_jlf_vol_L_OrIFG', 'mprage_jlf_vol_R_PCgG', 'mprage_jlf_vol_L_PCgG', 'mprage_jlf_vol_R_PCu', 'mprage_jlf_vol_L_PCu', 'mprage_jlf_vol_R_PHG', 'mprage_jlf_vol_L_PHG', 'mprage_jlf_vol_R_PIns', 'mprage_jlf_vol_L_PIns', 'mprage_jlf_vol_R_PO', 'mprage_jlf_vol_L_PO', 'mprage_jlf_vol_R_PoG', 'mprage_jlf_vol_L_PoG', 'mprage_jlf_vol_R_POrG', 'mprage_jlf_vol_L_POrG', 'mprage_jlf_vol_R_PP', 'mprage_jlf_vol_L_PP', 'mprage_jlf_vol_R_PrG', 'mprage_jlf_vol_L_PrG', 'mprage_jlf_vol_R_PT', 'mprage_jlf_vol_L_PT', 'mprage_jlf_vol_R_SCA', 'mprage_jlf_vol_L_SCA', 'mprage_jlf_vol_R_SFG', 'mprage_jlf_vol_L_SFG', 'mprage_jlf_vol_R_SMC', 'mprage_jlf_vol_L_SMC', 'mprage_jlf_vol_R_SMG', 'mprage_jlf_vol_L_SMG', 'mprage_jlf_vol_R_SOG', 'mprage_jlf_vol_L_SOG', 'mprage_jlf_vol_R_SPL', 'mprage_jlf_vol_L_SPL', 'mprage_jlf_vol_R_STG', 'mprage_jlf_vol_L_STG', 'mprage_jlf_vol_R_TMP', 'mprage_jlf_vol_L_TMP', 'mprage_jlf_vol_R_TrIFG', 'mprage_jlf_vol_L_TrIFG', 'mprage_jlf_vol_R_TTG', 'mprage_jlf_vol_L_TTG')


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

xvars <- gmvvars

############################ Females ############################

df_F <- df[df$sex == 2, c("bblid", "psTerminal", xvars, yvar)]
df_F$bblid <- as.factor(df_F$bblid)

for (i in 1:ncol(df_F)) {
	na_index <- is.na(df_F[,i])
	df_F <- df_F[!na_index,]
}

td_F <- df_F[df_F$psTerminal == "TD",]
rownames(td_F) <- 1:nrow(td_F)

# TD folds
folds5_TD <- createFolds(td_F$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5_TD[[i]]
	trainfolds <- subset(folds5_TD,!(grepl(i, names(folds5_TD))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_td_train_F_", i), td_F[train, c("bblid", xvars)])
	assign(paste0("x_td_test_F_", i), td_F[test, c("bblid", xvars)])
	assign(paste0("y_td_train_F_", i), td_F[train, c("bblid", yvar)])
	assign(paste0("y_td_test_F_", i), td_F[test, c("bblid", yvar)])
}

# PS folds
ps_F <- df_F[df_F$psTerminal == "PS",]
rownames(ps_F) <- 1:nrow(ps_F)
folds5_PS <- createFolds(ps_F$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5_PS[[i]]
	trainfolds <- subset(folds5_PS,!(grepl(i, names(folds5_PS))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_ps_train_F_", i), ps_F[train, c("bblid", xvars)])
	assign(paste0("x_ps_test_F_", i), ps_F[test, c("bblid", xvars)])
	assign(paste0("y_ps_train_F_", i), ps_F[train, c("bblid", yvar)])
	assign(paste0("y_ps_test_F_", i), ps_F[test, c("bblid", yvar)])
}


#####Elastic Net: out of sample predictions for TD's, and 1/5 each iteration for PS's
for (i in 1:5) {
	# td train
	x_td_train_df <- get(paste0("x_td_train_F_", i))
	x_td_train_input <- x_td_train_df[,xvars]
	y_td_train_df <- get(paste0("y_td_train_F_", i))
	y_td_train_input <- y_td_train_df$ageAtScan1
	# ps train
	x_ps_train_df <- get(paste0("x_ps_train_F_", i))
	x_ps_train_input <- x_ps_train_df[,xvars]
	y_ps_train_df <- get(paste0("y_ps_train_F_", i))
	y_ps_train_input <- y_ps_train_df$ageAtScan1
	# continued
	x_train_input <- rbind(x_td_train_input, x_ps_train_input)
	ageAtScan1 <- c(y_td_train_input, y_ps_train_input)
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_F_", i), elastic_net_model)
	# td predicted values
	x_td_test_df <- get(paste0("x_td_test_F_", i))
	y_td_test_df <- get(paste0("y_td_test_F_", i))
	y_td_test_predicted <- predict(elastic_net_model, x_td_test_df)
	# ps predicted values
	x_ps_test_df <- get(paste0("x_ps_test_F_", i))
	y_ps_test_df <- get(paste0("y_ps_test_F_", i))
	y_ps_test_predicted <- predict(elastic_net_model, x_ps_test_df)
	if (i == 1) { 
		df_td_F <- merge(cbind(x_td_test_df, y_td_test_predicted), y_td_test_df)
		df_ps_F <- merge(cbind(x_ps_test_df, y_ps_test_predicted), y_ps_test_df)
	} else { 
		df_td_F <- rbind(df_td_F, merge(cbind(x_td_test_df, y_td_test_predicted), y_td_test_df))
		df_ps_F <- rbind(df_ps_F, merge(cbind(x_ps_test_df, y_ps_test_predicted), y_ps_test_df))
	}
}
names(df_td_F)[names(df_td_F) == 'y_td_test_predicted'] <- 'predicted_ageAtScan1'
names(df_ps_F)[names(df_ps_F) == 'y_ps_test_predicted'] <- 'predicted_ageAtScan1'

# Create years age variables
df_ps_F$real_age <- df_ps_F$ageAtScan1/12
df_ps_F$pred_age <- df_ps_F$predicted_ageAtScan1/12
df_td_F$real_age <- df_td_F$ageAtScan1/12
df_td_F$pred_age <- df_td_F$predicted_ageAtScan1/12

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
	else if (df_both_F[row,"real_age"] >= 20 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat"] = "20_23" }
	if (df_both_F[row,"pred_age"] >= 18) { df_both_F[row,"pred_adult"] = 1 }
}

df_both_F$age_cat <- as.factor(df_both_F$age_cat)

# Summary Dataframes
# Elastic Net
summary_F <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_F$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

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
TD_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"])/length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_F$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

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
N_TD_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat"]),"/",length(df_both_F[df_both_F$age_cat == "20_23" & df_both_F$diagnosis == "PS", "age_cat"]))
summary_F$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)


summary_F$Age_Category <- factor(summary_F$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

# Three-year age bins
df_both_F$age_cat_3 <- 0

for (row in 1:nrow(df_both_F)) {
	if (df_both_F[row,"real_age"] < 12) { df_both_F[row,"age_cat_3"] = "8_11" }
	else if (df_both_F[row,"real_age"] >= 12 & df_both_F[row,"real_age"] < 15) { df_both_F[row,"age_cat_3"] = "12_14" }
	else if (df_both_F[row,"real_age"] >= 15 & df_both_F[row,"real_age"] < 18) { df_both_F[row,"age_cat_3"] = "15_17" }
	else if (df_both_F[row,"real_age"] >= 18 & df_both_F[row,"real_age"] < 21) { df_both_F[row,"age_cat_3"] = "18_20" }
	else if (df_both_F[row,"real_age"] >= 21 & df_both_F[row,"real_age"] < 24) { df_both_F[row,"age_cat_3"] = "21_23" }
}

df_both_F$age_cat_3 <- as.factor(df_both_F$age_cat_3)
summary_3_F <- data.frame(matrix(0, nrow=10, ncol=4))
colnames(summary_3_F) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_3_F$Age_Category <- c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3_F$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_11 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "TD", "age_cat_3"]))
PS_8_11 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "PS", "age_cat_3"]))
TD_12_14 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "TD", "age_cat_3"]))
PS_12_14 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "PS", "age_cat_3"]))
TD_15_17 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "TD", "age_cat_3"]))
PS_15_17 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "PS", "age_cat_3"]))
TD_18_20 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "TD", "age_cat_3"]))
PS_18_20 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "PS", "age_cat_3"]))
TD_21_23 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "TD", "age_cat_3"]))
PS_21_23 <- 100*(length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"])/length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "PS", "age_cat_3"]))
summary_3_F$Percent_Pred_Adult <- c(TD_8_11, PS_8_11, TD_12_14, PS_12_14, TD_15_17, PS_15_17, TD_18_20, PS_18_20, TD_21_23, PS_21_23)

N_TD_8_11 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "TD", "age_cat_3"]))
N_PS_8_11 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "8_11" & df_both_F$diagnosis == "PS", "age_cat_3"]))
N_TD_12_14 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "TD", "age_cat_3"]))
N_PS_12_14 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "12_14" & df_both_F$diagnosis == "PS", "age_cat_3"]))
N_TD_15_17 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "TD", "age_cat_3"]))
N_PS_15_17 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "15_17" & df_both_F$diagnosis == "PS", "age_cat_3"]))
N_TD_18_20 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "TD", "age_cat_3"]))
N_PS_18_20 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "18_20" & df_both_F$diagnosis == "PS", "age_cat_3"]))
N_TD_21_23 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "TD" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "TD", "age_cat_3"]))
N_PS_21_23 <- paste0(length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "PS" & df_both_F$pred_adult == 1, "age_cat_3"]),"/",length(df_both_F[df_both_F$age_cat_3 == "21_23" & df_both_F$diagnosis == "PS", "age_cat_3"]))
summary_3_F$N <- c(N_TD_8_11, N_PS_8_11, N_TD_12_14, N_PS_12_14, N_TD_15_17, N_PS_15_17, N_TD_18_20, N_PS_18_20, N_TD_21_23, N_PS_21_23)


summary_3_F$Age_Category <- factor(summary_3_F$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))


# Plot Info
td_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "TD", "real_age"], df_both_F[df_both_F$diagnosis == "TD", "pred_age"])$p
if (td_p_F < .001) { td_p_F <- .001 } else if (td_p_F < .01) { td_p_F <- .01 } else if (td_p_F < .05) { td_p_F <- .05 } else { td_p_F <- round(td_p_F, digits=3) }
td_N_F <- nrow(df_both_F[df_both_F$diagnosis == "TD",])
ps_corr_F <- round(corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_F <- corr.test(df_both_F[df_both_F$diagnosis == "PS", "real_age"], df_both_F[df_both_F$diagnosis == "PS", "pred_age"])$p
if (ps_p_F < .001) { ps_p_F <- .001 } else if (ps_p_F < .01) { ps_p_F <- .01 } else if (ps_p_F < .05) { ps_p_F <- .05 } else { ps_p_F <- round(ps_p_F, digits=3) }
ps_N_F <- nrow(df_both_F[df_both_F$diagnosis == "PS",])

substring_F <- paste0("TD: r = ", td_corr_F, ", p < ", td_p_F,", N = ", td_N_F, "\nPS: r = ", ps_corr_F, ", p < ", ps_p_F,", N = ", ps_N_F)


# Plots
real_pred_F <- ggplot(data=df_both_F, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Female (El5 Only Both): Gray Matter Volume ROIs", subtitle = substring_F) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

bar_F <- ggplot(data=summary_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female (El5 Only Both): Gray Matter Volume ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")

bar_3_F <- ggplot(data=summary_3_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female (El5 Only Both): Gray Matter Volume ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")



############################ Males ############################

df_M <- df[df$sex == 1, c("bblid", "psTerminal", xvars, yvar)]
df_M$bblid <- as.factor(df_M$bblid)

for (i in 1:ncol(df_M)) {
	na_index <- is.na(df_M[,i])
	df_M <- df_M[!na_index,]
}

td_M <- df_M[df_M$psTerminal == "TD",]
rownames(td_M) <- 1:nrow(td_M)

# TD folds
folds5_TD <- createFolds(td_M$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5_TD[[i]]
	trainfolds <- subset(folds5_TD,!(grepl(i, names(folds5_TD))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_td_train_M_", i), td_M[train, c("bblid", xvars)])
	assign(paste0("x_td_test_M_", i), td_M[test, c("bblid", xvars)])
	assign(paste0("y_td_train_M_", i), td_M[train, c("bblid", yvar)])
	assign(paste0("y_td_test_M_", i), td_M[test, c("bblid", yvar)])
}

# PS folds
ps_M <- df_M[df_M$psTerminal == "PS",]
rownames(ps_M) <- 1:nrow(ps_M)
folds5_PS <- createFolds(ps_M$ageAtScan1, k = 5, list = TRUE)

for (i in 1:5) {
	test <- folds5_PS[[i]]
	trainfolds <- subset(folds5_PS,!(grepl(i, names(folds5_PS))))
	train <- c()
	for (j in 1:4) { train <- c(train, trainfolds[[j]]) }
	assign(paste0("x_ps_train_M_", i), ps_M[train, c("bblid", xvars)])
	assign(paste0("x_ps_test_M_", i), ps_M[test, c("bblid", xvars)])
	assign(paste0("y_ps_train_M_", i), ps_M[train, c("bblid", yvar)])
	assign(paste0("y_ps_test_M_", i), ps_M[test, c("bblid", yvar)])
}


#####Elastic Net: out of sample predictions for TD's, and 1/5 each iteration for PS's
for (i in 1:5) {
	# td train
	x_td_train_df <- get(paste0("x_td_train_M_", i))
	x_td_train_input <- x_td_train_df[,xvars]
	y_td_train_df <- get(paste0("y_td_train_M_", i))
	y_td_train_input <- y_td_train_df$ageAtScan1
	# ps train
	x_ps_train_df <- get(paste0("x_ps_train_M_", i))
	x_ps_train_input <- x_ps_train_df[,xvars]
	y_ps_train_df <- get(paste0("y_ps_train_M_", i))
	y_ps_train_input <- y_ps_train_df$ageAtScan1
	# continued
	x_train_input <- rbind(x_td_train_input, x_ps_train_input)
	ageAtScan1 <- c(y_td_train_input, y_ps_train_input)
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_M_", i), elastic_net_model)
	# td predicted values
	x_td_test_df <- get(paste0("x_td_test_M_", i))
	y_td_test_df <- get(paste0("y_td_test_M_", i))
	y_td_test_predicted <- predict(elastic_net_model, x_td_test_df)
	# ps predicted values
	x_ps_test_df <- get(paste0("x_ps_test_M_", i))
	y_ps_test_df <- get(paste0("y_ps_test_M_", i))
	y_ps_test_predicted <- predict(elastic_net_model, x_ps_test_df)
	if (i == 1) { 
		df_td_M <- merge(cbind(x_td_test_df, y_td_test_predicted), y_td_test_df)
		df_ps_M <- merge(cbind(x_ps_test_df, y_ps_test_predicted), y_ps_test_df)
	} else { 
		df_td_M <- rbind(df_td_M, merge(cbind(x_td_test_df, y_td_test_predicted), y_td_test_df))
		df_ps_M <- rbind(df_ps_M, merge(cbind(x_ps_test_df, y_ps_test_predicted), y_ps_test_df))
	}
}
names(df_td_M)[names(df_td_M) == 'y_td_test_predicted'] <- 'predicted_ageAtScan1'
names(df_ps_M)[names(df_ps_M) == 'y_ps_test_predicted'] <- 'predicted_ageAtScan1'

# Create years age variables
df_ps_M$real_age <- df_ps_M$ageAtScan1/12
df_ps_M$pred_age <- df_ps_M$predicted_ageAtScan1/12
df_td_M$real_age <- df_td_M$ageAtScan1/12
df_td_M$pred_age <- df_td_M$predicted_ageAtScan1/12

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
	else if (df_both_M[row,"real_age"] >= 20 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat"] = "20_23" }
	if (df_both_M[row,"pred_age"] >= 18) { df_both_M[row,"pred_adult"] = 1 }
}

df_both_M$age_cat <- as.factor(df_both_M$age_cat)

# Summary Dataframes
# Elastic Net
summary_M <- data.frame(matrix(0, nrow=14, ncol=4))
colnames(summary_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_M$Age_Category <- c("8_9", "8_9", "10_11", "10_11", "12_13", "12_13", "14_15", "14_15", "16_17", "16_17", "18_19", "18_19", "20_23", "20_23")
summary_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

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
TD_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
PS_20_23 <- 100*(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"])/length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_M$Percent_Pred_Adult <- c(TD_8_9, PS_8_9, TD_10_11, PS_10_11, TD_12_13, PS_12_13, TD_14_15, PS_14_15, TD_16_17, PS_16_17, TD_18_19, PS_18_19, TD_20_23, PS_20_23)

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
N_TD_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "TD", "age_cat"]))
N_PS_20_23 <- paste0(length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat"]),"/",length(df_both_M[df_both_M$age_cat == "20_23" & df_both_M$diagnosis == "PS", "age_cat"]))
summary_M$N <- c(N_TD_8_9, N_PS_8_9, N_TD_10_11, N_PS_10_11, N_TD_12_13, N_PS_12_13, N_TD_14_15, N_PS_14_15, N_TD_16_17, N_PS_16_17, N_TD_18_19, N_PS_18_19, N_TD_20_23, N_PS_20_23)


summary_M$Age_Category <- factor(summary_M$Age_Category, levels = c("8_9", "10_11", "12_13", "14_15", "16_17", "18_19", "20_23"))

# Three-year age bins
df_both_M$age_cat_3 <- 0

for (row in 1:nrow(df_both_M)) {
	if (df_both_M[row,"real_age"] < 12) { df_both_M[row,"age_cat_3"] = "8_11" }
	else if (df_both_M[row,"real_age"] >= 12 & df_both_M[row,"real_age"] < 15) { df_both_M[row,"age_cat_3"] = "12_14" }
	else if (df_both_M[row,"real_age"] >= 15 & df_both_M[row,"real_age"] < 18) { df_both_M[row,"age_cat_3"] = "15_17" }
	else if (df_both_M[row,"real_age"] >= 18 & df_both_M[row,"real_age"] < 21) { df_both_M[row,"age_cat_3"] = "18_20" }
	else if (df_both_M[row,"real_age"] >= 21 & df_both_M[row,"real_age"] < 24) { df_both_M[row,"age_cat_3"] = "21_23" }
}

df_both_M$age_cat_3 <- as.factor(df_both_M$age_cat_3)
summary_3_M <- data.frame(matrix(0, nrow=10, ncol=4))
colnames(summary_3_M) <- c("Age_Category", "Diagnosis", "Percent_Pred_Adult", "N")
summary_3_M$Age_Category <- c("8_11", "8_11", "12_14", "12_14", "15_17", "15_17", "18_20", "18_20", "21_23", "21_23")
summary_3_M$Diagnosis <- c("TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS", "TD", "PS")

TD_8_11 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "TD", "age_cat_3"]))
PS_8_11 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "PS", "age_cat_3"]))
TD_12_14 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "TD", "age_cat_3"]))
PS_12_14 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "PS", "age_cat_3"]))
TD_15_17 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "TD", "age_cat_3"]))
PS_15_17 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "PS", "age_cat_3"]))
TD_18_20 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "TD", "age_cat_3"]))
PS_18_20 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "PS", "age_cat_3"]))
TD_21_23 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "TD", "age_cat_3"]))
PS_21_23 <- 100*(length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"])/length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "PS", "age_cat_3"]))
summary_3_M$Percent_Pred_Adult <- c(TD_8_11, PS_8_11, TD_12_14, PS_12_14, TD_15_17, PS_15_17, TD_18_20, PS_18_20, TD_21_23, PS_21_23)

N_TD_8_11 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "TD", "age_cat_3"]))
N_PS_8_11 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "8_11" & df_both_M$diagnosis == "PS", "age_cat_3"]))
N_TD_12_14 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "TD", "age_cat_3"]))
N_PS_12_14 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "12_14" & df_both_M$diagnosis == "PS", "age_cat_3"]))
N_TD_15_17 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "TD", "age_cat_3"]))
N_PS_15_17 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "15_17" & df_both_M$diagnosis == "PS", "age_cat_3"]))
N_TD_18_20 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "TD", "age_cat_3"]))
N_PS_18_20 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "18_20" & df_both_M$diagnosis == "PS", "age_cat_3"]))
N_TD_21_23 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "TD" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "TD", "age_cat_3"]))
N_PS_21_23 <- paste0(length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "PS" & df_both_M$pred_adult == 1, "age_cat_3"]),"/",length(df_both_M[df_both_M$age_cat_3 == "21_23" & df_both_M$diagnosis == "PS", "age_cat_3"]))
summary_3_M$N <- c(N_TD_8_11, N_PS_8_11, N_TD_12_14, N_PS_12_14, N_TD_15_17, N_PS_15_17, N_TD_18_20, N_PS_18_20, N_TD_21_23, N_PS_21_23)


summary_3_M$Age_Category <- factor(summary_3_M$Age_Category, levels = c("8_11", "12_14", "15_17", "18_20", "21_23"))

# Plot Info
td_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age"])$r, digits=3)
td_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "TD", "real_age"], df_both_M[df_both_M$diagnosis == "TD", "pred_age"])$p
if (td_p_M < .001) { td_p_M <- .001 } else if (td_p_M < .01) { td_p_M <- .01 } else if (td_p_M < .05) { td_p_M <- .05 } else { td_p_M <- round(td_p_M, digits=3) }
td_N_M <- nrow(df_both_M[df_both_M$diagnosis == "TD",])
ps_corr_M <- round(corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age"])$r, digits=3)
ps_p_M <- corr.test(df_both_M[df_both_M$diagnosis == "PS", "real_age"], df_both_M[df_both_M$diagnosis == "PS", "pred_age"])$p
if (ps_p_M < .001) { ps_p_M <- .001 } else if (ps_p_M < .01) { ps_p_M <- .01 } else if (ps_p_M < .05) { ps_p_M <- .05 } else { ps_p_M <- round(ps_p_M, digits=3) }
ps_N_M <- nrow(df_both_M[df_both_M$diagnosis == "PS",])

substring_M <- paste0("TD: r = ", td_corr_M, ", p < ", td_p_M,", N = ", td_N_M, "\nPS: r = ", ps_corr_M, ", p < ", ps_p_M,", N = ", ps_N_M)


# Plots
real_pred_M <- ggplot(data=df_both_M, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Male (El5 Only Both): Gray Matter Volume ROIs", subtitle = substring_M) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

bar_M <- ggplot(data=summary_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male (El5 Only Both): Gray Matter Volume ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")

bar_3_M <- ggplot(data=summary_3_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male (El5 Only Both): Gray Matter Volume ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")


# Save plots
pdf(file="/home/butellyn/age_prediction/plots/gmv_fiveFoldBoth_ElasticNet.pdf", width=12, height=6)
grid.arrange(real_pred_F, real_pred_M, ncol=2)
grid.arrange(bar_F, bar_M, ncol=2)
grid.arrange(bar_3_F, bar_3_M, ncol=2)
dev.off()















