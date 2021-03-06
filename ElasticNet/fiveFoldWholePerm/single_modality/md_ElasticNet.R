### This script produces linear models, features for which have been selected with a grid search of
### alphas and lambdas for elastic net, for cortical thickness predicting age in two same-sex samples who
### did not have a mental illness (TD) at their final visit. All predictions are out-of-sample, such that
### TD's obtain their predicted age from a model built on the other 4/5's of TD's, and people with psychosis-
### spectrum symptoms (PS's) obtain their predicted age from a model built on all of the TD's.
###
### Ellyn Butler
### March 4, 2019 - present


# Load libraries
library('glmnet')
library('ggplot2')
library('reshape2')
library('gridExtra')
library('psych')
library('dplyr')
library('caret')
library('permute')
library('grid')

# Source my functions
source("/home/butellyn/ButlerPlotFuncs/plotFuncs_AdonPath.R")

# Load data #February 25: Change clinical variable to psTerminal
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

mdvars <- c('dti_jlf_tr_R_Accumbens_Area','dti_jlf_tr_L_Accumbens_Area', 'dti_jlf_tr_R_Amygdala', 'dti_jlf_tr_L_Amygdala', 'dti_jlf_tr_R_Caudate', 'dti_jlf_tr_L_Caudate', 'dti_jlf_tr_R_Cerebellum_Exterior', 'dti_jlf_tr_L_Cerebellum_Exterior', 'dti_jlf_tr_R_Hippocampus', 'dti_jlf_tr_L_Hippocampus', 'dti_jlf_tr_R_Pallidum', 'dti_jlf_tr_L_Pallidum', 'dti_jlf_tr_R_Putamen', 'dti_jlf_tr_L_Putamen', 'dti_jlf_tr_R_Thalamus_Proper', 'dti_jlf_tr_L_Thalamus_Proper', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_I.V', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VI.VII', 'dti_jlf_tr_Cerebellar_Vermal_Lobules_VIII.X', 'dti_jlf_tr_R_ACgG', 'dti_jlf_tr_L_ACgG', 'dti_jlf_tr_R_AIns', 'dti_jlf_tr_L_AIns', 'dti_jlf_tr_R_AOrG', 'dti_jlf_tr_L_AOrG', 'dti_jlf_tr_R_AnG', 'dti_jlf_tr_L_AnG', 'dti_jlf_tr_R_Calc', 'dti_jlf_tr_L_Calc', 'dti_jlf_tr_R_CO', 'dti_jlf_tr_L_CO', 'dti_jlf_tr_R_Cun', 'dti_jlf_tr_L_Cun', 'dti_jlf_tr_R_Ent', 'dti_jlf_tr_L_Ent', 'dti_jlf_tr_R_FO', 'dti_jlf_tr_L_FO', 'dti_jlf_tr_R_FRP', 'dti_jlf_tr_L_FRP', 'dti_jlf_tr_R_FuG', 'dti_jlf_tr_L_FuG', 'dti_jlf_tr_R_GRe', 'dti_jlf_tr_L_GRe', 'dti_jlf_tr_R_IOG', 'dti_jlf_tr_L_IOG', 'dti_jlf_tr_R_ITG', 'dti_jlf_tr_L_ITG', 'dti_jlf_tr_R_LiG', 'dti_jlf_tr_L_LiG', 'dti_jlf_tr_R_LOrG', 'dti_jlf_tr_L_LOrG', 'dti_jlf_tr_R_MCgG', 'dti_jlf_tr_L_MCgG', 'dti_jlf_tr_R_MFC', 'dti_jlf_tr_L_MFC', 'dti_jlf_tr_R_MFG', 'dti_jlf_tr_L_MFG', 'dti_jlf_tr_R_MOG', 'dti_jlf_tr_L_MOG', 'dti_jlf_tr_R_MOrG', 'dti_jlf_tr_L_MOrG', 'dti_jlf_tr_R_MPoG', 'dti_jlf_tr_L_MPoG', 'dti_jlf_tr_R_MPrG', 'dti_jlf_tr_L_MPrG', 'dti_jlf_tr_R_MSFG', 'dti_jlf_tr_L_MSFG', 'dti_jlf_tr_R_MTG', 'dti_jlf_tr_L_MTG', 'dti_jlf_tr_R_OCP', 'dti_jlf_tr_L_OCP', 'dti_jlf_tr_R_OFuG', 'dti_jlf_tr_L_OFuG', 'dti_jlf_tr_R_OpIFG', 'dti_jlf_tr_L_OpIFG', 'dti_jlf_tr_R_OrIFG', 'dti_jlf_tr_L_OrIFG', 'dti_jlf_tr_R_PCgG', 'dti_jlf_tr_L_PCgG', 'dti_jlf_tr_R_PCu', 'dti_jlf_tr_L_PCu', 'dti_jlf_tr_R_PHG', 'dti_jlf_tr_L_PHG', 'dti_jlf_tr_R_PIns', 'dti_jlf_tr_L_PIns', 'dti_jlf_tr_R_PO', 'dti_jlf_tr_L_PO', 'dti_jlf_tr_R_PoG', 'dti_jlf_tr_L_PoG', 'dti_jlf_tr_R_POrG', 'dti_jlf_tr_L_POrG', 'dti_jlf_tr_R_PP', 'dti_jlf_tr_L_PP', 'dti_jlf_tr_R_PrG', 'dti_jlf_tr_L_PrG', 'dti_jlf_tr_R_PT', 'dti_jlf_tr_L_PT', 'dti_jlf_tr_R_SCA', 'dti_jlf_tr_L_SCA', 'dti_jlf_tr_R_SFG', 'dti_jlf_tr_L_SFG', 'dti_jlf_tr_R_SMC', 'dti_jlf_tr_L_SMC', 'dti_jlf_tr_R_SMG', 'dti_jlf_tr_L_SMG', 'dti_jlf_tr_R_SOG', 'dti_jlf_tr_L_SOG', 'dti_jlf_tr_R_SPL', 'dti_jlf_tr_L_SPL', 'dti_jlf_tr_R_STG', 'dti_jlf_tr_L_STG', 'dti_jlf_tr_R_TMP', 'dti_jlf_tr_L_TMP', 'dti_jlf_tr_R_TrIFG', 'dti_jlf_tr_L_TrIFG')


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

xvars <- mdvars

############################ Females ############################

df_F <- df[df$sex == 2, c("bblid", "psTerminal", xvars, yvar)]
df_F$bblid <- as.factor(df_F$bblid)

for (i in 1:ncol(df_F)) {
	na_index <- is.na(df_F[,i])
	df_F <- df_F[!na_index,]
}

df_F <- df_F[df_F$psTerminal == "TD" | df_F$psTerminal == "PS",]
rownames(df_F) <- 1:nrow(df_F)

##### PERMUTE LABELS
perm <- shuffle(nrow(df_F))
df_F_tmp <- df_F
for (i in 1:length(perm)) {
	row_new_diag <- perm[[i]]
	new_diag <- as.character(df_F_tmp[row_new_diag, "psTerminal"])
	df_F[i, "psFake"] <- new_diag
}

##### confusion matrix
df_F$psFake <- factor(df_F$psFake)
df_F$psTerminal <- factor(df_F$psTerminal)
confusion_F <- as.data.frame(matrix(0, nrow=2, ncol=2))
colnames(confusion_F) <- c("Real TD", "Real PS")
rownames(confusion_F) <- c("Fake TD", "Fake PS")
confusion_F[1,1] <- nrow(df_F[df_F$psFake == "TD" & df_F$psTerminal == "TD",])
confusion_F[1,2] <- nrow(df_F[df_F$psFake == "TD" & df_F$psTerminal == "PS",])
confusion_F[2,1] <- nrow(df_F[df_F$psFake == "PS" & df_F$psTerminal == "TD",])
confusion_F[2,2] <- nrow(df_F[df_F$psFake == "PS" & df_F$psTerminal == "PS",])

td_F <- df_F[df_F$psFake == "TD",]
rownames(td_F) <- 1:nrow(td_F)

##### Elastic Net

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
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_td_F_", i), elastic_net_model)
	x_test_df <- get(paste0("x_td_test_F_", i))
	y_test_df <- get(paste0("y_td_test_F_", i))
	y_test_predicted <- predict(elastic_net_model, x_test_df)
	if (i == 1) { predicted_values <- merge(cbind(x_test_df, y_test_predicted), y_test_df)
	} else { predicted_values <- rbind(predicted_values, merge(cbind(x_test_df, y_test_predicted), y_test_df))
	}
}
names(predicted_values)[names(predicted_values) == 'y_test_predicted'] <- 'predicted_ageAtScan1'

#####Elastic Net: predictions for PS's 
# train on full TD set
x_td_train_input <- td_F[,xvars]
ageAtScan1 <- td_F[,yvar]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 

# predict on PS's
x_ps_F <- df_F[df_F$psFake == "PS", c("bblid", xvars)]
y_ps_F <- df_F[df_F$psFake == "PS", c("bblid", yvar)]
x_ps_input <- x_ps_F[,xvars]
y_ps_input <- y_ps_F[,yvar]
ageAtScan1 <- y_ps_input
y_ps_predicted <- predict(elastic_net_model, x_ps_input)

# create final dfs
df_td_F <- predicted_values
df_ps_F <- merge(x_ps_F, y_ps_F)
df_ps_F$predicted_ageAtScan1 <- y_ps_predicted

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
	labs(title = "Female (El5 Fake): Mean Diffusivity ROIs", subtitle = substring_F) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

bar_F <- ggplot(data=summary_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Female (El5 Fake): Mean Diffusivity ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")





############################ Males ############################

df_M <- df[df$sex == 1, c("bblid", "psTerminal", xvars, yvar)]
df_M$bblid <- as.factor(df_M$bblid)

for (i in 1:ncol(df_M)) {
	na_index <- is.na(df_M[,i])
	df_M <- df_M[!na_index,]
}

df_M <- df_M[df_M$psTerminal == "TD" | df_M$psTerminal == "PS",]
rownames(df_M) <- 1:nrow(df_M)

##### PERMUTE LABELS
perm <- shuffle(nrow(df_M))
df_M_tmp <- df_M
for (i in 1:length(perm)) {
	row_new_diag <- perm[[i]]
	new_diag <- as.character(df_M_tmp[row_new_diag, "psTerminal"])
	df_M[i, "psFake"] <- new_diag
}

##### confusion matrix
df_M$psFake <- factor(df_M$psFake)
df_M$psTerminal <- factor(df_M$psTerminal)
confusion_M <- as.data.frame(matrix(0, nrow=2, ncol=2))
colnames(confusion_M) <- c("Real TD", "Real PS")
rownames(confusion_M) <- c("Fake TD", "Fake PS")
confusion_M[1,1] <- nrow(df_M[df_M$psFake == "TD" & df_M$psTerminal == "TD",])
confusion_M[1,2] <- nrow(df_M[df_M$psFake == "TD" & df_M$psTerminal == "PS",])
confusion_M[2,1] <- nrow(df_M[df_M$psFake == "PS" & df_M$psTerminal == "TD",])
confusion_M[2,2] <- nrow(df_M[df_M$psFake == "PS" & df_M$psTerminal == "PS",])

td_M <- df_M[df_M$psFake == "TD",]
rownames(td_M) <- 1:nrow(td_M)

##### Elastic Net

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
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_td_M_", i), elastic_net_model)
	x_test_df <- get(paste0("x_td_test_M_", i))
	y_test_df <- get(paste0("y_td_test_M_", i))
	y_test_predicted <- predict(elastic_net_model, x_test_df)
	if (i == 1) { predicted_values <- merge(cbind(x_test_df, y_test_predicted), y_test_df)
	} else { predicted_values <- rbind(predicted_values, merge(cbind(x_test_df, y_test_predicted), y_test_df))
	}
}
names(predicted_values)[names(predicted_values) == 'y_test_predicted'] <- 'predicted_ageAtScan1'

#####Elastic Net: predictions for PS's 
# train on full TD set
x_td_train_input <- td_M[,xvars]
ageAtScan1 <- td_M[,yvar]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 

# predict on PS's
x_ps_M <- df_M[df_M$psFake == "PS", c("bblid", xvars)]
y_ps_M <- df_M[df_M$psFake == "PS", c("bblid", yvar)]
x_ps_input <- x_ps_M[,xvars]
y_ps_input <- y_ps_M[,yvar]
ageAtScan1 <- y_ps_input
y_ps_predicted <- predict(elastic_net_model, x_ps_input)

# create final dfs
df_td_M <- predicted_values
df_ps_M <- merge(x_ps_M, y_ps_M)
df_ps_M$predicted_ageAtScan1 <- y_ps_predicted

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
	labs(title = "Male (El5 Fake): Mean Diffusivity ROIs", subtitle = substring_M) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
	theme(plot.title = element_text(family="Times", face="bold", size=18)) 

bar_M <- ggplot(data=summary_M, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
	geom_bar(stat="identity", position=position_dodge()) +
	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
	ggtitle("Male (El5 Fake): Mean Diffusivity ROIs") + xlab("Age Category") + ylab("Predicted Adult (%)")


# Save plots
g_F <- tableGrob(confusion_F, theme=ttheme_minimal(base_size=20))
g_M <- tableGrob(confusion_M, theme=ttheme_minimal(base_size=20))

pdf(file="/home/butellyn/age_prediction/plots/md_fiveFoldWholePerm_ElasticNet.pdf", width=12, height=6)
grid.arrange(real_pred_F, real_pred_M, ncol=2)
grid.arrange(bar_F, bar_M, ncol=2)
grid.arrange(g_F, g_M, ncol=2, top = textGrob("Female and Male Confusion Matrices",gp=gpar(fontsize=30,font=3)))
dev.off()















