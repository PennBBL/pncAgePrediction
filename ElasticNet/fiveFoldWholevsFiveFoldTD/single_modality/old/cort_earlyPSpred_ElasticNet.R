### This script produces linear models, features for which have been selected with a grid search of
### alphas and lambdas for elastic net, for cortical thickness predicting age in two same-sex samples who
### did not have a mental illness (TD) at their final visit. All predictions are out-of-sample, such that
### TD's obtain their predicted age from a model built on the other 4/5's of TD's, and people with psychosis-
### spectrum symptoms (PS's) obtain their predicted age from a model built on all of the TD's.
###
### Ellyn Butler
### March 11, 2019 - present


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

# Load data #February 25: Change clinical variable to psTerminal
df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv", header=T)

cortvars <- c('mprage_jlf_ct_R_ACgG', 'mprage_jlf_ct_L_ACgG', 'mprage_jlf_ct_R_AIns', 'mprage_jlf_ct_L_AIns', 
            'mprage_jlf_ct_R_AOrG', 'mprage_jlf_ct_L_AOrG', 'mprage_jlf_ct_R_AnG', 'mprage_jlf_ct_L_AnG', 
            'mprage_jlf_ct_R_Calc', 'mprage_jlf_ct_L_Calc', 'mprage_jlf_ct_R_CO', 'mprage_jlf_ct_L_CO', 
            'mprage_jlf_ct_R_Cun', 'mprage_jlf_ct_L_Cun', 'mprage_jlf_ct_R_Ent', 'mprage_jlf_ct_L_Ent', 
            'mprage_jlf_ct_R_FO', 'mprage_jlf_ct_L_FO', 'mprage_jlf_ct_R_FRP', 'mprage_jlf_ct_L_FRP', 
            'mprage_jlf_ct_R_FuG', 'mprage_jlf_ct_L_FuG', 'mprage_jlf_ct_R_GRe', 'mprage_jlf_ct_L_GRe', 
            'mprage_jlf_ct_R_IOG', 'mprage_jlf_ct_L_IOG', 'mprage_jlf_ct_R_ITG', 'mprage_jlf_ct_L_ITG', 
            'mprage_jlf_ct_R_LiG', 'mprage_jlf_ct_L_LiG', 'mprage_jlf_ct_R_LOrG', 'mprage_jlf_ct_L_LOrG', 
            'mprage_jlf_ct_R_MCgG', 'mprage_jlf_ct_L_MCgG', 'mprage_jlf_ct_R_MFC', 'mprage_jlf_ct_L_MFC', 
            'mprage_jlf_ct_R_MFG', 'mprage_jlf_ct_L_MFG', 'mprage_jlf_ct_R_MOG', 'mprage_jlf_ct_L_MOG', 
            'mprage_jlf_ct_R_MOrG', 'mprage_jlf_ct_L_MOrG', 'mprage_jlf_ct_R_MPoG', 'mprage_jlf_ct_L_MPoG', 
            'mprage_jlf_ct_R_MPrG', 'mprage_jlf_ct_L_MPrG', 'mprage_jlf_ct_R_MSFG', 'mprage_jlf_ct_L_MSFG', 
            'mprage_jlf_ct_R_MTG', 'mprage_jlf_ct_L_MTG', 'mprage_jlf_ct_R_OCP', 'mprage_jlf_ct_L_OCP', 
            'mprage_jlf_ct_R_OFuG', 'mprage_jlf_ct_L_OFuG', 'mprage_jlf_ct_R_OpIFG', 'mprage_jlf_ct_L_OpIFG', 
            'mprage_jlf_ct_R_OrIFG', 'mprage_jlf_ct_L_OrIFG', 'mprage_jlf_ct_R_PCgG', 'mprage_jlf_ct_L_PCgG', 
            'mprage_jlf_ct_R_PCu', 'mprage_jlf_ct_L_PCu', 'mprage_jlf_ct_R_PHG', 'mprage_jlf_ct_L_PHG', 
            'mprage_jlf_ct_R_PIns', 'mprage_jlf_ct_L_PIns', 'mprage_jlf_ct_R_PO', 'mprage_jlf_ct_L_PO', 
            'mprage_jlf_ct_R_PoG', 'mprage_jlf_ct_L_PoG', 'mprage_jlf_ct_R_POrG', 'mprage_jlf_ct_L_POrG', 
            'mprage_jlf_ct_R_PP', 'mprage_jlf_ct_L_PP', 'mprage_jlf_ct_R_PrG', 'mprage_jlf_ct_L_PrG', 
            'mprage_jlf_ct_R_PT', 'mprage_jlf_ct_L_PT', 'mprage_jlf_ct_R_SCA', 'mprage_jlf_ct_L_SCA', 
            'mprage_jlf_ct_R_SFG', 'mprage_jlf_ct_L_SFG', 'mprage_jlf_ct_R_SMC', 'mprage_jlf_ct_L_SMC', 
            'mprage_jlf_ct_R_SMG', 'mprage_jlf_ct_L_SMG', 'mprage_jlf_ct_R_SOG', 'mprage_jlf_ct_L_SOG', 
            'mprage_jlf_ct_R_SPL', 'mprage_jlf_ct_L_SPL', 'mprage_jlf_ct_R_STG', 'mprage_jlf_ct_L_STG', 
            'mprage_jlf_ct_R_TMP')


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

xvars <- cortvars

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


#####Elastic Net: out of sample predictions for TD's
for (i in 1:5) {
	x_td_train_df <- get(paste0("x_td_train_F_", i))
	x_td_train_input <- x_td_train_df[,xvars]
	y_td_train_df <- get(paste0("y_td_train_F_", i))
	y_td_train_input <- y_td_train_df$ageAtScan1
	ageAtScan1 <- y_td_train_input
	train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
	elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
				# just used parameters from here: https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
	assign(paste0("mod_td_F_", i), elastic_net_model)
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
		coefs <- coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)# Added March 11, 2019
	} else { 
		df_td_F <- rbind(df_td_F, merge(cbind(x_td_test_df, y_td_test_predicted), y_td_test_df))
		df_ps_F <- rbind(df_ps_F, merge(cbind(x_ps_test_df, y_ps_test_predicted), y_ps_test_df))
		coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda))# Added March 11, 2019
	}
}
names(df_td_F)[names(df_td_F) == 'y_td_test_predicted'] <- 'predicted_ageAtScan1'
names(df_ps_F)[names(df_ps_F) == 'y_ps_test_predicted'] <- 'predicted_ageAtScan1'

#####Elastic Net: predictions for PS's 
# train on full TD set
x_td_train_input <- td_F[,xvars]
ageAtScan1 <- td_F[,yvar]
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
elastic_net_model <- train(ageAtScan1 ~ . , data = cbind(ageAtScan1, x_td_train_input), method = "glmnet", 
				preProcess = c("center", "scale"), tuneLength = 25, trControl = train_control) 
coefs <- cbind(coefs, coef(elastic_net_model$finalModel, s = elastic_net_model$bestTune$lambda)) # Added March 11, 2019
coefs <- as.matrix(coefs) # Added March 11, 2019
coefs_F <- data.frame(coefs) # Added March 11, 2019
colnames(coefs_F) <- c("First_Fold", "Second_Fold", "Third_Fold", "Fourth_Fold", "Fifth_Fold", "Whole_Sample") # Added March 11, 2019

# predict on PS's
#x_ps_F <- df_F[df_F$psTerminal == "PS", c("bblid", xvars)]
#y_ps_F <- df_F[df_F$psTerminal == "PS", c("bblid", yvar)]
#x_ps_input <- x_ps_F[,xvars]
#y_ps_input <- y_ps_F[,yvar]
#ageAtScan1 <- y_ps_input
#y_ps_predicted <- predict(elastic_net_model, x_ps_input)

# create final dfs
#df_td_F <- predicted_values
#df_ps_F <- merge(x_ps_F, y_ps_F)
#df_ps_F$predicted_ageAtScan1 <- y_ps_predicted

# get predictions for PSs based on five models for TDs # Added March 11, 2019
#for (i in 1:5) {
#	elastic_net_model <- get(paste0("mod_td_F_", i))
#	df_ps_F[,paste0("fold_", i, "_ageAtScan1")] <- predict(elastic_net_model, x_ps_input)
#}

# Create years age variables
df_ps_F$real_age <- df_ps_F$ageAtScan1/12
df_ps_F$pred_age <- df_ps_F$predicted_ageAtScan1/12
df_td_F$real_age <- df_td_F$ageAtScan1/12
df_td_F$pred_age <- df_td_F$predicted_ageAtScan1/12

# Plot Percent Predicted to be over 18 in each two-year age bins, split by diagnosis
df_ps_F$diagnosis <- "PS"
df_td_F$diagnosis <- "TD"
df_both_F <- rbind(df_ps_F[,!(names(df_ps_F) %in% c("fold_1_ageAtScan1", "fold_2_ageAtScan1", "fold_3_ageAtScan1", "fold_4_ageAtScan1", "fold_5_ageAtScan1"))], df_td_F)
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

#bar_fiveFoldWhole_F <- ggplot(data=summary_F, aes(x=Age_Category, y=Percent_Pred_Adult, fill=Diagnosis)) +
#	geom_bar(stat="identity", position=position_dodge()) +
#	geom_text(aes(label=N), vjust=1.6, color="white", position = position_dodge(0.9), size=2.5) +
#	scale_fill_brewer(palette="Paired") + ylim(c(0,100)) +
#	labs(title = "Age (TDs from Five; PSs from Whole)", subtitle = substring_F) +
#	theme(plot.title = element_text(family="Times", face="bold", size=18)) +
#	xlab("Age Category") + ylab("Predicted Adult (%)")

real_pred_fiveFold_F <- ggplot(data=df_both_F2, aes(real_age, pred_age, color=diagnosis)) +
	geom_point(shape = 16, size = 2, show.legend = TRUE, alpha=.4) + theme_minimal() +
	xlim(0,25) + ylim(0,25) + geom_abline() +
	labs(title = "Age (TDs and PSs from Five)", subtitle = substring_F2) + 
	xlab("Real Age") + ylab("Predicted Age") + geom_smooth(se = TRUE, method = "lm") +
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







# Save plots
pdf(file="/home/butellyn/age_prediction/plots/cort_F_earlyPSpred_fiveFoldWholeTDvsFiveFoldTD_ElasticNet.pdf", width=12, height=6)
plot(0:10, type = "n", xaxt="n", yaxt="n", bty="n", xlab = "", ylab = "")
text(5, 8, "Predicted Age for Females Using Cortical Thickness ROIs in an Elastic Net Model")
grid.arrange(real_pred_fiveFoldWhole_F, real_pred_fiveFold_F, ncol=2)
grid.arrange(predage_RvsW_F, predage_Rvs1_F, ncol=2)
grid.arrange(predage_Rvs2_F, predage_Rvs3_F, ncol=2)
grid.arrange(predage_Rvs4_F, predage_Rvs5_F, ncol=2)
grid.arrange(predage_1vs5_F, predage_Wvs1_F, ncol=2)
grid.arrange(predage_Wvs2_F, predage_Wvs3_F, ncol=2)
grid.arrange(predage_Wvs4_F, predage_Wvs5_F, ncol=2)
grid.arrange(betas_1vs5_F, betas_Wvs1_F, ncol=2)
grid.arrange(betas_Wvs2_F, betas_Wvs3_F, ncol=2)
grid.arrange(betas_Wvs4_F, betas_Wvs5_F, ncol=2)
dev.off()














