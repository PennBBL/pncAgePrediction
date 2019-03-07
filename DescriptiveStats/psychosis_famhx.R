### This script identifies how many people from GO1 with a family history of psychosis have imaging data
### by sex and their own terminal diagnosis
### 
### Ellyn Butler
### February 27, 2019

# Read in the data
famhx_df <- read.csv("/home/butellyn/age_prediction/data/Psychosis_FamHx_2-27-2019.csv")
neurodiag_df <- read.csv("/home/butellyn/age_prediction/data/n1601_imagingclinicalcognitive_20190130.csv")

df <- merge(famhx_df, neurodiag_df, by="bblid")
df <- df[,c("bblid", "sex", "Fam_NoPsychosis", "goassessDxpmr7", "mprage_jlf_vol_R_Hippocampus")]

for (i in 1:ncol(df)) {
	na_index <- is.na(df[,i])
	df <- df[!na_index,]
}

N_famhx <- nrow(df[df$Fam_NoPsychosis == 0,])

N_F_famhx <- nrow(df[df$sex == 2 & df$Fam_NoPsychosis == 0,])
N_F_famhx_ps <- nrow(df[df$sex == 2 & df$Fam_NoPsychosis == 0 & df$goassessDxpmr7 == "PS",])
N_F_famhx_td <- nrow(df[df$sex == 2 & df$Fam_NoPsychosis == 0 & df$goassessDxpmr7 == "TD",])


N_M_famhx <- nrow(df[df$sex == 1 & df$Fam_NoPsychosis == 0,])
N_M_famhx_ps <- nrow(df[df$sex == 1 & df$Fam_NoPsychosis == 0 & df$goassessDxpmr7 == "PS",])
N_M_famhx_td <- nrow(df[df$sex == 1 & df$Fam_NoPsychosis == 0 & df$goassessDxpmr7 == "TD",])
