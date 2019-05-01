### This script implements the calculation of delta 2 from "Estimation of Brain Age Delta from Brain Imaging"
###
### Ellyn Butler
### April 3, 2019

deltaTrans <- function(dataframe, xcols, ycol) {
	library('far')
	rownames(dataframe) <- 1:nrow(dataframe)
	# 1. Your vector of ages is Y (subjects x 1)
	y <- dataframe[,ycol]
	# 2. Your matrix of brain imaging measures is X (subjects x features)
	x <- dataframe[,xcols]
	# 3. Subtract the means from Y and all columns in X
	y <- y - mean(y)
	for (i in 1:nrow(x)) {
		x[,i] <- x[,i] - mean(x[,i])
	}
	# 4. Use SVD to replace X with its top 10-25% vertical eigenvectors (Screw this)
	# 5. Compute Y2, demean it and orthogonalise it with respect to Y to give Yo2
	ysq <- y^2 - mean(y^2)
	ysq2 <- lm(ysq~y)
	
	# 6. Create matrix Y2 = [Y Yo2]
	
	# 7. The initial model is YB1 = XB1 + d1
	# a) Compute initial age prediction B1 = X
	# b)
	# 8.
	# a)
	# b)
	
}
