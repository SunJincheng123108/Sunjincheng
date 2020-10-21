# Sunjincheng
classification
---
title: "Assignment for ASML2 Epiphany 2020"
author: "Sun Jincheng"
output:
  word_document: default
  html_document: 
    df_print: paged
  latex_engine: xelatex
  html_notebook: 
    df_print: paged
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(results = "hide",echo=FALSE,fig.width = 4,fig.height = 3.5)
```

# General Instructions

Please go through the R notebook below, and carry out the requested tasks. You will provide all your answers directly into this .Rmd file. Add code into the R chunks where requested. You can create new chunks where required. Where text answers are requested, please add them directly into this document, typically below the R chunks, using R Markdown syntax as adequate. 

At the end, you will submit both your worked .Rmd file, and a `knitted' PDF version, through DUO.

**Important**: Please ensure carefully whether all chunks compile, and also check in the knitted PDF whether all R chunks did *actually* compile, and all images that you would like to produce have *actually* been generated.  **An R chunk which does not compile will give zero marks, and a picture which does not exist will give zero marks, even if some parts of the required code are correct.**

**Note**: It is appreciated that some of the requested analyses requires running R code which is not deterministic. So, you will not have full control over the output that is finally generated in the knitted document. This is fine. It is clear that the methods under investigation carry uncertainty, which is actually part of the problem tackled in this assignment. Your analysis should, however, be robust enough so that it stays in essence correct under repeated execution of your data analysis.  

# Reading in data

We consider data from an industrial melter system. The melter is part of a disposal procedure, where a powder (waste material) is clad in glass. The melter vessel is
continuously filled with powder, and raw glass is discretely introduced in the form of glass frit. This binary composition is heated by  induction coils, positioned around the melter vessel. Resulting from this heating procedure, the glass becomes
molten homogeneously  [(Liu et al, 2008)](https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/aic.11526).

Measurements of 15 temperature sensors `temp1`, ..., `temp15` (in $^{\circ} C$), the power in four
induction coils `ind1`,...,  `ind4`,  the `voltage`, and the `viscosity` of the molten glass, were taken every 5 minutes. The sample size available for our analysis is $n=900$. 

We use the following R chunk to read the data in

```{r}
melter<-read.table("http://maths.dur.ac.uk/~dma0je/Data/melter.dat", header=TRUE)

```

If this has gone right, then the following code
```{r}
is.data.frame(melter)
dim(melter)
```

should tell you that `melter` is a data frame of dimension $900 \times 21$. Otherwise something has gone wrong, and you need to start again.

To get more familiar with the data frame, please also execute

```{r}
head(melter)
colnames(melter)
boxplot(melter)
```


# Task 1: Principal component analysis (10 marks) 
We consider initially only the 15 variables representing the temperature sensors. Please create a new data frame, called `Temp`, which contains only these 15 variables. Then carry out a principal component analysis. (Use your judgement on whether, for this purpose,  the temperature variables require scaling or not). Produce a screeplot, and also answer the following questions: How many principal components are needed to capture 90% of the total variation? How many are needed to capture 98%?

**Answer:**

```{r}
# ---
temp=melter[,7:21]
temp.pr <- prcomp(temp)
temp.pr$sdev
Lambda<- temp.pr$sdev^2
round(Lambda, digits=6) 
round(Lambda/sum(Lambda), digits=3)
summary(temp.pr)
plot(temp.pr)
```

In the first question, the square roots of the eigenvalues are Proportion of Variance. Then normalize this, get: 0.640 0.166 0.071 0.058 0.019 0.011 0.009 0.008 0.005 0.004 0.004 0.003 0.001 0.001 0.001
From this result, I think there may be too many temperature variables, for some of the λ occupy less than 0.01. See from the Cumulative Proportion, I know I need 4 principal components to capture 90%, and 8 principal components to capture 98%.

# Task 2: Multiple linear regression (20 marks)

We consider from now on, and for the remainder of this assignment, `viscosity` as the response variable. 

Fit a linear regression model, with `viscosity` as response variable, and all other variables as predictors, and  produce the `summary` output of the fitted model. In this task, we are mainly interested in the standard errors of the estimated coefficients. Create a vector, with name `melter.fit.sd`, which contains the standard errors of all estimated coefficients, except the intercept. (So, this vector should have length 20). Then produce a `barplot` of these standard errors (where the height of each bar indicates the value of the standard error of the respective coefficients). Please use blue color to fill the bars of the barplot.

**Answer:**

```{r}
#---
melter.fit<-lm(melter$viscosity~melter$voltage+melter$ind1+melter$ind2+melter$ind3+melter$ind4+melter$temp1+melter$temp2+melter$temp3+melter$temp4+melter$temp5+melter$temp6+melter$temp7+melter$temp8+melter$temp9+melter$temp10+melter$temp11+melter$temp12+melter$temp13+melter$temp14+melter$temp15)
summary(melter.fit)
melter.fit.sd<-c(0.1187182,2.0218666,2.3166066,4.0011192,2.5631995,0.8122402,1.1884594,1.8228624,0.5293583,0.3952806,0.5776417,0.8423071,1.4509358,0.6492039,0.7184004,0.4189843,0.6999098,0.6261754,1.2645329,1.3274590)
barplot(melter.fit.sd,col="blue")
```

(1). In this question, I first consider to use Least squares regression. Then get the standard errors of the estimated coefficients:
melter.fit.sd<-c(0.1187182,2.0218666,2.3166066,4.0011192,2.5631995,0.8122402,1.1884594,1.8228624,0.5293583,0.3952806,0.5776417,0.8423071,1.4509358,0.6492039,0.7184004,0.4189843,0.6999098,0.6261754,1.2645329,1.3274590). Then I can get the plot.

Now repeat this analysis, but this time using a Bayesian linear regression. Use adequate methodology to fit the Bayesian version of the linear model considered above.  It is your choice whether you would like to employ ready-to-use R functions which carry out this task for you, or whether you would like to implement this procedure from scratch, for instance using `jags`. 

In either case, you need to be able to extract posterior draws of the estimated parameters from the fitted object, and compute their standard deviations. Please save these standard deviations, again excluding that one for the intercept, into a vector `melter.bayes.sd`.  Produce now a barplot which displays both of `melter.fit.sd` and `melter.bayes.sd` in one plot, and allows a direct comparison  of the frequentist and Bayesian standard errors (by having the corresponding bars for both methods directly side-by-side, with the Bayesian ones in red color). The barplot should be equipped with suitable labels and legends to enable good readability.

Comment on the outcome.

**Answer**:

```{r}
#---
require(LearnBayes)
X=cbind(1,melter$voltage,melter$ind1,melter$ind2,melter$ind3,melter$ind4,melter$temp1,melter$temp2,melter$temp3,melter$temp4,melter$temp5,melter$temp6,melter$temp7,melter$temp8,melter$temp9,melter$temp10,melter$temp11,melter$temp12,melter$temp13,melter$temp14,melter$temp15)
bayes.fit <- blinreg(melter$viscosity,X,1000)
XX=melter[,2:21]
x0=c(360,20,20,8,12,1000,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100)
XS <- apply(+rbind(XX, x0), 2, scale)
require(rjags)
model2_string <- "model{
 # Likelihood
  for(i in 1:N){  
    Y[i]   ~ dnorm(mu[i],inv.var)
    mu[i] <- alpha+inprod(X[i,],beta[])
  }
  
  # Values to predict 
  Y0  ~ dnorm(mu0,inv.var)
  mu0 <- alpha+inprod(X[N+1,],beta[])

  # Priors
  for(j in 1:p){
    beta[j] ~ dnorm(0,0.0001)
  }
  alpha     ~ dnorm(0, 0.01)
  inv.var   ~ dgamma(0.01, 0.01)
  sigma     <- 1/sqrt(inv.var)
}"
model2 <- jags.model(
   textConnection(model2_string), 
   data = list(X=XS, Y=melter$viscosity,N=dim(XX)[1], p=dim(XX)[2])
 )
update(model2,10000)
postmod2.samples = coda.samples(
   model2, 
   c("alpha", "beta", "sigma", "Y0", "mu0"), 
   10000
 )[[1]]
summary(postmod2.samples)
melter.bayes.sd=c(0.08248 ,0.09864 ,0.10658,0.12132, 0.12070, 0.17852,0.25105,0.55814,0.16203,0.28153 ,0.15457,0.26683,0.47085,0.21689,0.16373,0.16607,0.33831,0.24888,0.49832,0.45562)

barplot(melter.bayes.sd,col="red")
melter.sd <- c(0.1187182,0.08248,2.0218666,0.09864,2.3166066,0.10658,4.0011192,0.12132,2.5631995,0.12070,0.8122402,0.17852,1.1884594,0.25105,1.8228624,0.55814,0.5293583,0.16203,0.3952806,0.28153,0.5776417,0.15457,0.8423071,0.26683,1.4509358,0.47085,0.6492039,0.21689,0.7184004,0.16373,0.4189843,0.16607,0.6999098,0.33831,0.6261754,0.24888,1.2645329,0.49832,1.3274590,0.45562
)
melter.sd <- matrix(melter.sd,2,20)
barplot(melter.sd,col=c("blue","red"),legend = TRUE,beside=TRUE)
```

(2). The melter.bayes.sd should be: c(0.08248,0.09864,0.10658,0.12132,0.12070,0.17852,0.25105,0.55814,0.16203,0.28153,0.15457,0.26683,0.47085,0.21689,0.16373,0.16607,0.33831,0.24888,0.49832,0.45562), which can be seen from the summary of Naive SE.

The standard error of linear regression model is large and have much difference in different items. While the standard error of Bayesian linear regression model is much small, and there are not much difference in each items.
Therefore, I think the  Bayesian linear regression model is better.

# Task 3: The Lasso (20 marks)

We would like to reduce the dimension of the currently 20-dimensional space of predictors. We employ the LASSO to carry out this task. Carry out this analysis, which should feature the following elements:

 * the trace plot of the fitted coefficients, as a function of $\log(\lambda)$, with $\lambda$ denoting the penalty parameter;
 * a graphical illustration of the cross-validation to find $\lambda$;
 * the chosen value of $\lambda$ according to the `1-se` criterion, and a brief explanation of what this criterion actually does;
 * the fitted coefficients according to this choice of $\lambda$.

**Answer:**

```{r}
#---
require(glmnet) 
melter<-read.table("http://maths.dur.ac.uk/~dma0je/Data/melter.dat", header=TRUE)
melterX = as.matrix(melter[,2:21])
melter.fit= glmnet(melterX,melter$viscosity, alpha=1 )
plot(melter.fit, xvar="lambda",cex.axis=2, cex.lab=1.1,cex=1.1)
```

First see the trace plot of the fitted coefficients, as a function of log(λ). We see that, for higher λ, the ˆβj(λ) get shrunk, and less of them included.From the plots I find log(λ) is around 2. Then use a graphical illustration of the cross-validation to find λ.

```{r}
melter.cv.viscosity = cv.glmnet(melterX,melter$viscosity, alpha=1)
plot(melter.cv.viscosity)
lambda1 <- melter.cv.viscosity$lambda.1se
log(lambda1)
cbind( coef(melter.cv.viscosity, s="lambda.min"),coef(melter.cv.viscosity, s="lambda.1se"))
melter.refit<-lm(melter$viscosity~melter$voltage+melter$ind2+melter$temp2+melter$temp5+melter$temp6+melter$temp7+melter$temp8+melter$temp9+melter$temp12+melter$temp13+melter$temp14)
summary(melter.refit)
```

Then, calculate log(λ)=2.272147, so λ1se =exp(2.27)=9.7, so λ1se =9 (or less than 9)is a good choice.
Then, from the result, I think I can choose voltage, ind2, temp2, temp7, temp8, temp9, temp13, temp14. And the Coefficients are 0.315, 9.323, -2.666, -2.070, 2.292, -2.606, 1.282, 1.943 in order from the R result.

Next, carry out a Bayesian analysis of the lasso.  Visualize the full posterior distributions of all coefficients (except the intercept) in terms of boxplots, and also visualize the resulting standard errors of the coefficients, again using a barplot (with red bars).
Give an interpretation of the results, especially in terms of the evidence that this analysis gives in terms of inclusion/non-inclusion of certain variables.

**Answer:** 

```{r}
#---
require(monomvn)
melter.blas <- blasso(melterX, melter$viscosity)
melter.blas
boxplot(melter.blas$beta[,1:20]) 

melter.blas.sd=apply(melter.blas$beta,2,sd)
melter.blas.mean=apply(melter.blas$beta,2,mean)
barplot(melter.blas.sd, beside=TRUE, col="red")
melter.blas.mean-1.3*melter.blas.sd
melter.blas.mean+1.3*melter.blas.sd

```

At last, analysis gives in terms of inclusion/non-inclusion of certain variables and the boxplot, there are 7 or 8 items almost certain not equal to 0, and since λ should less than 9.So 8 is a good choice. And from all the plots, I think I can choose voltage, ind2, temp2, temp7,temp8, temp9, temp13, temp14 for 8 items.
I can also choose the items by mean and sigma, and here I choose 1.3*sigma as one direction of interval. And finally seclet the 8 items.
And from the barplot I can find ind2,ind3,ind4,temp3,temp8,temp15 have lager standard error, means they are more uncertaintity maybe.

# Task 4: Bootstrap (20 marks)

A second possibility to assess uncertainty of any estimator is the Bootstrap. Implement a nonparametric bootstrap procedure to assess the uncertainty of your frequentist lasso fit from Task 3.

Produce boxplots of the full bootstrap distributions for all coefficients (similar as in Task 3).

Then, add (green) bars with the resulting standard errors to the bar plot produced in Task 3, allowing for comparison between Bootstrap and Bayesian standard errors. Interpret the results.

**Answer:**

```{r}
#---
require(boot)
bs <- function(formula, data, indices) {
d <- data[indices,] # allows boot to select sample
fit <- lm(formula, data=d)
return(coef(fit))
}
results <- boot(data=melter, statistic=bs,  R=1000, formula=viscosity~voltage+ind1+ind2+ind3+ind4+temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9+temp10+temp11+temp12+temp13+temp14+temp15) 
results
boxplot(results$t[,2:21])
```

From the coefficients of the bootstrap, I can find the mean of each beta (in each group) and their standard error. Then calculate the mean of bootstrap model and then consider the two sigma theom, the t2(voltage), t4(ind2), t6(ind4), t7(temp1), t8(temp2), t10(temp4), t12(temp6), t14(temp8), t15(temp9), t21(temp15) should be contained since these beta won’t be 0 for high probability. And others can be removed.
I can see this from the boxplot, there are over 10 items almost certainly not be 0, So I think this model is not very good.

```{r}
melter.boot.sd=c(0.08641831,2.43894776,2.69301840,5.00473944,2.10844034,0.87508852,1.17275837,2.01161094,0.55720046,0.39654510,0.55528572,0.93306605,1.36032058,0.73597993,0.70723921,0.38904941,0.63179977,0.68962599,1.40806813,1.46744354)
barplot(melter.boot.sd, beside=TRUE, col="green")
```

```{r}
melter.sd=c(0.1599775,0.08641831, 0.9638781,2.43894776, 2.3265623,2.69301840, 2.2809233,5.00473944, 2.5812060,2.10844034, 0.9713306,0.87508852,1.3066594, 1.17275837,1.4409362, 2.01161094,0.6668648,0.55720046, 0.1623073 ,0.39654510,
0.7451741,0.55528572, 0.5636176,0.93306605, 1.6109732, 1.36032058,0.6667354,0.73597993, 0.2707238,0.70723921, 0.2780370,0.38904941, 0.7794815,0.63179977, 0.3882084, 0.68962599,1.1504105, 1.40806813,1.6948070,1.46744354)

melter.sd <- matrix(melter.sd,2,20)
barplot(melter.sd,col=c("red","green"),legend = TRUE,beside=TRUE)
```

The red is the standard errors of bayes model, and the green is the the standard errors of bootstrap model. 
Compared the standard error of Bootstrap model and Bayes model, most of them are almost the same, but some of the standard error of the Bootstrap model is lager than that of the bayes model. This means these beta in the bootstrap model have more uncertainty, so I think maybe the bayes lasso model is better here.

# Task 5: Model choice (10 marks)

Based on all considerations and analyses carried out so far, decide on a suitable model that you would present to a client, if you had been the statistical consultant. 

Formulate the model equation in mathematical notation.

Refit your selected model using ordinary Least Squares. Carry out some residual diagnostics for this fitted model, and display the results. Discuss these briefly.

**Answer:**

From the analysis I think I should choose from the Bootstrap model and Bayes model.
For the Bootstrap model, through the Q4, I know although for 2 sigma, there are still 10 items(a little much), so I think it is not very good to choose this model, and the R square of the fit model is also too small. For the beta of Bayes model, Since I get lambda is about 9 in the Q2, so I can choose 9 or less items to fit the function.Then I can remove the beta can’t be reject to be 0. So, I just leave some(actuarry 8) items.
I think I can still keep b1(voltage), b3(ind2), b6(temp1), b7(temp2), b13(temp8), b14(temp9), b19(temp14), b20(temp15) almost certainly for these 9.(b9, b11, b17 are also good but can’t compare with the items which choose) Other beta I just removed, so I can get this Bayes model( this model is the model I think is best from the above analysis of Q1 ,Q2,Q3,Q4):
Viscosity=0.202*voltage+10.089*ind2+2.415*temp1-5.305*temp2+4.279*temp8-2.065*temp9+1.91*temp14-3.148*temp15+C
(where C is a constant).  
Then I should refit my selected model using ordinary Least Squares:

```{r}
#---
melter.adjuest2=lm(viscosity~voltage+ind2+temp1+temp2+temp8+temp9+temp14+temp15,data=melter)
summary(melter.adjuest2)
```

Then I find the suitable model:
Viscosity=0.305*voltage+9.48*ind2+3.32*temp1- 6.7*temp2+5.415*temp8-1.901*temp9+4.5*temp14-4.485*temp15-92.376.
Then, I carry out some residual diagnostics for this fitted model, and get some plots:

```{r}
plot(melter.adjuest2$fitted.values,melter.adjuest2$residuals)
```

Then, check adequacy of the model specification and homoscedasticity. Since a trumpet-shape appears here so model is homoscedasticity violated. Therefore a possible solution is to transform the response, and a Box-Cox-transformation can help to identify the best power transformation.

```{r}
plot(melter.adjuest2$residuals)
```

Then from the plot, I can checks independence of residuals , in short, there should just be no pattern of any type. 
Therefore, these residuals are independent.

```{r}
qqnorm(melter.adjuest2$residuals)
qqline(melter.adjuest2$residuals)
```

The qq plot shows these residuals not follows the normal distribution well. So I think the model should still be improved.

# Task 6: Extensions (20 marks)
For this task, take the model (T5) as the starting point.  Then consider extensions of your model in TWO of the following THREE directions (of your choice).


(1) Replace the temperature sensor variables in model (T5) by an adequate number of principal components (see Task 1).

(2) Replace the `voltage`, and the remaining induction variables, by nonparametric terms.  

(3) Consider a transformation of the response variable `viscosity`. 

Each time, report the fitted model through adequate means. Discuss whether the corresponding extension is useful, giving quantitative or graphical evidence where possible.

Give a short discussion on whether any of your extensions have led to an actual improvement compared to model (T5).

**Answer:**

I choose (1) and (3)
(1)
 I can find for 9 principal component (PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9) it is up to over 0.99. So I think remove the temperature sensor variables, and it is enough to use at most these 9 items to replace. Then fit the models.
 In the first fit , I find the principal component PC3, PC5, PC6( PC6 is good but not so good compared with others so I also removed it). Then fit the model again and I can find also remove PC8 for the p value(0.01) ,which means I can’t reject the assumption of PC=0 in 0.01(anyway, sometimes it is ok for PC8=0 although the possibility is small). Then, I fit my model again and I find this result is quite good now.(The result of model Q5 is also good)
 
```{r}
#---
temp=melter[,7:21]
temp.pr <- prcomp(temp)
PC1=temp.pr$x
PC2=temp.pr$x[,1:5]
melterX.new=melter[,1:6]
new.data=cbind(melterX.new,PC1)
new.data.melter=data.frame(new.data)
new.data.melter
melter.new.fit2=lm(viscosity~voltage+ind2+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9,data=new.data.melter)
summary(melter.new.fit2)
melter.new.fit2=lm(viscosity~voltage+ind2+PC1+PC2+PC4+PC7+PC8+PC9,data=new.data.melter)
summary(melter.new.fit2)
melter.new.fit2=lm(viscosity~voltage+ind2+PC1+PC2+PC4+PC7+PC9,data=new.data.melter)
summary(melter.new.fit2)
```

So, I just need variables: voltage, ind2, PC1, PC2, PC4, PC7, PC9. And the improved model analysis is just show as plo3, and compared with the model in Q5, it does not require ﬁt a high-dimensional model to start with. PCA can be applied on the predictor space before any regression takes place, so this can save time for high-dimensional questions.
However, the model still have some problems since the R square is still low, and compared with the model in Q5, the R square is even lower, and maybe not so right. 
So, I think this way is useful for very high dimensional model, but not so meaningful in this question.


(3)When homoscedasticity is violated, a possible solution is to transform the response. A Box-Cox-transformation can help to identify the best power transformation. (but I should remove some data(viscosity =0) first). 

```{r}
melter.v=melter$viscosity^(-1)
which(melter.v=="Inf")
new.melte=cbind(melter.v,melter[,2:21])
new.melter1=melter[-c(2 ,37 ,180 ,187 ,196 ,212 ,300 ,324 ,378 ,464 ,471 ,492 ,587 ,659 ,708 ,715 ,754 ,756,791 ,809 ,821 ,845 ,849 ,864 ,871 ,876),]
new.melter1.dataframe=data.frame(new.melter1) 
melter.adjuest.new=lm(viscosity~voltage+ind2+temp1+temp2+temp8+temp9+temp14+temp15,data=new.melter1.dataframe)

```

```{r}
require(MASS)
boxcox(melter.adjuest.new)
```

From the plot,I can find the suitable lambda is about 0.65, so I try to get the data viscosity^(0.65), and fit the model again.

```{r}
melter.data=new.melter1.dataframe$viscosity^(0.65)
datamelter=cbind(melter.data,new.melter1.dataframe[,2:21])
dataframe.melter=data.frame(datamelter)
refit.melter=lm(melter.data~voltage+ind2+temp1+temp2+temp8+temp9+temp14+temp15,data=dataframe.melter)
summary(refit.melter)
```

this time all the items fits better(all the p value <0.01, even 0.005), so of course I can’t reject them. The intercept is better but compared with the model in Q5, but still not good. The R square is 0.39 this time ( Although still not so good ) and better than R square of the model in Q5.
Then I see the residuals.

```{r}
plot(refit.melter$fitted.values,refit.melter$residuals)
```

First, I can say the residuals is smaller than the residuals in the model in Q5.
Then, in the plot I see this model solve the heteroscedasticity well(If violated, then often a trumpet-shape appears ,as it is not here!).So, I think this improvement is quite good.

```{r}
qqnorm(refit.melter$residuals)
qqline(refit.melter$residuals)
```

In short, the improvement (1) is good for high dimensional data, but not so useful in this question. And the improvement (3) is good one since it solve solve the heteroscedasticity(just as the above plot shows) and improve the p value of betas and the adjust R square of the model. At last, for further improvement of the model I think I can consider some nonlinear model.
