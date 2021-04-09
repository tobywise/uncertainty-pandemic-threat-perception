library(tidyverse)
library(psych)
library(flextable)
library(kutils)
library(semTable)
library(lavaan)
library(parameters)
library(semPlot)
library(semTools)
library(lavaanPlot)

# Import data
gamble_ids <- read.csv('data/gamble_subs_anon.csv')$subjectID
T1_data <- read.csv('data/T1_data_anon.csv')

###################
# OUTLIER REMOVAL #
###################

# Remove multivariate outliers
outlier(T1_data %>% select(-c(id)))

# Get distance
distance <- mahalanobis(T1_data %>% select(-c(id)), center = colMeans(T1_data %>% select(-c(id))), cov = cov(T1_data %>% select(-c(id))))
cutoff <- (qchisq(p=1-0.001, df=ncol(T1_data %>% select(-c(id)))))
outliers <- which(distance > cutoff)
length(outliers)

# Figure out how many outliers are being removed
nrow(T1_data %>% filter(!id %in% gamble_ids)) - nrow(T1_data %>% slice(-outliers) %>% filter(!id %in% gamble_ids))
nrow(T1_data %>% filter(id %in% gamble_ids)) - nrow(T1_data %>% slice(-outliers) %>% filter(id %in% gamble_ids))

# Remove
T1_data <- T1_data %>% slice(-outliers)

###############################
# EXPLORATORY FACTOR ANALYSIS #
###############################

# This is run on subjects who did not complete the gambling task but completed questionnaire measures.

# Select non-gamble subjects for fitting factor analysis
train_data <- T1_data %>% filter(!id %in% gamble_ids)
write.csv(train_data, 'data/T1_EFA_data_anon.csv', row.names = FALSE)
train_data <- train_data %>% select(-c(id))

# Internal consistency
omega(train_data, nfactors=5)

# And gamble subjects for analysis
test_data <- T1_data %>% filter(id %in% gamble_ids)

test_ids <- test_data$id  # Save ID for use later
test_data <- test_data %>% select(-c(id))

# FACTOR ANALYSIS
# Correlation matrix
cormat <- round(cor(train_data),2)
cormat[abs(cormat) < 0.7] <- NA
cormat

cortest.bartlett(train_data)

# Run FA
fa_model <- fa(train_data, nfactors=10, fm="ml", rotate='oblimin')

# Scree plot
plot(fa_model$e.values, type='b')

#https://sakaluk.wordpress.com/2016/05/26/11-make-it-pretty-scree-plots-and-parallel-analysis-using-psych-and-ggplot2/
scree_data <- data.frame(num=1:length(fa_model$e.values), eigenvalue=fa_model$e.values)

p = ggplot(scree_data, aes(x=num, y=eigenvalue)) +
   geom_line() +
   geom_point(size=4) +
   scale_y_continuous(name='Eigenvalue') +
   scale_x_continuous(name='Factor Number', breaks=min(scree_data$num):max(scree_data$num)) +
   geom_vline(xintercept = 5, linetype = 'dashed') + theme_classic()

p


# 5 factors looks good, rerun FA
fa_model_5f <- fa(train_data, nfactors=5, fm="ml", rotate='oblimin')

print(fa_model_5f)

# Get loadings and safe
efa_loadings <- as.data.frame(unclass(fa_model_5f$loadings))
colnames(efa_loadings) = c('General_anxiety__T1', 'Virus_likelihood__T1', 'Behavior__T1', 'Virus_anxiety__T1', 'Virus_severity__T1')
write.csv(efa_loadings, 'data/efa_loadings.csv')

# Put loadings into a nice table
ft <- flextable(tibble::rownames_to_column(efa_loadings, "Question ID"))

ft <- colformat_num(x = ft,  big.mark = ",", digits = 2, na_str = "N/A")
ft

#################
# CFA in Lavaan #
#################

# CFA is run on subjects who did the gambling task


# Drop factors with loadings below 0.5 (there seems to be a paper recommending every possible threshold for this...)
fa_model_structure_5f <- efa_to_cfa(fa_model_5f, threshold = 0.5)

# Rename things
fa_model_structure_5f <-"
General_anx =~ Q048 + Q049 + Q050 + Q051 + Q052
Virus_likelihood =~ Q001 + Q005 + Q006 + Q008
Behavior =~ Q026 + Q030 + Q031
Virus_anxiety =~ Q033 + Q034 + Q035
Virus_severity =~ Q002 + Q003 + Q010
"

# Run CFA
fa_fit_5f <- cfa(fa_model_structure_5f, data = test_data, std.lv = FALSE)

# Check fit measures
fitMeasures(fa_fit_5f, c("chisq", "df", "pvalue", "cfi","rmsea","srmr"))
# Fit isn't perfect (on the border of acceptability for most measures), but isn't the worst

# Plot
semPaths(fa_fit_5f, what = "std", whatLabels = "std", 
         residuals = FALSE, intercepts = FALSE,
         # prettify
         fade = TRUE, sizeMan=3, layout='tree', rotation=4, curve=2,
         style = "lisrel", sizeLat = 8, 
         nCharNodes = 50, 
         edge.label.cex = 0.5, edge.color='#4a4a4a',
         color = list(lat = rgb(219, 219, 219, maxColorValue = 255), 
                      man = rgb(117, 188, 255, maxColorValue = 255)))

# Save factor scores
T1_factor_scores <- lavPredict(fa_fit_5f)
T1_factor_scores <- data.frame(T1_factor_scores, test_ids)
colnames(T1_factor_scores) <- c('General_anxiety__T1', 'Virus_likelihood__T1', 'Behavior__T1', 'Virus_anxiety__T1', 'Virus_severity__T1', 'id')
write.csv(T1_factor_scores, 'data/T1_factor_scores.csv', row.names = FALSE)

# Get loadings
loadings_T1 <- standardizedsolution(fa_fit_5f) %>% subset(op == '=~')

ft <- flextable(parameterEstimates(fa_fit_5f, standardized=TRUE) %>% 
             filter(op == "=~") %>% 
             select('Latent Factor'=lhs, Indicator=rhs, B=est, SE=se, Z=z, Beta=std.all))

ft <- colformat_num(x = ft, j = c('B', 'SE', 'Z', 'Beta'), big.mark = ",", digits = 2, na_str = "N/A")
ft


# Get T9 data
T9_data <- read.csv('data/T9_data_anon.csv')

# Remove multivariate outliers
outlier(T9_data %>% select(-c(id)))

# Get distance
distance <- mahalanobis(T9_data %>% select(-c(id)), center = colMeans(T9_data %>% select(-c(id))), cov = cov(T9_data %>% select(-c(id))))
cutoff <- (qchisq(p=1-0.001, df=ncol(T9_data %>% select(-c(id)))))
outliers <- which(distance > cutoff)

# Remove
T9_data<- T9_data %>% slice(-outliers)
outlier(T9_data %>% select(-c(id)))

T9_ids <- T9_data$id  # Save ID for use later
T9_data <- T9_data %>% select(-c(id))

# CFA on T9
fa_fit_5f_T9 <- cfa(fa_model_structure_5f, data = T9_data, std.lv = FALSE)
fitMeasures(fa_fit_5f_T9, c("chisq", "df", "pvalue", "cfi","rmsea","srmr"))

# pdf('cfa_figure_T9.pdf')
semPaths(fa_fit_5f_T9, what = "std", whatLabels = "std", 
         residuals = FALSE, intercepts = FALSE,
         # prettify
         fade = TRUE, sizeMan=3, layout='tree', rotation=4, curve=2,
         style = "lisrel", sizeLat = 8, 
         nCharNodes = 50, 
         edge.label.cex = 0.5, edge.color='#4a4a4a',
         color = list(lat = rgb(219, 219, 219, maxColorValue = 255), 
                      man = rgb(117, 188, 255, maxColorValue = 255)))
# dev.off()


# Compare fit

cfa_timepoint_comparison <- compareLavaan(c(fa_fit_5f, fa_fit_5f_T9))
write.csv(cfa_timepoint_comparison, 'data/cfa_timepoint_comparison.csv')

# Assess measurement invariance

test_data$time <- 1
T9_data$time <- 2

both_timepoints_data <- rbind(test_data, T9_data)

measurementInvariance(model=fa_model_structure_5f, 
                      data = both_timepoints_data, 
                      group = "time", strict=TRUE)


# Get T9 factor scores

T9_factor_scores <- lavPredict(fa_fit_5f_T9)
T9_factor_scores <- data.frame(T9_factor_scores, T9_ids)
colnames(T9_factor_scores) <- c('General_anxiety__T2', 'Virus_likelihood__T2', 'Behavior__T2', 'Virus_anxiety__T2', 'Virus_severity__T2', 'id')
write.csv(T9_factor_scores, 'data/T9_factor_scores.csv', row.names = FALSE)

loadings_T9 <- standardizedsolution(fa_fit_5f_T9) %>% subset(op == '=~')

# Check correlation between estimated loadings across time points
cor.test(loadings_T1$est.std, loadings_T9$est.std)

########################
# T1 LATENT PATH MODEL #
########################

# Get data
test_data$id <- test_ids
ses_data <- read.csv('data/ses_data.csv')
demographic_data <- read.csv('data/demographics_anon.csv')
demographic_data$id <- demographic_data$subjectID
test_data_merged <- merge(test_data, ses_data, on='id')
test_data_merged <- merge(test_data_merged, demographic_data, on='id')

model_data <- read_csv('data/model_measures.csv')
model_data$id <- model_data$subjectID
T1_combined_data <- inner_join(test_data_merged, model_data, by='id')

write.csv(T1_combined_data, 'data/T1_combined_data.csv')


# Run model with lavaan

latent_path_model_MB <-"

! Factors
General_anxiety =~ 1*Q048 + Q049 + Q050 + Q051 + Q052
Virus_likelihood =~ 1*Q001 + Q005 + Q006 + Q008
Behavior =~ 1*Q026 + Q030 + Q031
Virus_anxiety =~ 1*Q033 + Q034 + Q035
Virus_severity =~ 1*Q002 + Q003 + Q010
  
! Single indicator constructs - used for predictors
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Effects of regressors
Behavior ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age
Virus_anxiety ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age
General_anxiety ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age
Virus_likelihood ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age
Virus_severity ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age

rho ~~ 0*rho
lambda ~~ 0*lambda
alpha_sg ~~ 0*alpha_sg
alpha_sl ~~ 0*alpha_sl
alpha_rg ~~ 0*alpha_rg
alpha_rl ~~ 0*alpha_rl
estimated_LR_diff ~~ 0*estimated_LR_diff

";

result_latent_path_model_MB<-sem(latent_path_model_MB, data=T1_combined_data)

# Check fit
fitMeasures(result_latent_path_model_MB, c("chisq", "pvalue", "cfi","rmsea","srmr"))

# Save output
model_results <- c('result_latent_path_model_MB')

summaries <- c()

for (m in model_results) {
   v = get(m)
   model_summary <- standardizedSolution(v)
   model_pe <- as_tibble(model_summary)
   model_pe <- model_pe %>% subset(op == '~')
   model_pe$model_name <- m
   summaries <- rbind(summaries, model_pe)
}

write.csv(summaries, 'data/sem_outputs/latent_path_model_summaries.csv', row.names = FALSE)

# Predict factor scores
T1_predicted <- as.data.frame(lavPredict(result_latent_path_model_MB))
T1_predicted$id <- (T1_combined_data %>% subset(!is.na(T1_combined_data$age)))$id
T1_predicted$age <- (T1_combined_data %>% subset(!is.na(T1_combined_data$age)))$age

###################
# CONTROL FOR SES #
###################

latent_path_model_MB <-"

! Factors
General_anxiety =~ 1*Q048 + Q049 + Q050 + Q051 + Q052
Virus_likelihood =~ 1*Q001 + Q005 + Q006 + Q008
Behavior =~ 1*Q026 + Q030 + Q031
Virus_anxiety =~ 1*Q033 + Q034 + Q035
Virus_severity =~ 1*Q002 + Q003 + Q010
  
! Single indicator constructs - used for predictors
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Effects of regressors
Behavior ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age + ses
Virus_anxiety ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age + ses
General_anxiety ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age + ses
Virus_likelihood ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age + ses
Virus_severity ~ risk_aversion + loss_aversion + Amb_SG + Amb_SL + Amb_RG + Amb_RL + LR_diff + age + ses

rho ~~ 0*rho
lambda ~~ 0*lambda
alpha_sg ~~ 0*alpha_sg
alpha_sl ~~ 0*alpha_sl
alpha_rg ~~ 0*alpha_rg
alpha_rl ~~ 0*alpha_rl
estimated_LR_diff ~~ 0*estimated_LR_diff

";

result_latent_path_model_MB<-sem(latent_path_model_MB, data=T1_combined_data)
fitMeasures(result_latent_path_model_MB, c("chisq", "pvalue", "cfi","rmsea","srmr"))

model_results <- c('result_latent_path_model_MB')

summaries <- c()

for (m in model_results) {
   v = get(m)
   # model_summary <- summary(v, standardize=TRUE)
   model_summary <- standardizedSolution(v)
   model_pe <- as_tibble(model_summary)
   model_pe <- model_pe %>% subset(op == '~')
   model_pe$model_name <- m
   summaries <- rbind(summaries, model_pe)
}

write.csv(summaries, 'data/sem_outputs/latent_path_model_summaries_SES.csv', row.names = FALSE)

### PLOT A MODEL EXAMPLE

# pdf('latent_path.pdf')
semPaths(result_latent_path_model_MB, 
         residuals = TRUE, intercepts = FALSE,
         # prettify
         fade = TRUE, sizeMan=3, layout='tree', rotation=4, curve=2,
         sizeLat = 8, 
         nCharNodes = 50, 
         edge.label.cex = 0.5, edge.color='#4a4a4a',
         color = list(lat = rgb(219, 219, 219, maxColorValue = 255), 
                      man = rgb(117, 188, 255, maxColorValue = 255)))
# dev.off



#############################
# LATENT CHANGE SCORE MODEL #
#############################

# test_data_merged$id <- test_ids
T9_data$id <- T9_ids

gamble_data <- read_csv('data/mf_gamble_measures.csv')
gamble_data$id <- gamble_data$subjectID

longitudinal_data <- inner_join(T9_data, test_data_merged, by='id', suffix=c('_T2', '_T1'))

write.csv(longitudinal_data, 'data/longitudinal_data_gambling_anon.csv', row.names = FALSE)


# Compare subjects who dropped out
T1_predicted$follow_up <- T1_predicted$id %in% longitudinal_data$id
logistic_glm <- glm(follow_up ~ ., data = T1_predicted %>% select(!id), family = "binomial")
summary(logistic_glm)  # Nothing significant

ft <- as_flextable(logistic_glm)


longitudinal_data <- inner_join(longitudinal_data, model_data, by='id')
write.csv(longitudinal_data, 'data/longitudinal_data_gambling_anon.csv', row.names = FALSE)

## VIRUS ANXIETY
model_mb_virus_anxiety <-"

! Factors
  T1 =~ 1*Q033_T1 + (lambda1)*Q034_T1 + (lambda2)*Q035_T1
  ! For some reason it doesn't converge if using the same loading on Q35 across T1 & T2
  ! Measurement invariance checks were fine, and the T1 and T2 loadings using this model are almost
  ! Identical (1.418 & 1.421) - it also converges when SES is included
  T2 =~ 1*Q033_T2 + (lambda1)*Q034_T2 + (lambda3)*Q035_T2 
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q033_T1 ~~ VAR_Q033*Q033_T1
   Q034_T1 ~~ VAR_Q034*Q034_T1
   Q035_T1 ~~ VAR_Q035*Q035_T1

   Q033_T2 ~~ VAR_Q033*Q033_T2
   Q034_T2 ~~ VAR_Q034*Q034_T2
   Q035_T2 ~~ VAR_Q035*Q035_T2

   Q033_T1 ~~ COV_Q033*Q033_T2
   Q034_T1 ~~ COV_Q034*Q034_T2
   Q035_T1 ~~ COV_Q035*Q035_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q033_T1~0*1;
   Q034_T1~0*1;
   Q035_T1~0*1;
   Q033_T2~0*1;
   Q034_T2~0*1;
   Q035_T2~0*1;
";


result_mb_virus_anxiety<-sem(model_mb_virus_anxiety, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_anxiety, c("chisq", "pvalue", "cfi","rmsea","srmr"))
summary(result_mb_virus_anxiety, fit.measures=TRUE, standardize=TRUE)


##  GENERAL ANXIETY

model_mb_general_anxiety <-"

! Factors
  T1 =~ 1*Q048_T1 + (lambda1)*Q049_T1 + (lambda4)*Q050_T1 + (lambda2)*Q051_T1 + (lambda3)*Q052_T1
  T2 =~ 1*Q048_T2 + (lambda1)*Q049_T2 + (lambda4)*Q050_T2 + (lambda2)*Q051_T2 + (lambda3)*Q052_T2
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q048_T1 ~~ VAR_Q048*Q048_T1
   Q049_T1 ~~ VAR_Q049*Q049_T1
   Q050_T1 ~~ VAR_Q050*Q050_T1
   Q051_T1 ~~ VAR_Q051*Q051_T1
   Q052_T1 ~~ VAR_Q052*Q052_T1

   Q048_T2 ~~ VAR_Q048*Q048_T2
   Q049_T2 ~~ VAR_Q049*Q049_T2
   Q050_T2 ~~ VAR_Q050*Q050_T2
   Q051_T2 ~~ VAR_Q051*Q051_T2
   Q052_T2 ~~ VAR_Q052*Q052_T2

   Q048_T1 ~~ COV_Q048*Q048_T2
   Q049_T1 ~~ COV_Q049*Q049_T2
   Q050_T1 ~~ COV_Q050*Q050_T2
   Q051_T1 ~~ COV_Q051*Q051_T2
   Q052_T1 ~~ COV_Q052*Q052_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q048_T1~0*1;
   Q049_T1~0*1;
   Q050_T1~0*1;
   Q051_T1~0*1;
   Q052_T1~0*1;
   Q048_T2~0*1;
   Q049_T2~0*1;
   Q050_T2~0*1;
   Q051_T2~0*1;
   Q052_T2~0*1;
";

result_mb_general_anxiety<-sem(model_mb_general_anxiety, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_general_anxiety, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_general_anxiety, fit.measures=TRUE, standardize=TRUE)




##  VIRUS LIKELIHOOD

model_mb_virus_likelihood <-"

! Factors
  T1 =~ 1*Q001_T1 + (lambda1)*Q005_T1 + (lambda2)*Q006_T1 + (lambda3)*Q008_T1
  T2 =~ 1*Q001_T2 + (lambda1)*Q005_T2 + (lambda2)*Q006_T2 + (lambda3)*Q008_T2

risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q001_T1 ~~ VAR_Q001*Q001_T1
   Q005_T1 ~~ VAR_Q005*Q005_T1
   Q006_T1 ~~ VAR_Q006*Q006_T1
   Q008_T1 ~~ VAR_Q008*Q008_T1

   Q001_T2 ~~ VAR_Q001*Q001_T2
   Q005_T2 ~~ VAR_Q005*Q005_T2
   Q006_T2 ~~ VAR_Q006*Q006_T2
   Q008_T2 ~~ VAR_Q008*Q008_T2

   Q001_T1 ~~ COV_Q001*Q001_T2
   Q005_T1 ~~ COV_Q005*Q005_T2
   Q006_T1 ~~ COV_Q006*Q006_T2
   Q008_T1 ~~ COV_Q008*Q008_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q001_T1~0*1;
   Q005_T1~0*1;
   Q006_T1~0*1;
   Q008_T1~0*1;
   Q001_T2~0*1;
   Q005_T2~0*1;
   Q006_T2~0*1;
   Q008_T2~0*1;
";



result_mb_virus_likelihood<-sem(model_mb_virus_likelihood, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_likelihood, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_virus_likelihood, fit.measures=TRUE, standardize=TRUE)




##  VIRUS SEVERITY

model_mb_virus_severity<-"

! Factors
  T1 =~ 1*Q002_T1 + (lambda1)*Q003_T1 + (lambda2)*Q010_T1
  T2 =~ 1*Q002_T2 + (lambda1)*Q003_T2 + (lambda2)*Q010_T2

   risk_aversion =~ 1*rho
   loss_aversion =~ 1*lambda
   Amb_SG =~ 1*alpha_sg
   Amb_SL =~ 1*alpha_sl
   Amb_RG =~ 1*alpha_rg
   Amb_RL =~ 1*alpha_rl
   LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q002_T1 ~~ VAR_Q002*Q002_T1
   Q003_T1 ~~ VAR_Q003*Q003_T1
   Q010_T1 ~~ VAR_Q010*Q010_T1

   Q002_T2 ~~ VAR_Q002*Q002_T2
   Q003_T2 ~~ VAR_Q003*Q003_T2
   Q010_T2 ~~ VAR_Q010*Q010_T2

   Q002_T1 ~~ COV_Q002*Q002_T2
   Q003_T1 ~~ COV_Q003*Q003_T2
   Q010_T1 ~~ COV_Q010*Q010_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q002_T1~0*1;
   Q003_T1~0*1;
   Q010_T1~0*1;
   Q002_T2~0*1;
   Q003_T2~0*1;
   Q010_T2~0*1;
";


result_mb_virus_severity<-sem(model_mb_virus_severity, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_severity, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_virus_severity, fit.measures=TRUE, standardize=TRUE)




##  BEHAVIOUR

model_mb_behavior<-"

! Factors
  T1 =~ 1*Q030_T1 + (lambda1)*Q031_T1 + (lambda2)*Q026_T1
  T2 =~ 1*Q030_T2 + (lambda1)*Q031_T2 + (lambda2)*Q026_T2
  
   risk_aversion =~ 1*rho
   loss_aversion =~ 1*lambda
   Amb_SG =~ 1*alpha_sg
   Amb_SL =~ 1*alpha_sl
   Amb_RG =~ 1*alpha_rg
   Amb_RL =~ 1*alpha_rl
   LR_diff =~ 1*estimated_LR_diff


! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q030_T1 ~~ VAR_Q030*Q030_T1
   Q031_T1 ~~ VAR_Q031*Q031_T1
   Q026_T1 ~~ VAR_Q026*Q026_T1

   Q030_T2 ~~ VAR_Q030*Q030_T2
   Q031_T2 ~~ VAR_Q031*Q031_T2
   Q026_T2 ~~ VAR_Q026*Q026_T2

   Q030_T1 ~~ COV_Q030*Q030_T2
   Q031_T1 ~~ COV_Q031*Q031_T2
   Q026_T1 ~~ COV_Q026*Q026_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q030_T1~0*1;
   Q031_T1~0*1;
   Q026_T1~0*1;
   Q030_T2~0*1;
   Q031_T2~0*1;
   Q026_T2~0*1;
";

result_mb_behavior<-sem(model_mb_behavior, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_behavior, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_behavior, fit.measures=TRUE, standardize=TRUE)


#################################


# GET OUTPUTS AND MAKE THEM NICE
model_results <- c('result_mb_behavior', 'result_mb_virus_anxiety', 'result_mb_general_anxiety', 'result_mb_virus_likelihood', 'result_mb_virus_severity')

summaries <- c()

for (m in model_results) {
   v = get(m)
   # model_summary <- summary(v, standardize=TRUE)
   model_summary <- standardizedSolution(v)
   model_pe <- as_tibble(model_summary)
   model_pe <- model_pe %>% subset(op == '~')
   model_pe$model_name <- m
   summaries <- rbind(summaries, model_pe)
}

write.csv(summaries, 'data/sem_outputs/LC_model_summaries.csv', row.names = FALSE)

## MODEL FIT

comparison <- compareLavaan(c('Virus anxiety' = result_mb_virus_anxiety, 
                'General anxiety' = result_mb_general_anxiety, 
                'Virus likelihood' = result_mb_virus_likelihood, 
                'Virus severity' = result_mb_virus_severity, 
                'Virus behavior' = result_mb_behavior))
write.csv(comparison, 'data/sem_outputs/LCSM_comparison.csv')


##################################################
# LATENT CHANGE SCORE MODELS CONTROLLING FOR SES #
##################################################

## VIRUS ANXIETY

model_mb_virus_anxiety<-"

! Factors
  T1 =~ 1*Q033_T1 + (lambda1)*Q034_T1 + (lambda2)*Q035_T1
  T2 =~ 1*Q033_T2 + (lambda1)*Q034_T2 + (lambda2)*Q035_T2 

risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age
   dT1~(theta9)*ses

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q033_T1 ~~ VAR_Q033*Q033_T1
   Q034_T1 ~~ VAR_Q034*Q034_T1
   Q035_T1 ~~ VAR_Q035*Q035_T1

   Q033_T2 ~~ VAR_Q033*Q033_T2
   Q034_T2 ~~ VAR_Q034*Q034_T2
   Q035_T2 ~~ VAR_Q035*Q035_T2

   Q033_T1 ~~ COV_Q033*Q033_T2
   Q034_T1 ~~ COV_Q034*Q034_T2
   Q035_T1 ~~ COV_Q035*Q035_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q033_T1~0*1;
   Q034_T1~0*1;
   Q035_T1~0*1;
   Q033_T2~0*1;
   Q034_T2~0*1;
   Q035_T2~0*1;

";

result_mb_virus_anxiety<-sem(model_mb_virus_anxiety, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_anxiety, c("chisq", "pvalue", "cfi","rmsea","srmr"))
summary(result_mb_virus_anxiety, fit.measures=TRUE, standardize=TRUE)


##  GENERAL ANXIETY

model_mb_general_anxiety<-"

! Factors
  T1 =~ 1*Q048_T1 + (lambda1)*Q049_T1 + (lambda4)*Q050_T1 + (lambda2)*Q051_T1 + (lambda3)*Q052_T1
  T2 =~ 1*Q048_T2 + (lambda1)*Q049_T2 + (lambda4)*Q050_T2 + (lambda2)*Q051_T2 + (lambda3)*Q052_T2
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age
   dT1~(theta9)*ses

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q048_T1 ~~ VAR_Q048*Q048_T1
   Q049_T1 ~~ VAR_Q049*Q049_T1
   Q050_T1 ~~ VAR_Q050*Q050_T1
   Q051_T1 ~~ VAR_Q051*Q051_T1
   Q052_T1 ~~ VAR_Q052*Q052_T1

   Q048_T2 ~~ VAR_Q048*Q048_T2
   Q049_T2 ~~ VAR_Q049*Q049_T2
   Q050_T2 ~~ VAR_Q050*Q050_T2
   Q051_T2 ~~ VAR_Q051*Q051_T2
   Q052_T2 ~~ VAR_Q052*Q052_T2

   Q048_T1 ~~ COV_Q048*Q048_T2
   Q049_T1 ~~ COV_Q049*Q049_T2
   Q050_T1 ~~ COV_Q050*Q050_T2
   Q051_T1 ~~ COV_Q051*Q051_T2
   Q052_T1 ~~ COV_Q052*Q052_T2

   T2~~0*T2;

   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q048_T1~0*1;
   Q049_T1~0*1;
   Q050_T1~0*1;
   Q051_T1~0*1;
   Q052_T1~0*1;
   Q048_T2~0*1;
   Q049_T2~0*1;
   Q050_T2~0*1;
   Q051_T2~0*1;
   Q052_T2~0*1;
";

result_mb_general_anxiety<-sem(model_mb_general_anxiety, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_general_anxiety, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_general_anxiety, fit.measures=TRUE, standardize=TRUE)




##  VIRUS LIKELIHOOD

model_mb_virus_likelihood<-"

! Factors
  T1 =~ 1*Q001_T1 + (lambda1)*Q005_T1 + (lambda2)*Q006_T1 + (lambda3)*Q008_T1
  T2 =~ 1*Q001_T2 + (lambda1)*Q005_T2 + (lambda2)*Q006_T2 + (lambda3)*Q008_T2

risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age
   dT1~(theta9)*ses

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q001_T1 ~~ VAR_Q001*Q001_T1
   Q005_T1 ~~ VAR_Q005*Q005_T1
   Q006_T1 ~~ VAR_Q006*Q006_T1
   Q008_T1 ~~ VAR_Q008*Q008_T1

   Q001_T2 ~~ VAR_Q001*Q001_T2
   Q005_T2 ~~ VAR_Q005*Q005_T2
   Q006_T2 ~~ VAR_Q006*Q006_T2
   Q008_T2 ~~ VAR_Q008*Q008_T2

   Q001_T1 ~~ COV_Q001*Q001_T2
   Q005_T1 ~~ COV_Q005*Q005_T2
   Q006_T1 ~~ COV_Q006*Q006_T2
   Q008_T1 ~~ COV_Q008*Q008_T2

   T2~~0*T2;
   

! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q001_T1~0*1;
   Q005_T1~0*1;
   Q006_T1~0*1;
   Q008_T1~0*1;
   Q001_T2~0*1;
   Q005_T2~0*1;
   Q006_T2~0*1;
   Q008_T2~0*1;
";
result_mb_virus_likelihood<-sem(model_mb_virus_likelihood, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_likelihood, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_virus_likelihood, fit.measures=TRUE, standardize=TRUE)




##  VIRUS SEVERITY

model_mb_virus_severity<-"

! Factors
  T1 =~ 1*Q002_T1 + (lambda1)*Q003_T1 + (lambda2)*Q010_T1
  T2 =~ 1*Q002_T2 + (lambda1)*Q003_T2 + (lambda2)*Q010_T2

risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff

! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age
   dT1~(theta9)*ses

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q002_T1 ~~ VAR_Q002*Q002_T1
   Q003_T1 ~~ VAR_Q003*Q003_T1
   Q010_T1 ~~ VAR_Q010*Q010_T1

   Q002_T2 ~~ VAR_Q002*Q002_T2
   Q003_T2 ~~ VAR_Q003*Q003_T2
   Q010_T2 ~~ VAR_Q010*Q010_T2

   Q002_T1 ~~ COV_Q002*Q002_T2
   Q003_T1 ~~ COV_Q003*Q003_T2
   Q010_T1 ~~ COV_Q010*Q010_T2

   T2~~0*T2;
   
! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q002_T1~0*1;
   Q003_T1~0*1;
   Q010_T1~0*1;
   Q002_T2~0*1;
   Q003_T2~0*1;
   Q010_T2~0*1;
";
result_mb_virus_severity<-sem(model_mb_virus_severity, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_virus_severity, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_virus_severity, fit.measures=TRUE, standardize=TRUE)


##  BEHAVIOUR

model_mb_behavior<-"

! Factors
  T1 =~ 1*Q030_T1 + (lambda1)*Q031_T1 + (lambda2)*Q026_T1
  T2 =~ 1*Q030_T2 + (lambda1)*Q031_T2 + (lambda2)*Q026_T2
  
risk_aversion =~ 1*rho
loss_aversion =~ 1*lambda
Amb_SG =~ 1*alpha_sg
Amb_SL =~ 1*alpha_sl
Amb_RG =~ 1*alpha_rg
Amb_RL =~ 1*alpha_rl
LR_diff =~ 1*estimated_LR_diff


! Latent change scores
   T2~1*T1
   dT1=~1*T2
   dT1~(T1__dT1)*T1

! Effects of covarirates
   dT1~(theta1)*risk_aversion
   dT1~(theta2)*loss_aversion
   dT1~(theta3)*Amb_SG
   dT1~(theta4)*Amb_SL
   dT1~(theta5)*Amb_RG
   dT1~(theta6)*Amb_RL
   dT1~(theta7)*LR_diff
   dT1~(theta8)*age
   dT1~(theta9)*ses

   
! residuals, variances and covariances
   T1 ~~ VAR_T1*T1
   dT1 ~~ VAR_dT1*dT1

   Q030_T1 ~~ VAR_Q030*Q030_T1
   Q031_T1 ~~ VAR_Q031*Q031_T1
   Q026_T1 ~~ VAR_Q026*Q026_T1

   Q030_T2 ~~ VAR_Q030*Q030_T2
   Q031_T2 ~~ VAR_Q031*Q031_T2
   Q026_T2 ~~ VAR_Q026*Q026_T2

   Q030_T1 ~~ COV_Q030*Q030_T2
   Q031_T1 ~~ COV_Q031*Q031_T2
   Q026_T1 ~~ COV_Q026*Q026_T2

   T2~~0*T2;

! means
   dT1~slope*1
   T1~intercept*1
   T2~0*1;
   Q030_T1~0*1;
   Q031_T1~0*1;
   Q026_T1~0*1;
   Q030_T2~0*1;
   Q031_T2~0*1;
   Q026_T2~0*1;
";

result_mb_behavior<-sem(model_mb_behavior, data=longitudinal_data, fixed.x=FALSE, missing="FIML");
fitMeasures(result_mb_behavior, c("chisq", "pvalue", "cfi","rmsea","srmr"))

summary(result_mb_behavior, fit.measures=TRUE, standardize=TRUE)


#################################


# GET OUTPUTS AND MAKE THEM NICE
model_results <- c('result_mb_behavior', 'result_mb_virus_anxiety', 'result_mb_general_anxiety', 'result_mb_virus_likelihood', 'result_mb_virus_severity')

summaries <- c()

for (m in model_results) {
   print(m)
   v = get(m)
   # model_summary <- summary(v, standardize=TRUE)
   model_summary <- standardizedSolution(v)
   model_pe <- as_tibble(model_summary)
   model_pe <- model_pe %>% subset(op == '~')
   model_pe$model_name <- m
   summaries <- rbind(summaries, model_pe)
}

write.csv(summaries, 'data/sem_outputs/LC_model_summaries_SES.csv', row.names = FALSE)


###############################


# pdf('latent_path.pdf')
semPaths(result_mb_behavior, 
         residuals = TRUE, intercepts = FALSE,
         # prettify
         fade = TRUE, sizeMan=3, layout='tree', rotation=1, curve=2,
         sizeLat = 8, 
         nCharNodes = 50, 
         edge.label.cex = 0.5, edge.color='#4a4a4a',
         color = list(lat = rgb(219, 219, 219, maxColorValue = 255), 
                      man = rgb(117, 188, 255, maxColorValue = 255)))
# dev.off()

# Output latent variables etc
mb_results <- c('result_mb_virus_anxiety', 'result_mb_general_anxiety', 'result_mb_virus_likelihood', 'result_mb_virus_severity', 'result_mb_behavior')

predicted <- data.frame(id=longitudinal_data$id)

for (m in mb_results) {
   v = get(m)
   model_predicted <- lavPredict(v)
   model_predicted <- as.data.frame(model_predicted)
   model_predicted <- model_predicted %>% select(c('T1', 'T2', 'dT1'))
   colnames(model_predicted) <- c(paste(sub('result_mb_', '', m), '__T1', sep=''), 
                                  paste(sub('result_mb_', '', m), '__T2', sep=''), 
                                  paste(sub('result_mb_', '', m), '__dT1', sep=''))
   predicted <- data.frame(predicted, model_predicted)
}

write.csv(predicted, 'data/factor_scores_wide.csv', row.names = FALSE)


factor_scores_long <- predicted %>% select(contains(c('_T1', '_T2'))) %>% pivot_longer(cols=contains('_T'), names_to=c('factor', 'time_point'), names_sep='__', values_to='score')
write.csv(factor_scores_long, 'data/factor_scores_long.csv', row.names = FALSE)

ggplot(factor_scores_long, aes(x = time_point, y = score, group=factor, color=factor)) + 
   facet_wrap(vars(factor), scales = "free") + 
   geom_line(aes(group = id, color = factor), alpha=0.1) +
   stat_summary(fun = "mean", geom = "line", size = 2) +
   stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .1,
                position = position_dodge(.1), color = "black") +
   scale_color_manual(values=c('#267fd3', '#d3265a', '#fc9403', '#44af69', '#009cb8')) +
   theme_bw()























