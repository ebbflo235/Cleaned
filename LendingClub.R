####################################################################################################################
### Section 1 ######################################################################################################
## Load libraries and read in data #################################################################################
####################################################################################################################

#Remove all variables
rm(list = ls())
setwd("C:/Users/Trader/Desktop/PSC/PSC_Senior Associate_Modeling Test")

library(foreign)
library(data.table)
library(ggplot2)
library(readxl)
library(glmnet)
library(dplyr)
library(pROC)

#Read in data
OriginationsTape <- as.data.table(read_excel("Originations Tape.xlsx"))
PaymentHistory1 <- as.data.table(read_excel("PaymentHistoryData_Part_1.xlsx"))
PaymentHistory2 <- as.data.table(read_excel("PaymentHistoryData_Part_1.xlsx"))
#Combine payment history
PaymentHistoryTotal = rbind(PaymentHistory1,PaymentHistory2)

####################################################################################################################
## Section 2a ######################################################################################################
## Data Cleaning ###################################################################################################
####################################################################################################################

#Figure out missing data
colnames(OriginationsTape)[colSums(is.na(OriginationsTape)) > 0]

#32 in total
# a little cumbersome but go one by one to figure out whats best to do with it
# not many missing
OriginationsTape = OriginationsTape[-which(is.na(OriginationsTape$dti)),]
OriginationsTape$mths_since_last_delinq = OriginationsTape$mths_since_last_record = NULL
#going to drop mths_since_last_deliq and mths_since_last_record
OriginationsTape = OriginationsTape[-which(is.na(OriginationsTape$revol_util)),]
OriginationsTape$mths_since_last_major_derog = NULL
#turn NAs to false for next two variables
OriginationsTape$annual_inc_joint[is.na(OriginationsTape$annual_inc_joint)] <- "FALSE"
OriginationsTape$dti_joint[is.na(OriginationsTape$dti_joint)] <- "FALSE"
#this one is all NA
OriginationsTape$verification_status_joint = NULL
#captured in open accounts, sparse data
OriginationsTape$open_act_il = OriginationsTape$open_il_12m = OriginationsTape$open_il_24m = NULL
#not enough data to distinguish, also captured
OriginationsTape$mths_since_rcnt_il = NULL
#captured in other total / utilization metrics
OriginationsTape$total_bal_il = OriginationsTape$il_util = OriginationsTape$open_rv_12m = OriginationsTape$open_rv_24m = NULL
#a little too granular, not enough data
OriginationsTape$max_bal_bc = NULL
#captured in broader metrics
OriginationsTape$all_util = OriginationsTape$inq_fi = OriginationsTape$total_cu_tl = OriginationsTape$inq_last_12m = NULL
# not nany NAs here, set to zero as is intuitive that NA would be nothing open to buy
OriginationsTape$bc_open_to_buy[is.na(OriginationsTape$bc_open_to_buy)] <- 0
OriginationsTape$bc_util[is.na(OriginationsTape$bc_util)] <- 0
#very granular information, not enough data
OriginationsTape$mo_sin_old_il_acct = OriginationsTape$mths_since_recent_bc = OriginationsTape$mths_since_recent_bc_dlq = OriginationsTape$mths_since_recent_inq = OriginationsTape$mths_since_recent_revol_delinq = NULL 
#only one line to delete
OriginationsTape = OriginationsTape[-which(is.na(OriginationsTape$num_rev_accts)),]
#covered in defintion of default and deliquencies 
#OriginationsTape$num_tl_120dpd_2m = OriginationsTape$percent_bc_gt_75 = NULL
#Also want to get rid of some variables that are all the same
#all the same
OriginationsTape$issued_d = OriginationsTape$term = OriginationsTape$policy_code = NULL
#Copies of other columns
OriginationsTape$funded_amnt_inv = OriginationsTape$funded_amnt = OriginationsTape$out_prncp_inv = OriginationsTape$total_pymnt_inv = NULL
#only one under home ownership with any, change to most common
OriginationsTape$home_ownership[OriginationsTape$home_ownership == 'ANY'] <- "MORTGAGE"
#not going to worry about recoveries
OriginationsTape$total_rec_late_fee = OriginationsTape$recoveries = OriginationsTape$collection_recovery_fee = NULL
#offset by 4 so the same
OriginationsTape$fico_range_low = NULL
#verification status, verified and source verified, mean same thing, change to one
OriginationsTape$verification_status[OriginationsTape$verification_status == 'Source Verified'] <- "Verified"
OriginationsTape$verification_status[OriginationsTape$verification_status == 'Not Verified'] <- "Not_Verified"
#lagging indicator
#consumed in total high credit limit
OriginationsTape$last_fico_range_high = OriginationsTape$last_fico_range_low = NULL
OriginationsTape$total_bal_ex_mort = OriginationsTape$total_bc_limit = OriginationsTape$total_il_high_credit_limit = NULL
#captured in other variables
OriginationsTape$mo_sin_old_rev_tl_op = OriginationsTape$mo_sin_rcnt_rev_tl_op = OriginationsTape$mo_sin_rcnt_tl = NULL
OriginationsTape$num_actv_bc_tl = OriginationsTape$num_actv_rev_tl = OriginationsTape$num_bc_sats = OriginationsTape$num_bc_tl = OriginationsTape$num_il_tl = OriginationsTape$num_op_rev_tl = OriginationsTape$num_rev_accts = OriginationsTape$num_rev_tl_bal_gt_0 = OriginationsTape$num_tl_30dpd = OriginationsTape$num_tl_90g_dpd_24m = OriginationsTape$num_tl_op_past_12m = NULL
OriginationsTape$bc_open_to_buy = OriginationsTape$bc_util = NULL
#mostly 6m is mostly blank, 24mths consumed in total open
OriginationsTape$acc_open_past_24mths = OriginationsTape$open_acc_6m = NULL

############ DELETIONS FOR OTHER REASONS ##########################################################################
OriginationsTape$sub_grade = NULL #not performing better than grade alone


####################################################################################################################
### Section 2b #####################################################################################################
##  Feature Engineering ############################################################################################
####################################################################################################################

#Check for different outcomes
unique(OriginationsTape$loan_status)

#Will use last value in order to track result after an event
LastValue = setDT(PaymentHistoryTotal)[, tail(.SD, 1), by = LOAN_ID]

#Lets look at how being In Grace Period or Late leads to default for feature engineering
InGrace = PaymentHistoryTotal[PaymentHistoryTotal$PERIOD_END_LSTAT == "In Grace Period",]
InGrace_Merged = merge(InGrace,LastValue, by='LOAN_ID', all.x=TRUE)
InGrace_Merged = InGrace_Merged[InGrace_Merged$PERIOD_END_LSTAT.y != "In Grace Period"]
#percentage of loans that were In Grace Period at some point and were eventually charged off
sum(InGrace_Merged$PERIOD_END_LSTAT.y == "Charged Off") / length(InGrace_Merged$LOAN_ID)

Late1_Merged = PaymentHistoryTotal[PaymentHistoryTotal$PERIOD_END_LSTAT == "Late (16-30 days)",]
Late1_Merged = merge(Late1_Merged,LastValue, by='LOAN_ID', all.x=TRUE)
Late1_Merged = Late1_Merged[Late1_Merged$PERIOD_END_LSTAT.y != "Late (16-30 days)"]
#percentage of loans that were Late 16-30 days at some point and were eventually charged off
sum(Late1_Merged$PERIOD_END_LSTAT.y == "Charged Off") / length(Late1_Merged$LOAN_ID)

Late2_Merged = PaymentHistoryTotal[PaymentHistoryTotal$PERIOD_END_LSTAT == "Late (31-120 days)",]
Late2_Merged = merge(Late2_Merged,LastValue, by='LOAN_ID', all.x=TRUE)
Late2_Merged = Late2_Merged[Late2_Merged$PERIOD_END_LSTAT.y != "Late (31-120 days)"]
#percentage of loans that were Late 31-120 days at some point and were eventually charged off
sum(Late2_Merged$PERIOD_END_LSTAT.y == "Charged Off") / length(Late2_Merged$LOAN_ID)

#Qualifying these as all bad loans
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Charged Off'] = "Default"
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Late (16-30 days)'] = "Default"
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Late (31-120 days)'] = "Default"
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Default'] = "Default"

#Qualifying these as all good loans
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Current'] = "Good"
OriginationsTape$loan_status[OriginationsTape$loan_status == 'Fully Paid'] = "Good"
OriginationsTape$loan_status[OriginationsTape$loan_status == 'In Grace Period'] = "Good"

#only 2 values now
unique(OriginationsTape$loan_status)

#N/A and Other categories I will flag, the rest I will Not
OriginationsTape$emp_length[OriginationsTape$emp_length == 'n/a'] <- "Flag"
OriginationsTape$emp_length[OriginationsTape$emp_length != 'Flag'] <- "No_Flag"
#same with Purpose
OriginationsTape$purpose[OriginationsTape$purpose != 'other'] <- "No_Flag"
OriginationsTape$purpose[OriginationsTape$purpose == 'other'] <- "Flag"

#Binning for earliest credit line
OriginationsTape$earliest_cr_line <- cut(OriginationsTape$earliest_cr_line, 3, include.lowest=TRUE, labels=c("Early", "Middle", "Recent"))

#maybe try to catch people maxed out on CC
OriginationsTape$percent_bc_gt_75[OriginationsTape$percent_bc_gt_75 > 99] <- "Flag"
OriginationsTape$percent_bc_gt_75[OriginationsTape$percent_bc_gt_75 < 99] <- "No_Flag"
OriginationsTape$percent_bc_gt_75[is.na(OriginationsTape$percent_bc_gt_75)] <- "No_Flag"

#Create own ratios
OriginationsTape[, `:=`("Loan/Inst Ratio", loan_amnt / installment)]
OriginationsTape[,`:=`("Int_Rate_Sq", int_rate^2)]
OriginationsTape[, `:=`("Curr_Acc_Status", num_sats / open_acc)]
OriginationsTape[,`:=`("Hist_Acc_Status", num_accts_ever_120_pd / total_acc)]

####################################################################################################################
### Section 3a #####################################################################################################
##  Modelling - Logit Grade Model ##################################################################################
####################################################################################################################

#Create response column
OriginationsTape[, `:=`(Default, ifelse(loan_status == "Default", 1, 0))]

#average default rate in sample
(default_rate <- OriginationsTape[, mean(Default)])

#LC grade model
out_grade <- glm(Default ~ grade , family = "binomial", data = OriginationsTape)
#summary(out_grade)
#anova(out_grade)

# lift table
phat_grade <- predict(out_grade, type = "response")
# use function ntile from dplyr to create deciles
#deciles_grade <- ntile(phat_grade, n = 10)
#dt_grade <- data.table(deciles = deciles_grade, phat = phat_grade, default = OriginationsTape$Default)
#lift_grade <- dt_grade[, lapply(.SD, mean), by = deciles]
#lift_grade <- lift_grade[, .(deciles, default)]
#lift_grade[, `:=`(mean_response, default/mean(OriginationsTape$Default))]
#setkey(lift_grade, deciles)
#lift_grade

# TPR and FPR function
simple_roc <- function(labels, scores) {
  labels <- labels[order(scores, decreasing = TRUE)]
  data.frame(TPR = cumsum(labels)/sum(labels), FPR = cumsum(!labels)/sum(!labels),
             labels)
}

# Grade Model ROC over default
glm_roc_grade <- simple_roc(OriginationsTape$Default == "1", phat_grade)
TPR_grade <- glm_roc_grade$TPR
FPR_grade <- glm_roc_grade$FPR
data_grade <- data.table(TPR = TPR_grade, FPR = FPR_grade)
# plot the corresponding ROC curve
#ggplot(data_grade, aes(x = FPR, y = TPR)) + geom_line(color = "tomato2", size = 1.2) +
#  ggtitle("ROC Curve for Lending Club Logit Grade Model") + geom_abline(slope = 1,intercept = 0, linetype = "longdash") + theme_bw()


####################################################################################################################
### Section 3b #####################################################################################################
##  Modelling - LASS0 ##############################################################################################
####################################################################################################################

#create dummies for categorical variables
OriginationsTape <- OriginationsTape[,grade_factor:=as.factor(grade)]
OriginationsTape <- OriginationsTape[,emp_length_factor:=as.factor(emp_length)]
OriginationsTape <- OriginationsTape[,home_ownership_factor:=as.factor(home_ownership)]
OriginationsTape <- OriginationsTape[,verification_status_factor:=as.factor(verification_status)]
OriginationsTape <- OriginationsTape[,purpose_factor:=as.factor(purpose)]
#OriginationsTape <- OriginationsTape[,zip_code_factor:=as.factor(zip_code)]
#OriginationsTape <- OriginationsTape[,addr_state_factor:=as.factor(addr_state)]
OriginationsTape <- OriginationsTape[,initial_list_status_factor:=as.factor(initial_list_status)]
OriginationsTape <- OriginationsTape[,earliest_cr_line_factor:=as.factor(earliest_cr_line)]
OriginationsTape <- OriginationsTape[,application_type_factor:=as.factor(application_type)]
OriginationsTape <- OriginationsTape[,annual_inc_joint_factor:=as.factor(annual_inc_joint)]
OriginationsTape <- OriginationsTape[,dti_joint_factor:=as.factor(dti_joint)]
#OriginationsTape <- OriginationsTape[,pymnt_plan_factor:=as.factor(pymnt_plan)]
OriginationsTape <- OriginationsTape[,percent_bc_gt_75_factor:=as.factor(percent_bc_gt_75)]

# create function that makes dummies out of factor in order to use w glmnet --> glm no need
factorToDummy <- function(dtable, var.name){
  stopifnot(is.data.table(dtable))
  stopifnot(var.name %in% names(dtable))
  stopifnot(is.factor(dtable[, get(var.name)]))
  
  dtable[, paste0(var.name,"_",levels(get(var.name)))] -> new.names
  dtable[, (new.names) := transpose(lapply(get(var.name), FUN = function(x){x == levels(get(var.name))})) ]
  
  cat(paste("\nDummies created: ", paste0(new.names, collapse = ", ")))
}

factorToDummy(OriginationsTape,"grade_factor")
factorToDummy(OriginationsTape,"emp_length_factor")
factorToDummy(OriginationsTape,"home_ownership_factor")
factorToDummy(OriginationsTape,"verification_status_factor")
factorToDummy(OriginationsTape,"purpose_factor")
#factorToDummy(OriginationsTape,"zip_code_factor")
#factorToDummy(OriginationsTape,"addr_state_factor")
factorToDummy(OriginationsTape,"initial_list_status_factor")
factorToDummy(OriginationsTape,"earliest_cr_line_factor")
factorToDummy(OriginationsTape,"application_type_factor")
factorToDummy(OriginationsTape,"annual_inc_joint_factor")
factorToDummy(OriginationsTape,"dti_joint_factor")
#factorToDummy(OriginationsTape,"pymnt_plan_factor")
factorToDummy(OriginationsTape,"percent_bc_gt_75_factor")

#took out payment plan factor
regressors <- as.matrix(OriginationsTape[,list(loan_amnt, int_rate, installment,grade_factor_A, grade_factor_B, grade_factor_C, grade_factor_D, grade_factor_E, grade_factor_F, grade_factor_G,emp_length_factor_Flag, emp_length_factor_No_Flag,home_ownership_factor_MORTGAGE, home_ownership_factor_OWN, home_ownership_factor_RENT,annual_inc,verification_status_factor_Not_Verified, verification_status_factor_Verified,purpose_factor_Flag,purpose_factor_No_Flag, dti,delinq_2yrs,earliest_cr_line_factor_Early,earliest_cr_line_factor_Middle,earliest_cr_line_factor_Recent,fico_range_high,inq_last_6mths,open_acc, pub_rec,revol_bal,revol_util,total_acc,initial_list_status_factor_f, initial_list_status_factor_w,collections_12_mths_ex_med,annual_inc_joint_factor_FALSE, annual_inc_joint_factor_TRUE,dti_joint_factor_FALSE, dti_joint_factor_TRUE,acc_now_delinq,tot_coll_amt,tot_cur_bal,total_rev_hi_lim,avg_cur_bal,chargeoff_within_12_mths,mort_acc,num_accts_ever_120_pd,num_sats,pct_tl_nvr_dlq,pub_rec_bankruptcies,tax_liens,tot_hi_cred_lim,Curr_Acc_Status,Int_Rate_Sq,Hist_Acc_Status,`Loan/Inst Ratio`,application_type_factor_Individual,`application_type_factor_Joint App`,percent_bc_gt_75_factor_Flag, percent_bc_gt_75_factor_No_Flag)])
#regressors <- as.matrix(OriginationsTape[,list(installment, grade_factor_A, grade_factor_C,emp_length_factor_Flag,emp_length_factor_No_Flag,pymnt_plan_factor_n, pymnt_plan_factor_y,dti,fico_range_high,inq_last_6mths,avg_cur_bal,mort_acc,num_sats, `Loan/Inst Ratio`)])
outloans_lasso = cv.glmnet(regressors,as.vector(OriginationsTape$Default), family=c("binomial"),alpha = 1, standardize = TRUE)
plot.glmnet(outloans_lasso$glmnet.fit, "lambda", label = TRUE)
plot.cv.glmnet(outloans_lasso)
phat_lasso = predict(outloans_lasso, regressors, s="lambda.1se", type = "response")

#coefficients from 1SE
coefs = coef(outloans_lasso,s="lambda.1se")
coefs

glm2_roc <- simple_roc(OriginationsTape$Default == "1", phat_lasso)
TPR2 <- glm2_roc$TPR
FPR2 <- glm2_roc$FPR
data2 <- data.table(TPR = TPR2, FPR = FPR2)
data2 <- data.table(TPR = TPR2, FPR = FPR2)
data_grade[, `:=`(Model, "Grade")]
data2[, `:=`(Model, "LASSO")]
data <- rbind(data_grade, data2)
# plot the corresponding ROC curve
ggplot(data, aes(x = FPR, y = TPR, color = Model)) + geom_line(size = 1.2) +
  ggtitle("ROC Curve for Lending Club Logit Grade Model and LASSO") + geom_abline(slope = 1,
                                                                                    intercept = 0, linetype = "longdash") + theme_bw()


out = as.data.frame(as.matrix(coefs))
#####selected predictors#####
predictors_outlasso = subset(out,abs(out$`1`) > 0)
new_regs <- matrix(c(50000/10000,1,0,200000/10000,0.05*10,0,1,0,0,0,0,0),nrow = 1)
predict(outloans_lasso, new_regs, s="lambda.1se", type = "response")

glm(Default ~ grade + emp_length + home_ownership + pymnt_plan + dti + fico_range_high + inq_last_6mths + avg_cur_bal + mort_acc + num_sats + `Loan/Inst Ratio` , family = "binomial", data = OriginationsTape)
