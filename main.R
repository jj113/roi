library(BPST); library(Triangulation); library(MFPCA); library(readr); library(grplasso); library(StatMeasures)
library(pROC);library(parallel);library(foreach);library(doMC); library(grpreg); library(survAUC)
library(mvtnorm)
source('ancillary.R')

n = 1500; n.tr = 1000; n.test = 500
n1=40; n2=40; 
u1=seq(0,1,length.out=n1)
v1=seq(0,1,length.out=n2)
uu=rep(u1,each=n2)
vv=rep(v1,times=n1)
Z=as.matrix(cbind(uu,vv))

load("Brain.V2.rda");load("Brain.Tr2.rda")
V.est<-Brain.V2; Tr.est<-Brain.Tr2;
TriPlot(V.est, Tr.est)
d.est<-2; r<-1;

Bfull.est <- basis(V.est,Tr.est,d.est,r,Z)
B.save <- t(as.matrix(Bfull.est$B)) # DIM: npts * ntriangle*((d+1)(d+2)/2)
ind.inside <- Bfull.est$Ind.inside
Q2 = Bfull.est$Q2

# -- inner product --- #
org.idx = c(1:n1*n2)
desMat = matrix(0, nrow = n1*n2, ncol = 480)

for(zz in 1:ncol(desMat)){
  desMat[ind.inside,zz] = B.save[zz,]
}

B <- aperm(array(desMat, c(n1, n2, 
                           480)), c(3, 1, 2)) #reorder, 3rd element is the first dimension etc.
ind.outside = matrix( rep( rep(NA, nrow(Z)), n), nrow = n )
mya = array(dim = c(1, n1, n2))
for(i in 1){
  mya[i,,] = matrix(ind.outside[i,], n1, n2)
}

g <- funData(list(c(1:n1), c(1:n2)), mya) 

W = MFPCA:::calcBasisIntegrals(B, 2, g@argvals) 


Z1 = read_rds('Z1')
est = bernstein(Y = Z1, V.est = V.est, Tr.est = Tr.est, 
                d.est = d.est, r = r, Z = Z, lambda = c(0, 1e-4, 1e-2, 0.1, 1, 10, 100))

phi = t(as.matrix(est[[1]]))
cijk = t(est[[2]])
cijk = cijk %*% W

data.tr = read_rds('train_data')
data.test = read_rds('test_data')

idx.grp = rep(c(1:80), each = 6)

tr.cijk = matrix(data.tr$cijk, ncol = 480)

tune = cv.grpsurv(X = tr.cijk,  y = cbind(data.tr$time, data.tr$event), group = idx.grp, 
                  nfolds = 10)

fit = grpsurv(X = tr.cijk, y = cbind(data.tr$time, data.tr$event), group = idx.grp, 
              penalty = 'grLasso', lambda = tune$lambda.min)

test.cijk = matrix(data.test$cijk, ncol = 480)

train.prob = predict(fit, X = tr.cijk)
surv.prob = predict(fit, X = test.cijk)

Surv.train = Surv(data.tr$time, data.tr$event)
Surv.test = Surv(data.test$time, data.test$event)

times = seq(0, 15, by = 0.1)

# cumulative AUC 
AUC.est = AUC.uno(Surv.train, Surv.test, surv.prob, times)$iauc

AUC.est


