bernstein = function(Y, V.est, Tr.est, d.est, r, Z, lambda){
  n <- nrow(Y)
  # Basis for estimating mu: Tr.est,V.est,d.est,r
  Bfull.est <- basis(V.est,Tr.est,d.est,r,Z)
  B <- Bfull.est$B # DIM: npts * ntriangle*((d+1)(d+2)/2)
  ind.inside <- Bfull.est$Ind.inside
  Q2 <- Bfull.est$Q2
  K <- Bfull.est$K
  Y <- matrix(Y[,ind.inside],nrow=n)
  #Y <- matrix(Y,nrow=n)
  
  lambda <- as.matrix(lambda)
  t.area = Bfull.est$tria.all # area of the triangles
  
  this.call <- match.call()
  n <- nrow(Y)
  npix <- ncol(Y)
  J <- ncol(Q2)
  
  W <- as.matrix(B%*%Q2)
  WW <- crossprod(W,W)
  rhs <- crossprod(W,t(Y))
  D <- crossprod(t(crossprod(Q2,as.matrix(K))),Q2)
  D <- as.matrix(D)
  
  flag <- (rankMatrix(WW)<J)
  if(!flag){
    Ainv <- chol(WW,pivot=TRUE)
    A <- solve(t(Ainv))
    ADA <- A%*%D%*%t(A)
    eigs <- eigen(ADA)
    Cval <- eigs$values
  }
  
  nl <- length(lambda)
  
  gcv_all <- sapply(lambda,FUN=function(Lam){  # eventually a n*nl matrix
    Dlam <- Lam*D
    lhs <- WW+Dlam
    lhs.inv <- chol2inv(chol(lhs));
    theta <- crossprod(t(lhs.inv),rhs)
    gamma <- crossprod(t(Q2),theta)
    Yhat <- crossprod(t(W),theta)
    res <- t(Y)-Yhat
    sse <- apply(res^2,2,sum)
    if(!flag){
      df <- sum(1/(1+Cval*Lam))
    }
    if(flag){
      Hmtx <- crossprod(t(crossprod(t(W),lhs.inv)),t(W))
      df <- sum(diag(Hmtx))
    }
    gcv <- npix*sse/(npix-df)^2
  })
  gcv_all <- matrix(gcv_all,nrow=n)
  lam.ind <- apply(gcv_all,1,which.min)
  lambdac <- lambda[lam.ind]
  
  theta <- c()
  gamma <- c()
  Yhat <- c()
  df <- c()
  for (i in 1:n){
    lamc.tmp <- lambdac[i]
    Dlam <- lamc.tmp*D
    lhs <- WW+Dlam
    lhs.inv <- chol2inv(chol(lhs));
    rhs.tmp <- as.matrix(rhs[,i],ncol=1)
    theta.tmp <- crossprod(t(lhs.inv),rhs.tmp)
    theta <- cbind(theta,theta.tmp)
    gamma.tmp <- crossprod(t(Q2),theta.tmp) #gamma = Q2*theta this is the spline coefficient
    gamma <- cbind(gamma,gamma.tmp)
    Yhat.tmp <- crossprod(t(W),theta.tmp)
    Yhat <- cbind(Yhat,Yhat.tmp)
    if(!flag){
      df.tmp <- sum(1/(1+Cval*lamc.tmp))
    }
    if(flag){
      Hmtx <- crossprod(t(crossprod(t(W),lhs.inv)),t(W))
      df.tmp <- sum(diag(Hmtx))
    }
    df <- c(df,df.tmp)
  }
  
  return(list(B, gamma, lambdac, t.area))
}
