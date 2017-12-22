install.packages("car")
library("car")

data("Anscombe")
?Anscombe
head(Anscombe)

pairs(Anscombe)
model1 = lm(education~income+young+urban,data = Anscombe)
summary(model1)

library("rjags")

mod_string = "model{
  for (i in 1:length(education)){
    education[i]~dnorm(mu[i],prec)
    mu[i] = b0+b[1]*income[i]+b[2]*young[i]+b[3]*urban[i]
  }
  b0 ~ dnorm(0.0,1.0/1.0e6)
  for (i in 1:3) {
    b[i] ~ dnorm(0.0,1.0/1.0e6)
  }
  prec ~ dgamma(1.0/2.0, 1.0*1500.0/2.0)
  sig2 = 1.0/prec
  sig = sqrt(sig2)
}"

data_jags = as.list(Anscombe)
params1 = c("b0","b", "prec")

inits1 = function() {
  inits = list("b0"=rnorm(1,0.0,100.0),"b"=rnorm(3,0.0,100.0), "prec"=rgamma(1,1.0,1.0))
}

mod1 = jags.model(textConnection(mod_string), data=data_jags, inits=inits1, n.chains=3)
update(mod1, 1000) # burn-in

mod1_sim = coda.samples(model=mod1,
                        variable.names=params1,
                        n.iter=5000)

mod1_csim = do.call(rbind, mod1_sim) # combine multiple chains
gelman.diag(mod1_sim)
#autocorr.diag(mod1_sim)
#autocorr.plot(mod1_sim)
dic.samples(mod1,n.iter=1e5)

model2 = lm(education~income+young,data = Anscombe)

mod_string = "model{
  for (i in 1:length(education)){
education[i]~dnorm(mu[i],prec)
mu[i] = b0+b[1]*income[i]+b[2]*young[i]
}
b0 ~ dnorm(0.0,1.0/1.0e6)
for (i in 1:2) {
b[i] ~ dnorm(0.0,1.0/1.0e6)
}
prec ~ dgamma(1.0/2.0, 1.0*1500.0/2.0)
sig2 = 1.0/prec
sig = sqrt(sig2)
}"

data_jags = as.list(Anscombe)
params2 = c("b0","b", "prec")

inits2 = function() {
  inits = list("b0"=rnorm(1,0.0,100.0),"b"=rnorm(2,0.0,100.0), "prec"=rgamma(1,1.0,1.0))
}

mod2 = jags.model(textConnection(mod_string), data=data_jags, inits=inits2, n.chains=3)
update(mod2, 1000) # burn-in

mod2_sim = coda.samples(model=mod2,
                        variable.names=params2,
                        n.iter=5000)

dic.samples(mod2,n.iter=1e5)

model3 = lm(education~income+young+income*young,data = Anscombe)

mod_string = "model{
for (i in 1:length(education)){
education[i]~dnorm(mu[i],prec)
mu[i] = b0+b[1]*income[i]+b[2]*young[i]+b[3]*income[i]*young[i]
}
b0 ~ dnorm(0.0,1.0/1.0e6)
for (i in 1:3) {
b[i] ~ dnorm(0.0,1.0/1.0e6)
}
prec ~ dgamma(1.0/2.0, 1.0*1500.0/2.0)
sig2 = 1.0/prec
sig = sqrt(sig2)
}"

data_jags = as.list(Anscombe)
params3 = c("b0","b", "prec")

inits3 = function() {
  inits = list("b0"=rnorm(1,0.0,100.0),"b"=rnorm(3,0.0,100.0), "prec"=rgamma(1,1.0,1.0))
}

mod3 = jags.model(textConnection(mod_string), data=data_jags, inits=inits3, n.chains=3)
update(mod2, 1000) # burn-in

mod3_sim = coda.samples(model=mod3,
                        variable.names=params3,
                        n.iter=5000)

dic.samples(mod3,n.iter=1e5)


