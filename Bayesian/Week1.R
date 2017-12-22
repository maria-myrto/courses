theta = rbeta(n=10000, shape1=5.0, shape2=3.0)

mean(theta/(1-theta))

ind = theta/(1-theta) > 1.0 
mean(ind)

phi = rnorm(n=10000,0,1)

quantile(x=phi,probs=0.3)
qnorm(p=0.3,0,1)

sqrt(5.2/5000)