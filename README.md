This repository contains both Python and Matlab implementations of total-variation regularized differentiation (TV-differentiation) code for derivative estimation for discrete-time time series.

The TV-Diff algorithm is an iterative method for finding derivative estimates of noisy data based on the idea that differentiation of a time-series can be cast as an optimization problem where we are trying to find a function $u(t)$ such that the difference between the integral of $u(t)$ and $f(t)$ is minimized. That is:

$$\frac{df(t)}{dt} \approx u(t)$$

$$u(t) = \text{arg min} \frac{1}{2}\int \left(f(t) - \int_0^t u(\tau) d\tau\right)^2dt$$

The equation above describes basic differentiation; we augment this cost function with a total-variation penalty, such that $u(t)$ is discouraged from changing very rapidly. In effect,

$$u(t) = \text{arg min} \left[\frac{1}{2}\int \left(f(t) - \int_0^t u(\tau) d\tau\right)^2dt\right] + \alpha \int_0^t \left|\frac{du(\tau)}{d\tau}\right| d\tau$$

Note that this penalty is parameterized by a weight $\alpha$, which is the main parameter to vary if you are trying out this code. A good starting value is 1, but any given application may require you to go up or down several orders of magnitude.

---

How to use:

---

Matlab: install the regDiff mltbx file contained in releases. regDiff function should now be available in your environment
  `regDiff(data, alpha, alpha2, dx, u0, maxiter, deltacosttol, deltanormtol, ep, minIter, plotflag, diagflag)`

Python: (pip installation soon) Copy the python/regdiff folder into your working directory. In your file, `from regdiff import reg_diff`
  `reg_diff(raw_data,alpha,alpha2=None,dx=1,u0=None,min_iter=5,max_iter=100,delta_cost_tol=1e-6,delta_norm_tol=1e-6,ep=1e-8,diag=False)`

Some function parameters are named differently in the matlab and python implementations; these are noted below.

```Inputs:
data/raw_data                 Vector of data to be differentiated. Required.
alpha                         Regularization parameter.
alpha2                        Regularization parameter for second-order derivative. Skips second-order differentiation if omitted.
dx                            Data spacing. Default is 1.
u0                            Initialization of the derivatives. Default is to use a central difference method.
maxiter/max_iter              Maximum number of iterations to run the solver loop. Default is 100.
deltacosttol/delta_cost_tol   Minimum relative change in the cost. Value below this terminates solver. Default is 1e-6.
deltanormtol/delta_norm_tol   Minimum relative norm of adjustment. Value below this terminates solver. Default is 1e-6.
ep                            Parameter for avoiding division by zero.  Default value is 1e-6.
minIter/min_iter              Parameter mimum number of iterations. Default value is 5.
plotflag (matlab-only)        Flag whether to display plot at each iteration. Default is 0.
diagflag/diag                 Flag whether to display diagnostics at each iteration.  Default is 0/false.
```

```Outputs:
u           Estimate of the regularized derivative of data.
v           Estimate of the regularized second derivative of data (if alpha2 is passed)
```

---

Detailed notes:

First, this problem is globally convex, so an iterative method is sufficient for it.

The summary above has used a continuous-time formulation of the differentiation problem, but the actual target of this library is derivative estimation for discrete time. Our paper (insert link) contains the mathematical formalism for translating continuous to discrete time, so this readme will focus on details of the implementation.

First, we normalize the input to avoid numerical stability errors. We rescale the data so that the time interval is 1, the mean is 0, and the standard deviation is 1. For this reason, subsequent steps all lack any time-scaling.

In order to generate an initial guess for $u[t]$ and perform integrations, we construct some helper matrices, namely the central finite difference and the cumulative trapezoidal integration matrices, using `matrixDiff` and `matrixAntiDiff`. The nth-order anti-differentiation matrix is particularly important; is is the matrix

$$\mathbf{A}^n = (\Delta t)^n\begin{bmatrix}
    0 & 0 & 0 & \cdots & 0\\\
    \frac{1}{2} & \frac{1}{2} & 0 & \cdots & 0\\\
    \frac{1}{2} & 1 & \frac{1}{2} & \cdots & 0\\\
    \vdots & \vdots & \vdots & \ddots & \vdots\\\
    \frac{1}{2} & 1 & 1 & \cdots & \frac{1}{2}
\end{bmatrix}^n$$

WITH THE FIRST ROW OMITTED. By omitting the first row we avoid singularities in the inversion.

We also construct the simple difference matrix

$$D = \begin{bmatrix}-1 & 1& 0& 0& \cdots& 0 \\\ 0&-1&1&0&\cdots&0\\\ 0&0&-1&1&\cdots&0 \\\ \vdots&\vdots&\vdots&\vdots&\ddots&\vdots\\\0&0&0&0&\cdots&1\end{bmatrix}$$

Now, the cost for the optimization problem of calculating $u[t]$, the nth derivative of a function $f[t]$ is:

$$\sum \frac{1}{2}\left(A^n u[t] - f[t]\right)^2 + \alpha \sum \left|Du[t]\right|$$

The gradient of this cost is 

$$g = \left(A^n\right)^T\left(A^nu[t]-f[t]\right) + \alpha L\left(u[t]\right)u[t]$$

Where $L(u[t])$ is the discrete-time analogue of $-\nabla \cdot \frac{Du[t]}{\left|Du[t]\right|}$. This is a complicated quantity to think about, especially given the nature of the absolute value function. However, if we say $|x| \approx \sqrt{x^2+\epsilon}$, it can be calculated as

$$L\left(u[t]\right) = D^TE\left(Du[t]\right)D$$

Where

$$E\left(Du[t]\right) = \begin{bmatrix}
    \frac{1}{\sqrt{Du[1]}+\epsilon} & 0 & 0 & \cdots & 0\\\
    0 & \frac{1}{\sqrt{Du[2]}+\epsilon} & 0 & \cdots & 0\\\
    0 & 0 & \frac{1}{\sqrt{Du[3]}+\epsilon} & \cdots & 0\\\
    \vdots & \vdots & \vdots & \ddots & \vdots\\\
    0 & 0 & 0 & \cdots & \frac{1}{\sqrt{Du[n]}+\epsilon}
\end{bmatrix}$$

Having the gradient, we can calculate the Hessian

$$H = \left(A^n\right)^T\left(A^n\right) + \alpha \left(L\left(Du[t]\right) + L'\left(Du[t]\right)[t]\right)$$

The very last term, $L'\left(Du[t]\right)[t]$, is both small and difficult to calculate, so we omit it from out calculation of the Hessian; therefore,

$$H \approx \left(A^n\right)^T\left(A^n\right) + \alpha L\left(Du[t]\right)$$

We can then perform Newton's method with the Hessian and gradient as:

$$u_{k+1} = u_k - H^{-1}g$$

---

Worked example:

For a sinusoidal input $f(t) = sin(t)$ for $t = 1:7$, and using $\alpha = 0.0001$:

`f = 0.8415    0.9093    0.1411   -0.7568   -0.9589   -0.2794    0.6570`

After preprocessing, this is transformaed into the zero-mean vector:

`f = 1.0762    1.1719    0.0875   -1.1800   -1.4653   -0.5061    0.8157`

With the initial guess for the derivative:

`u = 0.6858   -0.4943   -1.1759   -0.7764    0.3369    1.1405    1.5031`

We pre-calculate $\left(A^n\right)^Tf[t]$, which is constant, as

`DFb = -3.7665   -7.5809   -7.1345   -5.5121   -3.1134   -1.0515   -0.1302`

In each iteration, we then calculate $E\left(Du[t]\right)$ and $L\left(Du[t]\right)$:
```
En =
    0.8474         0         0         0         0         0
         0    1.4671         0         0         0         0
         0         0    2.5030         0         0         0
         0         0         0    0.8982         0         0
         0         0         0         0    1.2445         0
         0         0         0         0         0    2.7576
```
```
Lu =
    0.8474   -0.8474         0         0         0         0         0
   -0.8474    2.3145   -1.4671         0         0         0         0
         0   -1.4671    3.9701   -2.5030         0         0         0
         0         0   -2.5030    3.4012   -0.8982         0         0
         0         0         0   -0.8982    2.1426   -1.2445         0
         0         0         0         0   -1.2445    4.0021   -2.7576
         0         0         0         0         0   -2.7576    2.7576
```

This allows us to find the gradient and Hessian:

`gn = 1.0838    2.1673    2.0425    1.6478    1.0744    0.5785    0.1929`
```
Hn =
    1.5001    2.7499    2.2500    1.7500    1.2500    0.7500    0.2500
    2.7499    5.2502    4.4999    3.5000    2.5000    1.5000    0.5000
    2.2500    4.4999    4.2504    3.4997    2.5000    1.5000    0.5000
    1.7500    3.5000    3.4997    3.2503    2.4999    1.5000    0.5000
    1.2500    2.5000    2.5000    2.4999    2.2502    1.4999    0.5000
    0.7500    1.5000    1.5000    1.5000    1.4999    1.2504    0.4997
    0.2500    0.5000    0.5000    0.5000    0.5000    0.4997    0.2503
```

The update $s[t]$ is then $H^{-1}g$:

`s = -0.2567    0.2580    0.2383    0.3457   -0.2160   -0.2233    0.2225`

Yielding the new $u[t]$:

`u = 0.9425   -0.7523   -1.4142   -1.1221    0.5529    1.3638    1.2806`

Iteration then repeats from the step in which $E\left(Du[t]\right)$ is calculated, until either the maximum number of iteration is reached, the percent change in cost difference reaches a lower limit, or the relative step size reaches a lower limit. Results are then rescaled to match the initial input and returned. Here,

`u = 0.7109   -0.5758   -0.9599   -0.8353    0.4300    0.9302    0.9418`

For camparison, the analytical derivative is

`u = 0.5403   -0.4161   -0.9900   -0.6536    0.2837    0.9602    0.7539`

While the central difference derivative estimate is

`u = 0.4858   -0.3502   -0.8330   -0.5500    0.2387    0.8080    1.0648`

Notably, over this small timeframe, the error of TV-differentiation is basically the same as for central differencing. However, over the time span 0 to 100, the error of TV-differentiaion is nearly half that of central differencing.
