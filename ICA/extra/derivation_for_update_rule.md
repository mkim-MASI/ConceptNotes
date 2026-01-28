# Derivation for Update Rule

As mentioned, we are trying to find a $w$ by maximizing the expected value of the function $G(...)$:

$$
M=E[G(w^TZ)]
$$

One way of finding a maximum is to look for where the derivative of $M$ is equal to zero. However, notice that $M$ can become infinitely large by increasing the scale of $w$ (e.g. changing $w=[1, 1, 1]$ to $w=[100, 100, 100]$)..

Thus, we constrain this optimization by enforcing $w$ to be a unit vector, or $||w||^2 = w^T w = 1$. Normally, solving this constrained optimization problem would be difficult, but we can instead use Lagrange multipliers to turn the constrained optimization problem into an unconstrained one that we can solve through gradient descent (see example picture of Lagrangian):

![Screenshot 2026-01-27 at 12.45.46 PM.png](Derivation%20for%20Update%20Rule/Screenshot_2026-01-27_at_12.45.46_PM.png)

 We define the Lagrangian $\mathcal{L}$ as:

$$
\mathcal{L}(w,\beta) = E[G(w^TZ)] - \frac{\beta}{2}\left(w^Tw-1  \right)
$$

(We use $\beta/2$ just to make the derivative cleaner) Now, we find the gradient with respect to $w$ and set to zero:

$$
\begin{aligned}
\nabla_w\mathcal{L}(w,\beta) = \frac{d}{dw}\left[E[G(w^TZ)]\right] - \frac{d}{dw}\left[\frac{\beta}{2}\left(w^Tw-1  \right) \right]
\\
...
\\
\frac{d}{dw}\left[E[G(w^TZ)]\right] = E[ZG'(w^TZ)]
\\
...
\\
\frac{d}{dw}\left[\frac{\beta}{2}\left(w^Tw-1  \right)\right] = \left( \frac{\beta}{2} \right)(2w)=\beta w
\\...\\
\nabla_w\mathcal{L}(w,\beta) = E[ZG'(w^TZ)] - \beta w
\\
\bold{F(w)=E[ZG'(w^TZ)] - \beta w = 0 }
\end{aligned}
$$

So this gives us our equation $F(w)$ (we renamed $\nabla_w\mathcal{L}(w,\beta)$ ) that we wish to find the roots of (where it equals zero). We can do so using the Newton-Raphson method of gradient descent:

$$
\begin{aligned}
x^{n+1}=x^n - \frac{f(x)}{f'(x)}
\\OR\\
x^+ \leftarrow x - \frac{f(x)}{f'(x)}
\\\text{applied to our case, this becomes}\\
w^+ \leftarrow w - \frac{F(x)}{F'(x)}
\end{aligned}
$$

However, since $w$ is a vector, we need to use the matrix of partial derivatives (i.e. the Jacobian) in order to calculate $F'(x)=\nabla_wF(x)$. The partial derivative $\nabla_wF(x)$ is:

$$
\begin{aligned}
\nabla_wF(x) = \frac{d}{dw} \left[ E[ZG'(w^TZ)] - \beta w \right]
\\
= E[ZG''(w^TZ)Z^T] - \beta
\\
= E[ZZ^TG''(w^TZ)] - \beta, \text{since Z is a scalar matrix}
\end{aligned}
$$

At this point, we make another simplifying assumption: we assume that $ZZ^T$ and the term $G''(w^TZ)$ are uncorrelated. This allows us to split the expectation:

$$
E[ZZ^TG''(w^TZ)] - \beta \approx (E[ZZ^T]\cdot E[G''(w^TZ)]) - \beta
$$

And since we whitened our data, $E[ZZ^T]=1$, and thus we have:

$$
\nabla_wF(x) = E[G''(w^TZ)] - \beta
$$

Substituting both $F$ and $\nabla_w F$ into the Newton-Raphson equation:

$$
w^{n+1} = w^n - \frac{E[ZG'(w^TZ)] - \beta w^n}{E[G''(w^TZ)] - \beta}
$$

Now, replacing $G'(w^TZ)$ with $g$ (and similarly $G''(w^TZ)$ with $g'$:

$$
\begin{aligned}
w^{n+1} = w^n - \frac{E[Zg]-\beta w^n}{E[g'] - \beta}
\\
(E[g'] - \beta)w^{n+1} = w^n(E[g'] - \beta) - E[Zg] - \beta w^n
\\
(E[g'] - \beta)w^{n+1} = w^nE[g'] - w^n\beta - E[Zg] - \beta w
\\
(E[g'] - \beta)w^{n+1} = w^nE[g'] - E[Zg] 
\end{aligned}
$$

And finally, since we ONLY care about the direction, not the magnitude ($w$ is rescaled to be a unit vector), the term on the left side disappears:

$$
\begin{aligned}
w^{n+1} \leftarrow w^n E[g'] - E[Zg]
\\
w^{n+1} \leftarrow w^n E[G''(w^TZ)]-E[ZG'(w^TZ)]
\end{aligned}
$$

Since negation does not matter either, we end up with:

$$
w^{n+1} \leftarrow E[ZG'(w^TZ)] - w^n E[G''(w^TZ)]
$$
