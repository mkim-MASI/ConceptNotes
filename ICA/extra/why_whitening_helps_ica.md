# Why whitening helps ICA

For ICA, remember that the model assumes:

$$
X=As
\\
A^{-1}X=s
\\
WX=s
$$

where we have $X \in N \times M$, $A^{-1}=W$, and $A \in N \times N$. In this case, $W$ is the “un-mixing matrix” whereas $A$ is the “mixing matrix”. Since we know nothing about $A$, we need to estimate $N^2$ terms in our matrix, which is definitely not ideal.

Now, consider the case where we have $A^TA=I$. If so, then that would mean:

$$
A^T X=A^TAs=Is=s
$$

which would make solving ICA much easier.

How so?

Well, if it is the case that $A^TA=I$, then that means $A$ is **orthogonal**, or each column (and row) is an orthonormal vector (unit length and mutually perpendicular to each other). Thus, while we still need to estimate $N^2$ terms, we have additional constraints on solving for those terms. Specifically, each row/column must have unit length ($N$ constraints) and every column must be perpendicular to every other column ($\frac{N(N-1)}{2}$ constraints), meaning there are now:

$$
N^2-N-\frac{N(N-1)}{2}=\frac{N(N-1)}{2}
$$

degrees of freedom in the model, which is much better. As eigenvalue decomposition (PCA) gives us both orthogonal unit vectors and the variances in the directions of those unit vectors, that is how we achieve the data whitening.