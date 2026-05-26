Rest configuration of an object is the *Lagrangian frame* with vectors $\vec{\xi}=(\xi_{1},\xi_{2})$ and the deformed frame is the *Eulerian frame* with vectors $\vec{x}=(x_{1},x_{2})$(resp. undeformed reference configuration and deformed configuration).

> [!definition] Definition: Displacement Field
> The displacement field $\vec{u}(t,\vec{\xi})$ defines how much the (undeformed) reference coordinate $\vec{\xi}$ of a solid body has changed at a time $t$:
> (drawing)
>  $$
>  \vec{x}(t,\vec{\xi})=\vec{\xi}+\vec{u}(t,\vec{\xi})
>  $$

So, the deformation vector $\vec{u}(t,\vec{\xi})$ is really the interesting quantity.


> [!goal] Goal
> Derive a PDE describing the time evolution of $\vec{u}(t,\vec{\xi})$

### Elasticity Models
Pure displacement of the object obviously doesn't involve any strain. We need to look at a measure of how distance preserving the map between the non-deformed and deformed frames are. The Jacobian will be key. An important quantity when modeling elastic solids is the *Lagrangian strain tensor*.

$$	
\varepsilon(t,\vec{\xi}) =  \frac{1}{2}\left\{ \nabla_{\xi}\vec{u}+(\nabla_{\xi}\vec{u})^{\top}+ \nabla_{\xi}\vec{u}^{\top}\cdot \nabla_{\xi}u \right\}
$$

In one dimension, this becomes

$$	
\varepsilon(t,\xi)= \partial_{\xi}u+ \frac{1}{2}(\partial_{\xi}u)^{2}
$$

**Derivation:**
We begin with $\xi\leftrightarrow x(\xi)=\xi+u(\xi)$. Look at the unit vector $\hat{e}_{i}$ times a scalar $\xi_{s}:=\xi+s\hat{e}_{i}$. Then we define

$$	
\begin{align}
x_{s} & = \xi_{s}+u(\xi_{s}) \\
 & = \xi+ s\hat{e}_{i}u(\xi+s\hat{e}_{i})
\end{align}
$$

Now we look at the infinitesimal relative change in length within the frame:

$$	
\begin{align}
d_{\hat{e}_{i}}(\xi) & :=\lim_{ s \to 0 } \frac{\lVert x-x_{s} \rVert -\lVert \xi-\xi_{s} \rVert }{\lVert \xi-\xi_{s} \rVert }  \\
& = \lim_{ s \to 0 } \left( \frac{1}{s}\lVert \xi+u(\xi)-(\xi+s\hat{e}_{i}+u(\xi+s\hat{e}_{i})) \rVert  \right)-1 \\
& = \lim_{ s \to 0 } \left\lVert  \hat{e}_{i}+ \frac{u(\xi+s\hat{e}_{i})-u(\xi)}{s}  \right\rVert  - 1 \\
\end{align}
$$

Notice the directional derivative. Specifically a Gateaux derivative

$$	
\begin{align}
 & = \lVert \hat{e}_{i}+ \nabla u(\xi)\cdot \hat{e}_{i} \rVert -1 \\
 & = (\hat{e}_{i}+ \nabla u(\xi)\cdot \hat{e}_{i}, \space\hat{e}_{i}+\nabla u(\xi)\cdot \hat{e}_{i})_{e^{2}}^{-1/2}-1 \\
  & =\left\{ 1+ \left(\left\{\underbrace{  \nabla u(\xi)+ \nabla u(\xi)^{\top}+\nabla u(\xi)^{\top}\cdot \nabla u(\xi) }_{ \mathclap{ =: 2\varepsilon(\xi) } } \right\}\hat{e}_{i},\space \hat{e}_{i}\right)_{e^{2}} \right\}^{1/2}-1
\end{align} 
$$




The strain tensor is related to the *Piola-Kirchoff stress tensor* by a material that relates strain to stress

$$	
\overbrace{ K(t,\vec{\xi}) }^{ \mathclap{ D\times D\text{ matrix} }} = \underbrace{ C }_{ \mathclap{ \text{material law, typicaly linear} } }(\varepsilon(t,\xi))
$$

So, if $\varepsilon$ is a rank 2 tensor, then $C$ must be a rank 4 tensor. In one dimension, this simplifies greatly:

$$	
K(t,\xi)= c\varepsilon(t,\xi) 
$$

where $c\in \mathbb{R}$.

### Balance of Momentum
Expressing the balance of momentum equation in $(t,\vec{\xi})$ coordinates gives the Lame-Navier Equation

$$	
\partial_{t}^{2}u(t,\vec{\xi})-\text{div}\left\{ (\mathbb{I}+\nabla \vec{u})\cdot K(t,\vec{\xi}) \right\} = \text{forcing}
$$

then in one dimension this becomes

$$	
\partial_{t}^{2}u(t,\xi)-c\partial_{\xi}(\left( 1+\partial_{\xi}u)\left( \partial_{\xi}u+ \frac{1}{2}(\partial_{\xi}u)^{2} \right) \right) = \text{forcing}
$$

In most cases, we assume deformations are small. Assuming that small, ${\lvert \partial_{\xi}u \rvert}\ll1$, we can make two simplifications:

$$	
\begin{gather}
\varepsilon(t,\xi)\approx \partial_{\xi}u(t,\xi), & 1+ \partial_{\xi}u\approx1
\end{gather} 
$$

this then reduces to the wave equation, but with a fancy name
> [!definition] Linear Lame-Navier equation in 1D
>  
>  $$	
>  \partial_{t}^{2}u- c\partial_{\xi}^{2}u=f 
>  $$
>

---

We rewrite the wave equation $\partial_{t}^{2}u-c^{2}\partial_{x}^{2}u=0$ in first-order form:

$$	
\begin{cases}
\partial_{t}+\partial_{x}v=0, \\
\partial_{t}v+c^{2}\partial_{x}u=0
\end{cases} 
$$
> [!remark] 
> Given $(u,v)$ solving the above, and sufficiently regular:
> 
> $$	
> \begin{gather}
>  \partial_{t}^{2}u+\partial_{t}\partial_{x}v=0 \\
>  \Rightarrow \partial_{t}^{2}u+\partial_{x}(\partial_{t}v)=0 \\
>  \Rightarrow \partial_{t}^{2}u-c^{2}\partial_{x}^{2}u=0 \space\square
> \end{gather} 
> $$


Start with $\partial_{t}^{2}u-c^{2}\partial_{x}^{2}u=0$. Choose $v(t,\xi)$ such that $\partial_{t}v+c^{2}\partial_{x}u=0$. Substitute this fact back in to obtain

$$	
\partial_{t}(\partial_{t}u+\partial_{x}v)=0 \implies \partial_{t}u+\partial_{x}v = \text{const} 
$$

so choose $v_{0}$ such that $\partial_{x}v_{0}+\partial_{t}u(0,\xi)=0$.

**Boundary Conditions**
- Do nothing
- Post processing at $t_{n}+\tau=t_{n+1}$
- Dirichlet (left) $U_{0}^{n+1}=u_{man}(x=0,t=t_{n+1})$ (same for $v_{0}$)
- Reflecting: $V_{0}^{n+1}=0$


## Linear with Forcing

$$	
\begin{cases}
\partial_{t}u+\partial_{x}v=0 \\
\partial_{t}v+c^{2}\partial_{x}u=f
\end{cases} 
$$

(we take $f=gx- \frac{1}{2}$) becomes


$$	
\begin{align}
U_{i}^{n+1} & = U_{i}^{n+1} - 2\tau \delta_{h}^{c}(V) \\
V_{i}^{n+1} & = V_{i}^{n-1}-2\tau c^{2} \delta_{h}^{c}(U) + 2\tau\left(f\right)
\end{align} 
$$

Where $\delta_{h}^{c}(F) = \frac{F^{n}_{i+1}-F^{n}_{i-1}}{2h}$.

## Non-Linear with forcing
The first order system is

$$	
\begin{cases}
\partial_{t}u+\partial_{x}v=0 \\
\partial_{t}v+c^{2}(1+\partial_{x}u)\left( 1+ \frac{1}{2}\partial_{x}u \right)\partial_{x}u=f(x,t)
\end{cases} 
$$

which becomes

$$	
\begin{align}
U_{i}^{n+1} & = U_{i}^{n+1} - 2\tau \delta_{h}^{c}(V) \\
V_{i}^{n+1} & =V_{i}^{n-1}+2\tau c^{2}(1+\delta_{h}^{c}(U))\left( 1+ \frac{1}{2}\delta_{h}^{c}(U)\right)\delta_{h}^{c}+2\tau f(x,t)
\end{align} 
$$

## Smoothing
In any of the above discretized explicit equations for a value $F_{i}^{n+1}$ you can add a smoothing term which comes from $\varepsilon \Delta F$. So the expression for $F_{i}^{n+1}$ is:

$$	
\begin{gather}
F_{i}^{n+1}= F_{i}^{n-1}+ 2\tau(\dots) \\
\downarrow \\
F_{i}^{n+1}=F_{i}^{n-1}+2\tau(\dots) + 2\tau\varepsilon\Delta_{h}(F)
\end{gather} 
$$

where $\Delta_{h}(F)$ is the second order difference operator applied to $F$:

$$	
\Delta_{h}(F): = \frac{F_{i+1}^{n}-2F_{i}^{n}+F_{i-1}^{n}}{h^{2}}
$$