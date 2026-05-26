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
\begin{gather*}
F_{i}^{n+1}= F_{i}^{n-1}+ 2\tau(\dots) \\
\downarrow \\
F_{i}^{n+1}=F_{i}^{n-1}+2\tau(\dots) + \varepsilon\Delta_{h}(F)
\end{gather*} 
$$

where $\Delta_{h}(F)$ is the second order difference operator applied to $F$:

$$	
\Delta_{h}(F): = \frac{F_{i+1}^{n}-2F_{i}^{n}+F_{i-1}^{n}}{h^{2}}
$$