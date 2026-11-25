"""Reproduce the stiffening-trap LRT versus driven-dynamics comparison.

Requirements: Python 3, NumPy, Matplotlib.
Run: python compare_stiffening_lrt.py

All variables are dimensionless: EPS=delta_lambda/lambda_0,
R=tau/tau_R, tau_R=gamma/(2*lambda_0), z=beta*w/EPS.
The exact reference solves the full driven Feynman-Kac equation
numerically; the LRT reconstruction uses the equilibrium transform.
Finite FFT bandwidth can cause small ringing near the support boundary.
The output NPZ files retain the raw inversion (without clipping).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS=1e-5
RS=[.01,.1,1,10,100]
NS=60000
STEPS=1500
rng=np.random.default_rng(72124)
OUT=Path('stiffening_lrt_data')
OUT.mkdir(exist_ok=True)
summary=[]

def eq_transform(s,R):
    a=R/2
    d=np.sqrt(a*a+R*s)
    # Stable exponentially factored form of the closed transform.
    v=(a*a+d*d)/(2*a*d)
    return np.exp((a-d)/2)/np.sqrt((1+v+(1-v)*np.exp(-2*d))/2)

def driven_transform(s,R,n):
    # Backward Feynman-Kac Riccati equation, integrated by exact
    # constant-midpoint transfer matrices; limit n->infinity is exact.
    h=np.zeros_like(s,dtype=complex)
    logQ=np.zeros_like(h)
    for j in range(n):
        a=R/2*(1+EPS*(1-(j+.5)/n))
        d=np.sqrt(a*a+R*s)
        e=np.exp(-2*d/n)
        plus=1+e; minus=1-e
        den=(plus+minus*(a+R*h)/d)/2
        h=(plus*h+minus*(s-a*h)/d)/(2*den)
        logQ+=d/n+np.log(den)
    return np.exp((R/2*(1+EPS/2)-logQ-np.log1p(h))/2)

def simulate(R):
    x=rng.standard_normal(NS); xd=x.copy()
    acc=x*x/2; accd=acc.copy()
    dt=1/STEPS
    ae=R/2
    pe=np.exp(-ae*dt); qe=np.sqrt(-np.expm1(-2*ae*dt))
    for j in range(STEPS):
        a=ae*(1+EPS*(j+.5)*dt)
        p=np.exp(-a*dt); q=np.sqrt(R*(-np.expm1(-2*a*dt))/(2*a))
        noise=rng.standard_normal(NS)
        x=pe*x+qe*noise; xd=p*xd+q*noise
        wt=.5 if j==STEPS-1 else 1
        acc+=wt*x*x; accd+=wt*xd*xd
    return acc/(2*STEPS),accd/(2*STEPS)

fig,axes=plt.subplots(5,2,figsize=(12,15),layout='constrained')
fig.suptitle(r'Stiffening trap: $\delta\lambda/\lambda_0$ = '+f'{EPS:g}, linear protocol',fontsize=17)
for row,R in enumerate(RS):
    # Fourier inversion on x=z+1/2; x is near the positive quadratic functional.
    length=32 if R<=1 else (8 if R==10 else 4)
    nfft=2**(21 if R==.01 else 18 if R==.1 else 16 if R==1 else 14)
    dx=length/nfft
    x=np.arange(nfft)*dx; z=x-.5
    omega=2*np.pi*np.fft.rfftfreq(nfft,dx); s=1j*omega
    L=eq_transform(s,R)
    # The LRT shift is W/epsilon in the scaled coordinate.
    W=EPS**2/2*(R+np.expm1(-R))/R**2
    shiftL=W/EPS; shiftE=.5-np.log1p(EPS)/(2*EPS)
    FL=L*np.exp(-s*shiftL)
    # Compute only frequencies with meaningful transform magnitude.
    ix=np.flatnonzero(np.abs(L)>1e-13)[-1]+1
    coarse=driven_transform(s[:ix],R,32)
    fine=driven_transform(s[:ix],R,64)
    # Richardson removes leading midpoint error, and compare against 128.
    prev=(4*fine-coarse)/3
    if R<=.1:
        extrap=prev
        prev=fine
    else:
        finer=driven_transform(s[:ix],R,128)
        extrap=(4*finer-fine)/3
    FE=FL.copy()
    FE[:ix]=extrap*np.exp(-s[:ix]*shiftE)
    old=FL.copy(); old[:ix]=prev*np.exp(-s[:ix]*shiftE)
    pL=np.fft.irfft(FL,nfft)/dx
    pE=np.fft.irfft(FE,nfft)/dx
    diff=np.fft.irfft(FE-FL,nfft)/dx
    err=np.fft.irfft(FE-old,nfft)/dx
    tv=.5*np.sum(abs(diff))*dx
    conv=.5*np.sum(abs(err))*dx
    cdfL=np.cumsum(pL)*dx; cdfE=np.cumsum(pE)*dx
    negL=np.interp(.5-dx/2,x,cdfL); negE=np.interp(.5-dx/2,x,cdfE)
    simL,simE=simulate(R)
    simL+=shiftL-.5; simE+=shiftE-.5
    # Plot to central 99.8% quantile, with more bins near lower edge.
    hi=x[np.searchsorted(cdfE,.998)]-.5
    lo=x[np.searchsorted(cdfE,.00005)]-.5
    if R<=1: lo=-.5
    v=np.linspace(0,1,100)
    edges=lo+(hi-lo)*v**(2 if R<=1 else 1)
    aa,bb=axes[row]
    counts,_=np.histogram(simE,bins=edges)
    aa.stairs(counts/(NS*np.diff(edges)),edges,fill=True,color='#9db6cc',alpha=.55,label='Driven simulation')
    mask=(z>=lo)&(z<=hi)
    aa.plot(z[mask],pE[mask],color='#172c48',lw=2,label='Exact dynamics (FK)')
    aa.plot(z[mask],pL[mask],color='#d14b34',lw=1.4,ls='--',label='LRT reconstruction')
    aa.set_xlim(lo,hi); aa.set_ylim(bottom=0); aa.set_ylabel(r'$p(z)$')
    aa.set_title(f'R = {R:g}  |  density',loc='left',fontsize=11)
    aa.grid(alpha=.18)
    if row==0: aa.legend(fontsize=8,loc='upper right')
    peak=pE.max()
    maxdiff=np.max(abs(diff))/peak
    summary.append([R, W, tv, maxdiff, negL, negE, conv,
                    simL.mean(), simE.mean(), simL.var(), simE.var(),
                    np.mean(simL<0), np.mean(simE<0)])
    # Save complete numerical grids and both paired Monte Carlo samples.
    np.savez_compressed(OUT/f'R_{R:g}.npz', R=R, epsilon=EPS,
                        z=z, p_lrt=pL, p_exact=pE, difference=diff,
                        simulation_lrt=simL, simulation_driven=simE,
                        histogram_edges=edges, histogram_counts=counts)
    bb.plot(z[mask],diff[mask]/(EPS*peak),color='#6b448f',lw=1.5)
    bb.axhline(0,color='.5',lw=.7); bb.grid(alpha=.18)
    bb.set_xlim(lo, min(hi,-.5+max(.025,2*R)) if R<=1 else hi); bb.set_ylabel(r'$(p_{\rm exact}-p_{\rm LRT})/(\epsilon p_{\rm max})$')
    bb.set_title(f'R = {R:g}  |  amplified difference'+(' (zoom)' if R<=1 else ''),loc='left',fontsize=11)
    aa.set_xlabel(r'$z=\beta w/\epsilon$'); bb.set_xlabel(r'$z=\beta w/\epsilon$')
    print(f'R={R:g} Nfreq={ix} tail={abs(L[-1]):.3g} TV={tv:.9g} convergence={conv:.3g} PnegL={negL:.9f} PnegE={negE:.9f} maxdiff={np.max(abs(diff))/peak:.5g} minpdf={pE.min():.2g} sim_var={simE.var():.6g} exact_varL={2*W/EPS**2:.6g} paired_mean_diff={np.mean(simE-simL):.3g}',flush=True)
fig.savefig('stiffening_lrt_exact_comparison.png',dpi=190)
np.savetxt(OUT/'summary.csv',np.asarray(summary),delimiter=',',fmt='%.12g',
           header='R,beta_Wirr,TV,max_relative_to_peak,Pnegative_LRT,Pnegative_exact,FK_refinement_TV,simulation_LRT_mean_z,simulation_driven_mean_z,simulation_LRT_var_z,simulation_driven_var_z,simulation_LRT_Pnegative,simulation_driven_Pnegative',comments='')
plt.close(fig)
