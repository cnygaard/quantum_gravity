# Reproducible CAS validation of quantum-gravity-proposal-v12.tex
# Run via the sagemath-mcp server (evaluate_sage), which runs Sage WITHOUT the preparser:
#   use ** for powers, QQ()/RealField() for exact/precise numbers, M.metric('g') (name required).
# Companion report: theory/MATH_VALIDATION.md. QFIM sector: `cd theory/sage && python3 -m qfim.run_all`.

# ============================ A. GR / CURVATURE (SageManifolds) ============================

# A15 / A3-anchor: round S^3 Ricci scalar = 6/L^2
M=Manifold(3,'S3'); X=M.chart('ch th ph'); ch,th,ph=X[:]; g=M.metric('g'); Lv=var('Lv',domain='positive')
g[0,0]=Lv**2; g[1,1]=(Lv*sin(ch))**2; g[2,2]=(Lv*sin(ch)*sin(th))**2
assert (g.ricci_scalar().expr().simplify_full()-6/Lv**2)==0            # PASS

# A13/A14 + dS: Schwarzschild vacuum; SdS & dS  R_mn=(3/L^2)g, R=12/L^2
rs=var('rs',domain='positive')
def check_sss(name, Afun, Lam=None):
    Mn=Manifold(4,name); Xn=Mn.chart('t r th ph'); t,r,th,ph=Xn[:]; A=Afun(r); gg=Mn.metric('g')
    gg[0,0]=-A; gg[1,1]=1/A; gg[2,2]=r**2; gg[3,3]=r**2*sin(th)**2; Ric=gg.ricci()
    if Lam is None:
        return all(Ric[i,j].expr().simplify_full()==0 for i in range(4) for j in range(4))
    return all((Ric[i,j].expr()-Lam*gg[i,j].expr()).simplify_full()==0 for i in range(4) for j in range(4))
assert check_sss('Schw', lambda r:1-rs/r)                              # A13 vacuum
assert check_sss('dS',   lambda r:1-r**2/Lv**2, 3/Lv**2)              # dS  Lambda=3/L^2
assert check_sss('SdS',  lambda r:1-rs/r-r**2/Lv**2, 3/Lv**2)        # A14 SdS

# A3: perturbed de Sitter R^(3) on S^3.  gamma_ij = g^{S3}_ij + beta L^2 (E_ij/alpha),
#     beta = gamma_0 pi;  expect  R^(3) = 6/L^2 + beta(20 sin^2 chi - 15)/L^2 + O(beta^2).
M3=Manifold(3,'M3'); X3=M3.chart('c3 t3 p3'); c3,t3,p3=X3[:]; bb=var('bb')  # bb=beta=gamma_0 pi
g3=M3.metric('g3')
g3[0,0]=Lv**2*(1+bb*(QQ(1)/2-2*sin(c3)**2))
g3[1,1]=Lv**2*sin(c3)**2*(1+bb*QQ(1)/2)
g3[2,2]=Lv**2*sin(c3)**2*sin(t3)**2*(1+bb*QQ(1)/2)
R3lin=g3.ricci_scalar().expr().series(bb,2).truncate()
assert bool((R3lin-(6/Lv**2+bb*(20*sin(c3)**2-15)/Lv**2)).simplify_full()==0)   # A3 PASS

# A1: Fisher metric R = -6 lp^2/L^2  (H^3).  G^F=[2/sigma^2] g^dS, sigma^2=2 lp^2 (1-r^2/L^2).
lp=var('lp',domain='positive')
MF=Manifold(3,'MF'); XF=MF.chart('rr te pe'); rr,te,pe=XF[:]
ffac=1-rr**2/Lv**2; pf=1/(lp**2*ffac); gF=MF.metric('gF')
gF[0,0]=pf/ffac; gF[1,1]=pf*rr**2; gF[2,2]=pf*rr**2*sin(te)**2
assert bool((gF.ricci_scalar().expr()+6*lp**2/Lv**2).simplify_full()==0)        # A1 PASS

# A2 SdS Hessian/box/EST; A4 rho_m/Omega_m; A5 EST tracelessness (d=4); A6 galactic EST;
# A7/A9 FLRW EST; A8 w=1/3; A10/A11 SSS Ricci -> AB=1 -> Schwarzschild; A16 Kerr R=0.
# (documented in MATH_VALIDATION.md; each verified True.)

# ============================ B. ANALYTIC IDENTITIES ============================
zeta=var('zeta'); t=var('t',domain='positive')
Lp=2*zeta/t+1/zeta-coth(zeta); Lpp=2/t-1/zeta**2+1/sinh(zeta)**2
assert (diff(Lp,zeta)-Lpp).simplify_full()==0                         # B4  L''=dL'/dz
assert limit(Lpp,zeta=0)==2/t-1/3                                     # B4  flat limit

chi=var('chi'); assert (sin(2*chi)/sin(chi)-2*cos(chi)).simplify_full()==0   # B7

m,xi=var('m xi'); mu2=m**2+(6*xi-1)/Lv**2                             # B14 de Sitter EE limits
assert (mu2.subs(xi==QQ(1)/6)-m**2).simplify_full()==0
assert (mu2.subs({xi:0,m:0})+1/Lv**2).simplify_full()==0

# B1/B2/B3 numeric: Z1/Z0 -> 3/2t-1/4 ,  G=4/3 Z1/Z0 -> 2/t-1/3  (exact to O(e^-4pi^2/t))
RF=RealField(120)
def Zrat(tv,N=500):
    tv=RF(tv); Z0=sum(RF(n)**2*exp(-tv*(RF(n)**2-1)/4) for n in range(1,N))
    Z1=sum(RF(n)**2*(RF(n)**2-1)/4*exp(-tv*(RF(n)**2-1)/4) for n in range(1,N)); return Z0,Z1/Z0
for tv in [0.1,0.02]:
    Z0,r=Zrat(tv); assert abs(float(r-(RF(3)/(2*RF(tv))-RF(1)/4)))<1e-30

# ============================ C. NUMERIC ANCHORS ============================
g0=RF('0.274'); Om=RF('0.315'); Ob=RF('0.0493'); pi_=RF(pi)
sincf=lambda x: RF(1) if RF(x)==0 else sin(RF(x))/RF(x)
assert abs(float(pi_/(2*g0))-5.73)<0.01           # C1
assert abs(float(8*pi_*g0)-6.89)<0.01             # C7  sigma_0^2
assert abs(float(10*pi_*g0/3)-2.87)<0.01          # C9  (>1 tension, in-text flag)
w=var('w'); root=find_root(w-Ob*(1+(pi/(2*g0))*sin(sqrt(w))/sqrt(w)),0.1,0.9)
# C4: root = 0.3172  ->  paper's 0.317 correct; the 0.316 at L1864/1975 is a mis-quote
assert abs(root-0.3172)<1e-3

print("All assertions passed. See MATH_VALIDATION.md for the full per-equation table.")
