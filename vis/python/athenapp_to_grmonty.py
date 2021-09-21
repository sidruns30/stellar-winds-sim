def convert_to_gammie(a=0):
  global hslope, uu_gammie,ud_gammie,bu_gammie,bd_gammie
  hslope = 0.3
  #x1gammie = log(r)  r = exp(x1) 
  # dx1/dr = 1/r -> dx1/dr = r 
  #theta = pi*x2 + 0.5*(1-h)*sin(2*pi*x2)
  # dtheta/dx2 = pi + pi* (1-h) * cos(2*pi*x2)

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th*x2/x2
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (th/th)



  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2

  uu_gammie = uu 
  ud_gammie = ud 
  bu_gammie = bu 
  bd_gammie = bd 

  #u^\mu_gammie = u^\nu_athena dx^\mu_gammie/dx^\nu_athena
  #u_\mu_gammie = u_\nu_athena dx^\nu_athena/dx^\mu_gammie

  uu_gammie[1] *= dx1_dr
  uu_gammie[2] *= dx2_dtheta
  bu_gammie[1] *= dx1_dr
  bu_gammie[2] *= dx2_dtheta

  ud_gammie[1] *= dr_dx1 
  ud_gammie[2] *= dtheta_dx2
  bd_gammie[1] *= dr_dx1
  bd_gammie[2] *= dtheta_dx2



  # gprime_mu nu = dx^alpha/dx^mu dx^sigma/dx^nu g_alpha sigma
  # prime^mu^nu = dx^mu/dx^alpha dx^nu/dx^sigma g^alpha sigma

  # so gi_gammie ^mu nu = dx_gammie^mu/dx^sig dx_gammie^nu/dx^alph g^sig alph
  # only nonzero are dx1/dr and dx2/dth

  # gi_gammie00 is unchanged
  # gi_gammie10 = dx1/dr 
  #   g_inv(I00,i) = -(1.0 + 2.0*m*r/sigma);
  #   g_inv(I01,i) = 2.0*m*r/sigma;
  #   g_inv(I11,i) = delta/sigma;
  #   g_inv(I13,i) = a/sigma;
  #   g_inv(I22,i) = 1.0/sigma;
  #   g_inv(I33,i) = 1.0 / (sigma * sin2);
  sigma = r**2.0 + a**2.0 * np.cos(th)**2.0
  m = 1
  gi00 = -(1.0 + 2.0*m*r/sigma)
  gi01 = 2.0*m*r/sigma *dx1_dr
  gi02 = 0
  gi03 = 0
  global v1,v2,v3,B1,B2,B3,gdet_gammie
  v1 = uu_gammie[1] - gi01/gi00 * uu_gammie[0]
  v2 = uu_gammie[2] - gi02/gi00 * uu_gammie[0]
  v3 = uu_gammie[3] - gi03/gi00 * uu_gammie[0]

  B1 = bu_gammie[1]*uu_gammie[0] - bu_gammie[0]*uu_gammie[1]
  B2 = bu_gammie[2]*uu_gammie[0] - bu_gammie[0]*uu_gammie[2]
  B3 = bu_gammie[3]*uu_gammie[0] - bu_gammie[0]*uu_gammie[3]

  gdet_gammie = gdet * dr_dx1 * dtheta_dx2




def gammie_metric(r,th,a=0):
  global gg

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th*x2/x2
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (th/th)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  rfac = r;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  gg  = np.zeros((4,4,nx,ny,nz))
  gg[0][0] = -(1.0 - 2.0*m*r/sigma);
  gg[0][1] = 2.0*m*r/sigma * rfac
  gg[1][0] = gg[0][1] 
  gg[0][3] = -2.0*m*a*r/sigma * sin2
  gg[3][0] = gg[0][3]
  gg[1][1] = (1.0 + 2.0*m*r/sigma) * rfac * rfac
  gg[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2 * rfac
  gg[3][1] = gg[1][3] 
  gg[2][2] = sigma * hfac * hfac
  gg[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def gammie_gcon(r,th,a=0):
  global ggcon

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th*x2/x2
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (th/th)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  delta = r**2 - 2.0*m*r + a**2
  rfac = r;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  ggcon  = np.zeros((4,4,nx,ny,nz))
  ggcon[0][0] = -(1.0 + 2.0*m*r/sigma);
  ggcon[0][1] =  2.0*m*r/sigma /(rfac)
  ggcon[1][0] = ggcon[0][1] 
  ggcon[1][1] = delta/sigma /( rfac * rfac)
  ggcon[1][3] =  a/sigma / rfac
  ggcon[3][1] = ggcon[1][3] 
  ggcon[2][2] = 1.0/sigma /( hfac * hfac )
  ggcon[3][3] = 1.0 / (sigma * sin2)




def gammie_grid():
  global ri,thi,phii,x1_grid,x2_grid,x3_grid
  global igrid_new,jgrid_new,kgrid_new
  global igrid,jgrid,kgrid
  global x1,x2,x3
  x1_grid_faces = np.linspace(log(np.amin(r)),log(np.amax(r)),nx+1)  ##faces
  x2_grid_faces = np.linspace(0,1,ny+1)       ##faces
  x3_grid_faces = np.linspace(0,2.0*pi,nz+1)  ##faces
  x1_grid = ( (x1_grid_faces) + 0.5*np.diff(x1_grid_faces)[0] )[:-1]
  x2_grid = x2_grid_faces + 0.5*np.diff(x2_grid_faces)[0]
  if (nz==1): x3_grid = x3_grid_faces[0] + np.pi
  else: x3_grid =( x3_grid_faces + 0.5*np.diff(x3_grid_faces)[0] )[:-1]
  ri = np.exp(x1_grid)
  thi = np.pi*x2_grid + 0.5*(1.0-hslope)*np.sin(2.0*pi*x2_grid)
  phii = x3_grid

  ri,thi,phii = np.meshgrid(ri,thi,phii,indexing='ij')


  #kgrid,jgrid,igrid = meshgrid(np.arange(0,nz),np.arange(0,ny),np.arange(0,nx),indexing='ij')  I think the next line is correct...need to check
  igrid,jgrid,kgrid = meshgrid(np.arange(0,nx),np.arange(0,ny),np.arange(0,nz),indexing='ij')
  mgrid = igrid + jgrid*nx  + kgrid*nx*ny

  mnew = scipy.interpolate.griddata((r.flatten(),th.flatten(),ph.flatten()),mgrid[:,:,:].flatten(),(ri,thi,phii),method='nearest')


  igrid_new= mod(mod(mnew,ny*nx),nx)
  jgrid_new = mod(mnew,ny*nx)//nx
  kgrid_new = mnew//(ny*nx)


def make_grmonty_dump(fname,a=0):
  convert_to_gammie(a = a)
  gammie_grid()
  gammie_metric(ri,thi,a=a)
  gammie_gcon(ri,thi,a=a)
  dx1 = np.diff(x1_grid)[0]
  dx2 = np.diff(x2_grid)[0]
  if (nz==1): dx3 = 2.*np.pi
  else: dx3 = np.diff(x3_grid)[0]

  Nprim = 8 
  header = [str(t), str(nx), str(ny), str(nz), str(np.amin(x1_grid)-0.5*dx1),str(np.amin(x2_grid)-0.5*dx2),str(np.amin(x3_grid)-0.5*dx3), str(dx1),str(dx2),str(dx3),str(a),str(5./3.),str(np.amin(ri)),str(hslope),str(Nprim)]

  rhoi =rho[igrid_new,jgrid_new,kgrid_new]
  #uui = uu_gammie[:,igrid_new,jgrid_new,kgrid_new]

  v1i = v1[igrid_new,jgrid_new,kgrid_new]
  v2i = v2[igrid_new,jgrid_new,kgrid_new]
  v3i = v3[igrid_new,jgrid_new,kgrid_new]

  qsq = v1i*v1i*gg[1,1] + v2i*v2i*gg[2,2] + v3i*v3i*gg[3,3] + 2.*v1i*v2i*gg[1,2] + 2.*v1i*v3i*gg[1,3] + 2.*v2i*v3i*gg[2,3]
  alpha = 1./np.sqrt(-ggcon[0,0]) 
  beta = 0*uu_gammie
  for l in range(1,4): beta[l] = ggcon[0][l]*alpha*alpha ;

  qsq[qsq<0] = 1e-10
  
  gamma = np.sqrt(1.0 + qsq)
  uui = 0*uu_gammie
  uui[0] = gamma/alpha 
  uui[1] = v1i - gamma*beta[1]/alpha
  uui[2] = v2i - gamma*beta[2]/alpha
  uui[3] = v3i - gamma*beta[3]/alpha

  udi = Lower(uui,gg)



  B1i = B1[igrid_new,jgrid_new,kgrid_new]
  B2i = B2[igrid_new,jgrid_new,kgrid_new]
  B3i = B3[igrid_new,jgrid_new,kgrid_new]

  bui = bu_gammie*0

  bui[0] = B1i*udi[1] + B2i*udi[2] + B3i*udi[3]
  bui[1] = (B1i + bui[0]*uui[1])/uui[0]
  bui[2] = (B2i + bui[0]*uui[2])/uui[0]
  bui[3] = (B3i + bui[0]*uui[3])/uui[0]

  bdi = Lower(bui,gg)



  #udi = ud_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bui = bu_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bdi = bd_gammie[:,igrid_new,jgrid_new,kgrid_new]
  pressi = press[igrid_new,jgrid_new,kgrid_new]
  gdeti = gdet_gammie[igrid_new,jgrid_new,kgrid_new]

  tmp = rhoi*0
  x1_grid,x2_grid,x3_grid = meshgrid(x1_grid,x2_grid,x3_grid,indexing='ij')
  data = [igrid,jgrid,kgrid,x1_grid.astype(float32),x2_grid.astype(float32),x3_grid.astype(float32),ri.astype(float32),thi.astype(float32),phii.astype(float32),
          rhoi.astype(float32),(pressi/(5./3.-1.)).astype(float32),v1i.astype(float32),v2i.astype(float32),v3i.astype(float32),B1i.astype(float32),B2i.astype(float32),
          B3i.astype(float32),(pressi/rhoi**(5./3.)).astype(float32),
          uui[0].astype(float32),uui[1].astype(float32),uui[2].astype(float32),uui[3].astype(float32),udi[0].astype(float32),udi[1].astype(float32),udi[2].astype(float32),
          udi[3].astype(float32), bui[0].astype(float32),bui[1].astype(float32),bui[2].astype(float32),bui[3].astype(float32),bdi[0].astype(float32),bdi[1].astype(float32),
          bdi[2].astype(float32),bdi[3].astype(float32),gdeti.astype(float32)]
  data = np.array(data).astype(float32)
  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()