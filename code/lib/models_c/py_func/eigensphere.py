import numpy as np
import spiderman
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort

def princomp(A):
	""" performs principal components analysis 
	(PCA) on the n-by-p data matrix A
	Rows of A correspond to observations, columns to variables. 
	
	Returns :  
	coeff :
	is a p-by-p matrix, each column containing coefficients 
	for one principal component.
	score : 
	the principal component scores; that is, the representation 
	of A in the principal component space. Rows of SCORE 
	correspond to observations, columns to components.
	latent : 
	a vector containing the eigenvalues 
	of the covariance matrix of A.
	"""
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = linalg.eig(cov(M))
	idx = argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	score = dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent

def eigensphere(params, t, etc = []):
	"""
	This function creates a model that fits spherical harmonics and uses PCA to fit for orthogonal phase curves

	Parameters
	----------
	t0:		time of conjunction 
	per:		orbital period
	a_abs:		semi-major axis (AU)
	cos(i):	        cosine of the orbital inclination	
	ecc:		eccentricity
	w:		arg of periastron (deg)
	rp:		planet radius (stellar radii)
	a:		semi-major axis (stellar radii)
	p_u1:		planet linear limb darkening parameter
	p_u2:		planet quadratic limb darkening
	T_s:		stellar Teff
	l1:		short wavelength (m)
	l2:		long wavelength (m)
	degree:		maximum degree of harmonic (typically no more than 2)
	la0:		latitudinal offset of coordinate center from substellar point (degrees)
	lo0:		latitudinal offset of coordinate center from substellar point (degrees)
	npoints:        number of phase bins for light curve interpolation
	coeff0-3: four coefficients for eigencurves 
	
	
	**Note: 4 harmonic coefficients are needed for a model of degree 2
	
	Returns
	-------
	This function returns an array of planet/star flux values 
	
	Revisions
	---------
	2016-11-19 	Megan Mansfield
				meganmansfield@uchicago.edu

	"""
	p = spiderman.ModelParams(brightness_model =  'spherical', stellar_model = 'blackbody')
	p.nlayers = 5
	p.t0    	    = params[0]	#I want these to be values, not parameters
	p.per       	    = params[1]
	p.a_abs 	    = params[2]
	p.inc	    = np.arccos(params[3])*180./np.pi
	p.ecc	    = params[4]
	p.w	   	    = params[5]
	p.rp	    	    = params[6]
	p.a	   	    = params[7]
	p.p_u1	    = params[8]
	p.p_u2	    = params[9]
	p.T_s	    = params[10]
	p.l1	   	    = params[11]
	p.l2	    	    = params[12]
	p.degree 	    = int(params[13])
	p.la0	    = params[14]
	p.lo0	    = params[15]
	sph = 0
	ntimes	=	int(params[16])
	degree=int(params[13])

	#stuff from sh_lcs
	#lctimes=np.linspace(np.min(t),np.max(t),ntimes)
	#if np.size(ntimes) == 1:
	#		t= spider_params.t0 + np.linspace(0, spider_params.per,ntimes)  # TEST TIME RESOLUTION
	#	else:
	#		t= ntimes
	#		ntimes = t.size

	phase = (t - p.t0)/p.per
	phase -= np.round(phase)
	phase_bin = np.linspace(phase.min(), phase.max(), ntimes)
	lctimes = phase_bin*p.per + p.t0
	        
	if np.size(sph) == 1:
		
		allLterms = [0] * int(params[13])**2
		allLterms[0] = 1
		p.sph= allLterms# * coeff   # this is l0, so don't need a negative version
		lc=spiderman.web.lightcurve(lctimes,p)
		#lc = p.lightcurve(lctimes)
		# set up size of lc to be able to append full set of LCs
		lc = np.resize(lc,(1,ntimes))
		
		p.sph= [0] * int(params[13])**2
		# set up 2-d array of LCs for all SHs
		for i in range(1,len(p.sph)):
			p.sph[i]= -1#*coeff
			tlc = spiderman.web.lightcurve(lctimes,p)
			#tlc = p.lightcurve(lctimes)
			tlc = np.resize(tlc,(1,ntimes))
			lc = np.append(lc,tlc,axis=0)
			p.sph[i]= 1#*coeff
			tlc = spiderman.web.lightcurve(lctimes,p)
			#tlc = p.lightcurve(lctimes)
			tlc = np.resize(tlc,(1,ntimes))
			lc = np.append(lc,tlc,axis=0)
			p.sph[i]= 0    
	else:        
		# calcualte single lightcurve for single set of spherical harmonic coefficients
		p.sph= sph   # spherical harmonic coefficients
		lc = spiderman.web.lightcurve(lctimes,p)
		#lc = p.lightcurve(lctimes)

	# subtract off stellar flux
	lc = lc-1
	#print(p.sph)
	elc=np.zeros((np.shape(lc)[0],np.shape(lctimes)[0]))
	for i in np.arange(np.shape(lc)[0]):
		elc[i,:] = lc[i,:]

	#  PCA
	ecoeff,escore,elatent = princomp(elc[1:,:].T)
	escore=np.real(escore)

	#construct light curve model
	model = params[17]*elc[0,:] + params[18] + params[19]*escore[0,:] + params[20]*escore[1,:]	#these are the params I want to fit for!!!
	#finallc=np.interp(phase,phase_bin,model)
	#print(np.max(model))

	fcoeffbest=np.zeros_like(ecoeff)
	fcoeffbest[:,0]=params[19]*ecoeff[:,0]
	fcoeffbest[:,1]=params[20]*ecoeff[:,1]

	spheresbest=np.zeros(int(degree**2.))
	for j in range(0,len(fcoeffbest)):
		for i in range(1,int(degree**2.)):
			spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
	spheresbest[0] = params[17]
	#print(spheresbest)
	p.sph=list(spheresbest)
	finallc_bin=spiderman.web.lightcurve(lctimes,p)
	finallc=np.interp(phase,phase_bin,finallc_bin)
	#print(np.max(finallc))
	return finallc









