
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c PROGRAM: halofit

c The `halofit' code models the nonlinear evolution of cold matter 
c cosmological power spectra. The full details of the way in which 
c this is done are presented in Smith et al. (2002), MNRAS, ?, ?. 
c
c The code `halofit' was written by R. E. Smith & J. A. Peacock. 
c
c Last edited 8/5/2002.

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      program haloformula

      implicit none 

      real*8 rn,plin,pq,ph,pnl,rk,p_cdm,aexp,z
      real*8 rknl,rneff,rncur,d1,d2,rad,sig,rknl_int_pow
      real*8 om_m,om_v,om_b,h,p_index,gams,sig8,amp
      real*8 om_m0,om_v0,omega_m,omega_v,grow,grow0,gg
      real*8 rn_pd,rn_cdm,pnl_pd,rklin,f_pd
      real*8 f1a,f2a,f3a,f4a,f1b,f2b,f3b,f4b,frac,f1,f2,f3,f4
      real*8 diff,xlogr1,xlogr2,rmid

      integer i,j,k,ndat

      common/cospar/om_m,om_v,om_b,h,p_index,gams,sig8,amp
      
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      write(*,*) '****************************'
      write(*,*) '*** running halofit code ***'
      write(*,*) '****************************'

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      write(*,*)
      write(*,*) '*** CDM fitting ***'

c This section demonstrates how cdm spectra are generated

c cosmological parameters

      write(*,*) 
      write(*,*) 'input cosmological parameters:'
      write(*,*) 
      write(*,*) 'input z=0 matter density (om_m0)'
      read(*,*) om_m0
      write(*,*) 'input z=0 vacuum density (om_v0)'
      read(*,*) om_v0
      write(*,*) 'input z=0 normalisation of cdm power spectrum (sig8)'
      read(*,*) sig8
      write(*,*) 'input shape parameter of cdm power spectrum (gams)'
      read(*,*) gams
      write(*,*) 'input required redshift (z)'
      read(*,*) z

      aexp=1./(1.+z)            ! expansion factor

c calculate matter density, vacuum density at desired redshift

      om_m=omega_m(aexp,om_m0,om_v0) 
      om_v=omega_v(aexp,om_m0,om_v0)

c calculate the amplitude of the power spectrum at desired redshift 
c using linear growth factors (gg given by Caroll, Press & Turner 1992, ARAA, 30, 499)

      grow=gg(om_m,om_v) 
      grow0=gg(om_m0,om_v0)       
      amp=aexp*grow/grow0

c calculate nonlinear wavenumber (rknl), effective spectral index (rneff) and 
c curvature (rncur) of the power spectrum at the desired redshift, using method 
c described in Smith et al (2002).

      write(*,*) 'computing effective spectral quantities:'

      xlogr1=-2.0
      xlogr2=3.5
 10   rmid=(xlogr2+xlogr1)/2.0
      rmid=10**rmid
      call wint(rmid,sig,d1,d2)
      diff=sig-1.0
      if (abs(diff).le.0.001) then
         rknl=1./rmid
         rneff=-3-d1
         rncur=-d2                  
      elseif (diff.gt.0.001) then
         xlogr1=log10(rmid)
         goto 10
      elseif (diff.lt.-0.001) then
         xlogr2=log10(rmid)
         goto 10
      endif

      write(*,20) 'rknl [h/Mpc] =',rknl,'rneff=',rneff, 'rncur=',rncur
 20   format(a14,f12.6,2x,a6,f12.6,2x,a6,f12.6)

c now calculate power spectra for a logarithmic range of wavenumbers (rk)

      write(*,*) 'press return to compute halofit nonlinear power'
      read(*,*)

      ndat=100
      open(1,file='halofit.dat',status='unknown')
      write(1,*) ndat
      
      do i=1,ndat

         rk=-2.0+4.0*(i-1)/(ndat-1.)
         rk=10**rk

c linear power spectrum !! Remember => plin = k^3 * P(k) * constant
c constant = 4*pi*V/(2*pi)^3 

         plin=amp*amp*p_cdm(rk,gams,sig8)

c calculate nonlinear power according to halofit: pnl = pq + ph,
c where pq represents the quasi-linear (halo-halo) power and 
c where ph is represents the self-correlation halo term. 
 
         call halofit(rk,rneff,rncur,rknl,plin,pnl,pq,ph)   ! halo fitting formula 
         
         write(1,*) rk,plin,pnl,pq,ph

         write(*,*) 'halofit:',i,rk,pq,ph,pnl

      enddo
      
      close(1)

c comparison with Peacock & Dodds (1996)

      write(*,*) 'press return to compute PD96 nonlinear power'
      read(*,*)

      ndat=100
      open(1,file='pd96.dat',status='unknown')
      write(1,*) ndat
      
      do i=1,ndat          

c linear wavenumbers and power

         rklin=-2.0+4.0*(i-1)/(ndat-1.)
         rklin=10**rklin
         plin=amp*amp*p_cdm(rklin,gams,sig8)

c effective spectral index: rn_pd=dlogP(k/2)/dlogk          

         rn_pd=rn_cdm(rklin,gams,sig8)

c nonlinear power from linear power

         pnl_pd=f_pd(plin,rn_pd)

c scaling for nonlinear wavenumber

         rk=rklin*(1+pnl_pd)**(1./3.)

         write(*,*) 'PD96:',rklin,plin,rknl,pnl_pd

         write(1,*) rklin,plin,rk,pnl_pd

      enddo

      close(1)

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      write(*,*) 'press return to compute scale free-spectra'
      read(*,*) 

c *** POWER LAW SPECTRA fitting ***

      write(*,*)
      write(*,*) '*** POWER LAW SPECTRA fitting *** '
      write(*,*) 

c scale-free spectra      

      om_m=1.0
      om_v=0.0

      do j=1,4
      
      if (j.eq.1) rn=-2.0
      if (j.eq.2) rn=-1.5
      if (j.eq.3) rn=-1.0
      if (j.eq.4) rn=0.0
      
c determine nonlinear scale rknl 

      rknl=rknl_int_pow(rn)   
      rncur=0.0

      write(*,*) 'rknl/rk0=',rknl

c nonlinear power

      write(*,*) 'halofit scale free power'

      do i=1,100 
          rk=-1.+3.*(i-1)/99.
         plin=(10**rk)**(rn+3)
         call halofit(10**rk,rn,rncur,rknl,plin,pnl,pq,ph)
         write(*,*) 'halofit:',i,rn,10**rk,plin,pnl
      enddo

      write(*,*) 

      do i=1,100                ! peacock & dodds (1996)
         rklin=-1.+3.*(i-1)/99.
         rklin=10.**rklin
         plin=rklin**(3+rn)
         pnl_pd=f_pd(plin,rn)
         rk=rklin*(1+pnl_pd)**(1./3.)
         write(*,*) 'PD96:',i,rn,rklin,plin,rk,pnl_pd
      enddo

      enddo

      stop
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c halo model nonlinear fitting formula as described in 
c Appendix C of Smith et al. (2002)

      subroutine halofit(rk,rn,rncur,rknl,plin,pnl,pq,ph)

      implicit none

      real*8 gam,a,amod,b,c,xmu,xnu,alpha,beta,f1,f2,f3,f4
      real*8 rk,rn,plin,pnl,pq,ph
      real*8 om_m,om_v,om_b,h,p_index,gams,sig8,amp
      real*8 rknl,y,rncur,p_cdm
      real*8 f1a,f2a,f3a,f4a,f1b,f2b,f3b,f4b,frac

      common/cospar/om_m,om_v,om_b,h,p_index,gams,sig8,amp

      gam=0.86485+0.2989*rn+0.1631*rncur
      a=1.4861+1.83693*rn+1.67618*rn*rn+0.7940*rn*rn*rn+
     &0.1670756*rn*rn*rn*rn-0.620695*rncur
      a=10**a      
      b=10**(0.9463+0.9466*rn+0.3084*rn*rn-0.940*rncur)
      c=10**(-0.2807+0.6669*rn+0.3214*rn*rn-0.0793*rncur)
      xmu=10**(-3.54419+0.19086*rn)
      xnu=10**(0.95897+1.2857*rn)
      alpha=1.38848+0.3701*rn-0.1452*rn*rn
      beta=0.8291+0.9854*rn+0.3400*rn**2

      if(abs(1-om_m).gt.0.01) then ! omega evolution 
         f1a=om_m**(-0.0732)
         f2a=om_m**(-0.1423)
         f3a=om_m**(0.0725)
         f1b=om_m**(-0.0307)
         f2b=om_m**(-0.0585)
         f3b=om_m**(0.0743)       
         frac=om_v/(1.-om_m) 
         f1=frac*f1b + (1-frac)*f1a
         f2=frac*f2b + (1-frac)*f2a
         f3=frac*f3b + (1-frac)*f3a
      else         
         f1=1.0
         f2=1.
         f3=1.
      endif

      y=(rk/rknl)

      ph=a*y**(f1*3)/(1+b*y**(f2)+(f3*c*y)**(3-gam))
      ph=ph/(1+xmu*y**(-1)+xnu*y**(-2))
      pq=plin*(1+plin)**beta/(1+plin*alpha)*exp(-y/4.0-y**2/8.0)

      pnl=pq+ph

      return
      end       

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c Bond & Efstathiou (1984) approximation to the linear CDM power spectrum 

      function p_cdm(rk,gams,sig8)
      implicit none 
      real*8 p_cdm,rk,gams,sig8,p_index,rkeff,q,q8,tk,tk8
      p_index=1.
      rkeff=0.172+0.011*log(gams/0.36)*log(gams/0.36)
      q=1.e-20 + rk/gams
      q8=1.e-20 + rkeff/gams
      tk=1/(1+(6.4*q+(3.0*q)**1.5+(1.7*q)**2)**1.13)**(1/1.13)
      tk8=1/(1+(6.4*q8+(3.0*q8)**1.5+(1.7*q8)**2)**1.13)**(1/1.13)
      p_cdm=sig8*sig8*((q/q8)**(3.+p_index))*tk*tk/tk8/tk8
      return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c The subroutine wint, finds the effective spectral quantities
c rknl, rneff & rncur. This it does by calculating the radius of 
c the Gaussian filter at which the variance is unity = rknl.
c rneff is defined as the first derivative of the variance, calculated 
c at the nonlinear wavenumber and similarly the rncur is the second
c derivative at the nonlinear wavenumber. 

      subroutine wint(r,sig,d1,d2)
      
      implicit none
      real*8 sum1,sum2,sum3,t,y,x,w1,w2,w3
      real*8 rk, p_cdm, r, sig, d1,d2
      real*8 om_m,om_v,om_b,h,p_index,gams,sig8,amp
      integer i,nint

      common/cospar/om_m,om_v,om_b,h,p_index,gams,sig8,amp

      nint=3000
      sum1=0.d0
      sum2=0.d0
      sum3=0.d0
      do i=1,nint
         t=(float(i)-0.5)/float(nint)
         y=-1.d0+1.d0/t
         rk=y
         d2=amp*amp*p_cdm(rk,gams,sig8)
         x=y*r
         w1=exp(-x*x)
         w2=2*x*x*w1
         w3=4*x*x*(1-x*x)*w1
         sum1=sum1+w1*d2/y/t/t
         sum2=sum2+w2*d2/y/t/t
         sum3=sum3+w3*d2/y/t/t
      enddo
      sum1=sum1/float(nint)
      sum2=sum2/float(nint)
      sum3=sum3/float(nint)
      sig=sqrt(sum1)
      d1=-sum2/sum1
      d2=-sum2*sum2/sum1/sum1 - sum3/sum1
      
      return
      end
      
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c nonlinear wavenumber for power-law scale-free models

      function rknl_int_pow(rn)
      implicit none
      real*8 rknl_int_pow,rn,a,gammln,arg
      integer ifail
      arg=(3+rn)/2.
      a=exp(gammln(arg))
      rknl_int_pow=(0.5*(a))**(-1/(3+rn))
      return
      end

c The GAMMLN function was taken from 'numerical recipes in Fortran'

      FUNCTION gammln(xx)
      INTEGER j
      DOUBLE PRECISION gammln,xx,ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp
      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,
     *24.01409824083091d0,-1.231739572450155d0,.1208650973866179d-2,
     *-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+log(stp*ser/x)
      return
      END

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c effective spectral index used in peacock & dodds (1996)

      function rn_cdm(rk,gams,sig8)
      implicit none 
      real*8 rn_cdm,rk,gams,sig8,p_cdm,yplus,y
      y=p_cdm(rk/2.,gams,sig8)
      yplus=p_cdm(rk*1.01/2.,gams,sig8)
      rn_cdm=-3.+log(yplus/y)*100.5      
      return
      end
      
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c peacock & dodds (1996) nonlinear fitting formula

      function f_pd(y,rn)
      implicit none
      real*8 om_m,om_v,om_b,h,p_index,gams,sig8,amp
      real*8 f_pd,y,rn,g,a,b,alp,bet,vir
      common/cospar/om_m,om_v,om_b,h,p_index,gams,sig8,amp
      g=(5./2.)*om_m/(om_m**(4./7.)-om_v+(1+om_m/2.)*(1+om_v/70.))
      a=0.482*(1.+rn/3.)**(-0.947)
      b=0.226*(1.+rn/3.)**(-1.778)
      alp=3.310*(1.+rn/3.)**(-0.244)
      bet=0.862*(1.+rn/3.)**(-0.287)
      vir=11.55*(1.+rn/3.)**(-0.423)
      f_pd=y * ( (1.+ b*y*bet + (a*y)**(alp*bet)) /
     & (1.+ ((a*y)**alp*g*g*g/vir/y**0.5)**bet ) )**(1./bet) 
      return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c evolution of omega matter with expansion factor

      function omega_m(aa,om_m0,om_v0)
      implicit none
      real*8 omega_m,omega_t,om_m0,om_v0,aa
      omega_t=1.0+(om_m0+om_v0-1.0)/(1-om_m0-om_v0+om_v0*aa*aa+om_m0/aa)
      omega_m=omega_t*om_m0/(om_m0+om_v0*aa*aa*aa)
      return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c evolution of omega lambda with expansion factor

      function omega_v(aa,om_m0,om_v0)      
      implicit none
      real*8 aa,omega_v,om_m0,om_v0,omega_t
      omega_t=1.0+(om_m0+om_v0-1.0)/(1-om_m0-om_v0+om_v0*aa*aa+om_m0/aa)
      omega_v=omega_t*om_v0/(om_v0+om_m0/aa/aa/aa)
      return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c growth factor for linear fluctuations 

      function gg(om_m,om_v)        
      implicit none
      real*8 gg,om_m,om_v
      gg=2.5*om_m/(om_m**(4./7.)-om_v+(1d0+om_m/2.)*(1.+om_v/70.))
      return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
