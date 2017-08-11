/******************************************************************\
 * File:           Rngs.cpp for rng
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           March 4, 2010
\******************************************************************/

#include <math.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include "Rngs.h"
using namespace std;

double MAX_double = numeric_limits<double>::max();

// ===================== Exponential =====================
ExpGen::ExpGen(double a)
{
	if (a<= 0){
		cerr<<"Exponential parameter error: check lambda>0\n";
		exit(-1);
	}
	lambda=a;
	mean=1.0/a;
	name="Exponential";	
	cv=1;
	var=mean*mean;
}

ExpGen::ExpGen(char ch, double m)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (m<= 0){
		cerr<<"Exponential parameter error: check lambda>0\n";
		exit(-1);
	}
	lambda=1.0/m;
	mean=m;
	name="Exponential";	
	cv=1;
	var=mean*mean;
}

double ExpGen::genNext()
{
	return -log(stream.RandU01())/lambda;
}

double ExpGen::getMoment(int n) const
{
	double temp=1;
	for (int i=n;i>0;i--) temp*=(mean*i);
	return	temp;
}

// ===================== Deterministic =====================
DetGen::DetGen(double a)
{
	name="Deterministic";	
	mean=a;
	cv=0;
	var=0;
}

double DetGen::genNext()
{
	return mean;
}

double DetGen::getMoment(int n) const
{
	return	pow(mean, 3);
}

// ===================== Hyper Exponential /2 phase =====================
HyperExp2Gen::HyperExp2Gen(double a, double b, double q)
{
	if (a<= 0||b<=0||q<0||q>1){
		cerr<<"HyperExp2 parameter error: check a,b >0, q in [0,1]\n";
		exit(-1);
	}
	alpha=a;
	beta=b;
	p=q;
	name="HyperExp2phase";	
	mean=p/alpha+(1.0-p)/beta;
	cv=2*( p/(alpha*alpha)+(1.0-p)/(beta*beta) )/(mean*mean)-1;
	var=mean*mean*cv;
}

HyperExp2Gen::HyperExp2Gen(char ch, double m, double coe, double rat)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (m<= 0||coe<=0||rat<0){
		cerr<<"HyperExp2 parameter error: mean, scv >0, ratio>0\n";
		exit(-1);
	}
	mean=m;
	cv=coe;
	r=rat; //=(1-p)/beta:p/alpha
	       //r-->\infty ==> p-->2/cv+1, 3rdM--> a lower bound
	       //r-->\infty ==> p-->0,   3rdM--> infinity
	name="HyperExp2phase";
			
	double a=(cv+1)*(1+r)*(1+r);
	double b=2.0*(r*r-1)-a;
	double Delta=b*b-4*a*2.0;// c=2
	if(Delta<0) {
		cerr<<"Invalid Parameters for HyperExp2Gen!\n";
		exit(-1);
	}
	double delta=sqrt(Delta);

	p=(-b-delta)/(2.0*a);
	alpha=p*(1+r)/mean;
	beta=(1-p)*(1+r)/(r*mean);
	var=mean*mean*cv;
}

double HyperExp2Gen::genNext()
{
	if (stream.RandU01()<p)
		return -log(stream.RandU01())/alpha;
	else
		return -log(stream.RandU01())/beta;
}

double HyperExp2Gen::getMoment(int n) const
{
	double temp1=1, temp2=1;
	for(int i=n; i>0; i--){
		temp1*=(i/alpha);
		temp2*=(i/beta);
	}
	
	return	temp1*p+temp2*(1-p);
}

// ===================== Hyper2Star =====================
Hyper2StarGen::Hyper2StarGen(double u, double q)
{
	if (u<= 0||q<0||q>1){
		cerr<<"Hyper2Star parameter error: check u >0, q in [0,1]\n";
		exit(-1);
	}
	mu=u;
	p=q;
	name="Hyper2Star";	
	mean=p/mu;
	cv=2.0/p-1;
	var=mean*mean*cv;
}

Hyper2StarGen::Hyper2StarGen(char ch, double m, double coe)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (m<= 0||coe<=0){
		cerr<<"Hyper2Star parameter error: mean, scv >0, ratio>0\n";
		exit(-1);
	}
	mean=m;
	cv=coe;
	name="Hyper2Star";	
	p=2.0/(cv+1);
	mu=p/mean;
	var=mean*mean*cv;
}

double Hyper2StarGen::genNext()
{
	if (stream.RandU01()<p)
		return -log(stream.RandU01())/mu;
	else
		return 0;
}

double Hyper2StarGen::getMoment(int n) const
{
	double temp=1;
	for(int i=n; i>0; i--) temp*=(double(i)/mu);
	return	p*temp;
}

// ===================== Erlang =====================
ErlangGen::ErlangGen(double a, int n)
{
	if (a<= 0||n<=0){
		cerr<<"Erlang parameter error: check lambda >0, n> 0";
		exit(-1);
	}
	lambda=a;
	k=n;
	name="Erlang";	
	
	mean=k/lambda;
	cv=1.0/k;
	var=mean*mean*cv;
}

ErlangGen::ErlangGen(char ch, double m, int n)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (m<= 0||n<=0){
		cerr<<"Erlang parameter error: check lambda >0, n> 0";
		exit(-1);
	}
	mean=m;
	k=n;
	name="Erlang";
	
	lambda=k/m;
	cv=1.0/k;
	var=mean*mean*cv;
}

double ErlangGen::genNext()
{
	double z=1;
	for (int i=0;i<k;i++)
		z*=stream.RandU01();
	return -log(z)/lambda;
}

double ErlangGen::getMoment(int n) const
{
	double temp=1;
	for (int i=0; i<n; i++) temp*=(double(k+i)/lambda);
	return	temp;
}

// ===================== Bimodal =====================
BimodalGen::BimodalGen(double _a, double _p)
{
	if (_p<=0||_p>1){
		cerr<<"Bimodal parameters error: p in[0,1]\n";
		exit(-1);
	}
	a=_a;
	p=_p;
	name="Bimodal";
	mean=a*p;	
	cv=1.0/p-1.0;
	var=mean*mean*cv;
}

BimodalGen::BimodalGen(char ch, double m, double coe)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (coe<= 0){
		cerr<<"Bimodal parameters error: check lambda>0\n";
		exit(-1);
	}
	mean=m;
	cv=coe;
	var=mean*mean*cv;
	name="Bimodal";	
	a=(1+cv)*mean;
	p=mean/a;
}

double BimodalGen::genNext()
{
	return (stream.RandU01()<p)? a: 0;
}

double BimodalGen::getMoment(int n) const
{
	return	p*pow(a,n);
}


// ===================== Uniform =====================
UniformGen::UniformGen(double _a, double _b)
{
	if ( _a >= _b){
		cerr<<"Uniform parameters error: a >= b\n";
		exit(-1);
	}
	a=_a;
	b=_b;
	name="Uniform";
	mean=(a+b)/2;
	var=(b-a)*(b-a)/12;	
	cv= var/(mean*mean);
}

UniformGen::UniformGen(char ch, double m, double coe)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (coe<= 0){
		cerr<<"Uniform parameters error: check lambda>0\n";
		exit(-1);
	}
	cerr<<"Uniform construct error; using Uniform(a, b)";
	exit(-1);
}

double UniformGen::genNext()
{
  return a + stream.RandU01()*(b-a);
}

double UniformGen::getMoment(int n) const
{
  return	0;
}


// ===================== Normal =====================
NormalGen::NormalGen(double _mean, double _var)
{
	if (_var<= 0){
		cerr<<"Normal parameter error: check variance>0\n"; cerr<<_mean<<" "<<_var;
		exit(-1);
	}
	var=_var;
	sdv=sqrt(_var);
	mean=_mean;
	name="Normal";	
	if (mean!=0)
		cv=_var/(mean*mean);
	else
		cv=MAX_double;
	z2=0;
}

NormalGen::NormalGen(char ch, double _mean, double _cv)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (cv<= 0){
		cerr<<"Normal parameter error: check cv>0\n";
		exit(-1);
	}
	mean=_mean;
	name="Normal";
	sdv=mean*sqrt(_cv);
	z2=0;
}

double NormalGen::genNext()
{
	double x,y,z1,s;
	
	if (z2!=0.0)		/* use value from previous call */
	{
		z1=z2;
		z2=0.0;
	}
	else{
		do {
			x=2.0*stream.RandU01()-1.0;
			y=2.0*stream.RandU01()-1.0;
			s=x*x+y*y;
		} while (s>=1.0);
		s=sqrt((-2.0*log(s))/s);
		z1=x*s;
		z2=y*s;
	}
	return mean+z1*sdv;
}

double NormalGen::getMoment(int n) const
{
	return	0;
}


void NormalGen::restart() {  stream.ResetStartStream(); z2=0; }

void NormalGen::restartSub() {  stream.ResetStartSubstream(); z2=0; }

void NormalGen::startNextSub() {  stream.ResetNextSubstream(); z2=0; }

// ===================== Log-Normal =====================

LogNormalGen::LogNormalGen(double _mean, double _var)
{
	if (_var<= 0){
		cerr<<"LogNormal parameter error: check variance>0\n";
		exit(-1);
	}
	a_var=_var;
	a_mean=_mean;
	name="LogNormal";	
	if (a_mean!=0)
		a_cv=a_var/(a_mean*a_mean);
	else
		cerr<<"LogNormal parameter error: check mean>0\n";
	
	double t_var=log(1+a_cv);
	double t_mean=log(a_mean)-0.5*t_var;

	if (t_var<= 0){
			cerr<<"Normal parameter error: check variance>0\n"; 
			exit(-1);
		}
	var=t_var;
	sdv=sqrt(var);
	mean=t_mean;
	if (mean!=0)
		cv=var/(mean*mean);
	else
		cv=MAX_double;
	z2=0;

}

LogNormalGen::LogNormalGen(char ch, double _mu, double _sigma2)
{
	if (ch!='N') cerr<<"Please specify a way to construct!\n";
	if (cv<= 0){
		cerr<<"LogNormal parameter error: check sigma^2>0\n";
		exit(-1);
	}
	
	if (_sigma2<= 0){
			cerr<<"Normal parameter error: check variance>0\n"; 
			exit(-1);
		}
	var=_sigma2;
	sdv=sqrt(var);
	mean=_mu;
	if (mean!=0)
		cv=var/(mean*mean);
	else
		cv=MAX_double;
	z2=0;
	
	name="LogNormal";	

	a_mean=exp(mean+var/2.0);
	a_var=a_mean*a_mean*(exp(var-1));
	a_cv=a_var/(a_mean*a_mean);
	

}

double LogNormalGen::genNext()
{
	double temp=NormalGen::genNext();
	return exp(temp);
}

double LogNormalGen::getMoment(int n) const
{
	return	exp(n*mean+double(n*n)*var/2.0);
}

