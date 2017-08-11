/******************************************************************\
 * File:           Rngs.h for rng
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           13 July 2007
\******************************************************************/

#ifndef RNGS_H_
#define RNGS_H_

#include "RngBase.h"

// ===================== Exponential =====================
//! Exponential random number generator. 
/**
 * Generates random numbers according to Exponential distribution with parameter
 *  \f$\lambda\f$. 
 * The distribution function is \f$1-\exp(-\lambda x)\f$.
 */
class ExpGen: public RngBase
{
public:
	//! Default constructor.
	ExpGen(){};
	//! Standard constructor which sets standard parameter \f$\lambda=a\f$.
	ExpGen(double a);
	//! Non-standard constructor which sets not-standard parameter.
	/**
	 * \param ch : a char specifying the way to construct.
	 * \param m : mean
	 */	
	ExpGen(char ch, double m);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double lambda;
};

// ===================== Deterministic =====================
//! Deterministic. 
/**
 * There is nothing special, it just produce the same specified number.
 */
class DetGen: public RngBase
{
public:
	//! Constructor.
	DetGen(double a);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
};

// ===================== Hyper Exponential /2 phase =====================
//! Hyper Exponential with 2 phase random number generator. 
/**
 * Generates random numbers according to 2 phase Hyper Exponential distribution 
 * standard parameters \f$\alpha, \beta, p\f$. 
 * The distribution function is \f$1-p\exp(-\alpha x)-(1-p)\exp(-\beta x)\f$.
 */
class HyperExp2Gen: public RngBase
{
public:
	//! Default constructor.
	HyperExp2Gen(){};
	//! Standard constructor which sets standard parameters \f$\alpha=a,\beta=b,p=p\f$.
	HyperExp2Gen(double a, double b, double p);
	//! Non-standard constructor which sets non-standard parameters.
	/**
	 * \paran ch : a char specifying the way to construct.
	 * \param m : mean;
	 * \param coe : coefficient of variation;
	 * \param rat : ratio \f$rat=\frac{1-p}{\beta}:\frac{p}{\alpha}\f$.
	 */	
	HyperExp2Gen(char ch, double m, double coe, double rat);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double alpha,beta,p;
	double r;
};

// ===================== Hyper2Star =====================
//! Hyper* 2 phase random number generator. 
/**
 * Generates random numbers according to Hyper* 2 phase distribution with standard 
 * parameters \f$\mu\f$ and \f$p\f$. 
 * The distribution function is \f$1-p\exp(-\mu x)\f$.
 */
class Hyper2StarGen: public RngBase
{
public:
	//! Default constructor.
	Hyper2StarGen(){};
	//! Constructor which sets standard parameters \f$\mu=mu,p=p\f$.
	Hyper2StarGen(double mu, double p);
	//! Construct generator from non-standard parameters.
	/**
	 * \paran ch : a char specifying the way to construct.
	 * \param m : mean;
	 * \param coe : coefficient of variation.
	 */	
	Hyper2StarGen(char ch, double m, double coe);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double mu,p;

};

// ===================== Erlang =====================
//! Erlang random number generator. 
/**
 * Generates random numbers according to Erlang distribution with standard 
 * parameters \f$\mu\f$ and \f$p\f$. 
 * The distribution function is \f$\gamma(k, \lambda x)/(k-1)!\f$, 
 * where \f$\gamma(\cdot,\cdot)\f$ is the lower incomplete gamma function.
 */
class ErlangGen: public RngBase
{
public:
	//! Default constructor.
	ErlangGen(){};
	//! Constructor which sets standard parameters \f$\lambda=a\f$ and phase number \f$k=n\f$.
	ErlangGen(double a, int n);
	//! Construct generator from non-standard parameters.
	/**
	 * \paran ch : a char specifying the way to construct.
	 * \param m : mean;
	 * \param n : phase number.
	 */	
	ErlangGen(char ch, double m, int n);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double lambda, k;

};

// ===================== Bimodal =====================
//! Bimodal random number generator. 
/**
 * Generates random numbers with probability \f$p\f$ to be a constant \f$a\f$, 
 * and with probability  \f$1-p\f$ to be \f$0\f$.
 */
class BimodalGen: public RngBase
{
public:
	//! Default constructor.
	BimodalGen(){};
	//! Constructor which sets standard parameter \f$\lambda=a\f$.
	BimodalGen(double a, double p);
	//! Construct generator from non-standard parameter.
	/**
	 * \paran ch : a char specifying the way to construct.
	 * \param m : mean;
	 * \param m : cv.
	 */	
	BimodalGen(char ch, double m, double coe);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double a, p;
};

// ===================== Uniform =====================
//! Uniform random number generator. 
/**
 * Generates random numbers  that are uniform between \f$a\f$ and \f$b\f$.
 */
class UniformGen: public RngBase
{
public:
	//! Default constructor.
	UniformGen(){};
	//! Constructor which sets standard parameter \f$\lambda=a\f$.
	UniformGen(double , double );
	//! Construct generator from non-standard parameter.
	/**
	 * \paran ch : a char specifying the way to construct.
	 * \param m : mean;
	 * \param m : cv.
	 */	
	UniformGen(char ch, double m, double coe);
	//! Function that generates next random number.
	virtual double genNext();
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double a, b;
};

// ===================== Normal =====================
//! Normal random number generator. 
/**
 * Generates random numbers according to Normal distribution with mean
 *  \f$\mu\f$ and variance \f$\sigma^2\f$. 
 */
class NormalGen: public RngBase
{
public:
	//! Default constructor.
	NormalGen(){};
	//! Standard constructor sets \f$\mu=\_mean\f$ and \f$\sigma^2=\_var\f$.
	NormalGen(double _mean, double _var);
	//! Non-standard constructor which sets not-standard parameter.
	/**
	 * \param ch : a char specifying the way to construct.
	 * \param _mean : mean
	 * \param _cv : coefficient of variation
	 */	
	NormalGen(char ch, double _mean, double _cv);	
	//! Function that generates next random number.
	
	virtual double genNext();
	//! Get the third moment of the distribution.
	virtual double getMoment(int n) const;
	
	//! Restart the generator from the begining.
	void restart();
	//! Reset the generator, to the beginning of the substream.
	void restartSub();
	//! Start the generator from the beginning of the next substream.
	void startNextSub();
	
	double getSdv(){ return sdv; }
	
protected:
	double sdv;
	double z2;

};


// ===================== Log-Normal =====================
//! Log-Normal random number generator. 
/**
 * Generates random numbers according to Log-Normal distribution with mean
 *  \f$\mu\f$ and variance \f$\sigma^2\f$. 
 */
class LogNormalGen: public NormalGen
{
public:
	//! Default constructor.
	LogNormalGen(){};
	//! Standard constructor sets \f$\mu=\_mean\f$ and \f$\sigma^2=\_var\f$.
	LogNormalGen(double _mean, double _var);
	
	//! Non-standard constructor which sets not-standard parameter.
	/**
	 * \param ch : a char specifying the way to construct.
	 * \param _mu : mean of the normal part
	 * \param _sigma2 : variance of normal part
	 */	
	LogNormalGen(char ch, double _mu, double _sigma2);	
	
	//! Function that generates next random number.
	virtual double genNext();

	double getMean() const { return a_mean; }

	double getVar() const { return a_var; }

	double getCV() const { return a_cv; }

	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const;
	
protected:
	double a_mean, a_var, a_cv;

};


#endif /*RNGS_H_*/
