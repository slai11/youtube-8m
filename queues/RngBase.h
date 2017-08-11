/******************************************************************\
 * File:           RngBase.cpp for rng
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           13 July 2007
\******************************************************************/

#ifndef RNGBASE_H_
#define RNGBASE_H_

#include <string>
#include "RngStream.h"

//! Abstract base class.
/**
 * The class RngBase serves as an abstract base class, 
 * for implementations of various random number generator class.
 */

class RngBase
{
public:
	//! A destructor.
	virtual ~RngBase(){};
    //! Pure virtual function for inherited class to implement.
	virtual double genNext() =0;
	//! Increase Precision.
	/*!
	 * Inform the generator to generate numbers with extended precision (53
     * bits if machine follows IEEE 754 standard).
     */
	void increasePrecision() {  stream.IncreasedPrecis(true); }
	//! Restart the generator from the begining.
	virtual void restart() {  stream.ResetStartStream(); }
	//! Reset the generator, to the beginning of the substream.
	virtual void restartSub() {  stream.ResetStartSubstream(); }
	//! Start the generator from the beginning of the next substream.
	virtual void startNextSub() {  stream.ResetNextSubstream(); }
	//! Set antithetic.
	/*!
	 * Inform the generator to generate antithetic variates.
     */
	void setAntithetic() {  stream.SetAntithetic(true); }	
	//! Get the name of the generator.
	const std::string getName() const { return name; }
	//! Get the mean of the distribution.
	double getMean() const { return mean; }
	//! Get the variance of the distribution.
	double getVar() const { return var; }
	//! Get the coefficient of variance of the distribution.
	/*!
	 * It is \f${variance}/{mean^2}\f$
     */
	double getCV() const { return cv; }
	
	//! Get the \f$n\f$th moment of the distribution.
	virtual double getMoment(int n) const =0;
	
protected:

	RngStream stream;

	std::string name;

	double mean, cv, var;

};

#endif /*RNGBASE_H_*/
