/******************************************************************\
 * File:           SampleStat.cpp for PS
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           12 Feb 2008
\******************************************************************/

#include "SampleStat.h"
#include <limits>
#include <math.h>
#include <stdio.h>

using namespace std;

double max_double = numeric_limits<double>::max();
double min_double = numeric_limits<double>::min();

void SampleStat::reset()
{
	n=0;
	x=0.0; x2=0.0;
	minValue=min_double;
	maxValue=max_double;
}

void SampleStat::update(double s)
{
	n += 1;
    x += s;
    x2 += (s * s);
    if ( minValue > s) minValue = s;
    if ( maxValue < s) maxValue = s;
}


double SampleStat::mean()
{
	if ( n > 0)
	   return (x/n);
    else 
	   return 0;
}

double SampleStat::var()
{
	if ( n > 1) 
		return( (x2-((x*x)/double(n))) / (n-1.0) );
    else 
		return 0;
}

double SampleStat::stdDev()
{
	if ( n <= 0 || this->var() <= 0)
		return 0;
    else 
		return( (double) sqrt( this->var() ) );
}

// t-distribution: given p-value and degrees of freedom, return t-value
// adapted from Peizer & Pratt JASA, vol63, p1416

double SampleStat::tval(double p, int df) 
{
  double t;
  int positive = p >= 0.5;
  p = (positive)? 1.0 - p : p;
  if (p <= 0.0 || df <= 0)
    t = max_double;
  else if (p == 0.5)
    t = 0.0;
  else if (df == 1)
    t = 1.0 / tan((p + p) * 1.57079633);
  else if (df == 2)
    t = sqrt(1.0 / ((p + p) * (1.0 - p)) - 2.0);
  else
  {	
    double ddf = df;
    double a = sqrt(log(1.0 / (p * p)));
    double aa = a * a;
    a = a - ((2.515517 + (0.802853 * a) + (0.010328 * aa)) /
             (1.0 + (1.432788 * a) + (0.189269 * aa) +
              (0.001308 * aa * a)));
    t = ddf - 0.666666667 + 1.0 / (10.0 * ddf);
    t = sqrt(ddf * (exp(a * a * (ddf - 0.833333333) / (t * t)) - 1.0));
  }
  return (positive)? t : -t;
}

double SampleStat::confidence(double p_value)
{
  int df = n - 1;
  if (df <= 0) return maxValue;
  double t = tval((1.0 + p_value) * 0.5, df);
  if (t == maxValue)
    return t;
  else
    return (t * stdDev()) / sqrt(double(n));
}


/*/
char* SampleStat::getInterval(double p_value)
{
	char* st="asdf";
	double conf;
	
	conf=this->confidence(p_value);
	
	//sprintf(st, "(%f", conf);
	
	return st;
} //*/




