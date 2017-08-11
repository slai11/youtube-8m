/******************************************************************\
 * File:           SampleStat.cpp for PS
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           12 Feb 2008
\******************************************************************/

#ifndef SAMPLESTAT_H_
#define SAMPLESTAT_H_

class SampleStat
{
public:
	SampleStat(){reset();}
	virtual ~SampleStat(){};
	double min(){ return minValue; }
	double max(){ return maxValue; }
	int sampleNum(){ return n; }
	virtual void reset(); 
    virtual void update(double s);
	double mean();
	double var();
    double stdDev();
    double confidence(double p_value);
	double adjconfidence(double p_value);
    //char* getInterval(double p_value); 
    static double tval(double p, int df);  
    
protected:
    int n;
    double x;
    double x2;
    double minValue, maxValue;
	
};

class Counter
{
public:
	Counter(){reset();}
	virtual ~Counter(){};
	void reset(){ n=0; x=0; }
	void update(double s){ x+=s; n+=1; }
	void update(double s, double t){ x+=(s*t); n+=t; }
	double average(){ if (n>0) return x/n; else return 0; }
	
protected:
	double n;
	double x;
	
};

#endif /*SAMPLESTAT_H_*/
