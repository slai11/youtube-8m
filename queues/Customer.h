/******************************************************************\
 * File:           Customer.h for Rodin project
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           13 Feb 2008
\******************************************************************/

#ifndef CUSTOMER_H_
#define CUSTOMER_H_

#include<iostream>

using namespace std;

class Customer
{
public:
	Customer(double a, double s){ 
		t_arr=a; ser_time=s;
	}
	
	Customer(double a, double s, double p){
		t_arr=a; ser_time=s; pat_time=p; 
	}
	
	virtual ~Customer(){};
	
	void arrToRegBuf(double t){
		t_reg=t+pat_time;
		delayed=true;
		status=0;
		reg_ed=false;
		toDel=false;
	}
	
	void arrToSer(double t){
		t_start=t; 
		t_dep=t+ser_time;
		delayed=false;
		status=1;
		reg_ed=false;
		toDel=false;
	}
	
	void moveToSer(double t){ 
		t_start=t; 
		t_dep=t+ser_time;
		status=1;
	}

	void depart(double t){
		status=2;
	}

	void renege(double t)        
	{ 
		ser_time=0.0; /// delete this line when computing the mean buffer time
		reg_ed=true;	
	}

	void toBeDel()
	{
		toDel=true;
	}
	
	double getTimeToArr(){ return t_arr; }
	double getTimeToDep(){ return t_dep; }
	double getTimeToReg(){ return t_reg; }
	
	bool getDelayInfo()  { return delayed; }

	double getWaitTime() { return t_start-t_arr; }
	double getInSerTime(){ return t_dep-t_start; }
	double getRespTime() { return t_dep-t_arr; }

	double getPatTime()  { return pat_time; }
	double getSerTime()  { return ser_time; }
	
	int getStatus()     { return status; }
	bool reneged()       { return reg_ed; }
	bool Dele()       {return toDel;}
	
protected :
	double t_arr, t_start, t_dep, t_reg;
	double ser_time, pat_time;
	bool delayed;
	
	int status;
	bool reg_ed, toDel;

friend class CustPtCompare;

friend class CustPtCompareR;

};

class CustPtCompare
{ 
public : 
	int operator()( Customer* x, Customer* y )
	{	return (x->t_dep > y->t_dep); }
};

class CustPtCompareR
{ 
public : 
	int operator()( Customer* x, Customer* y )
	{	return (x->t_reg > y->t_reg); }
};

#endif /*CUSTOMER_H_*/
