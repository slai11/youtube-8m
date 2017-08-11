/******************************************************************\
 * File:           MS_Queue_Abd.h for MSQ
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           25 April 2008
\******************************************************************/

#ifndef MS_QUEUE_ABD_H_
#define MS_QUEUE_ABD_H_

#include <queue>
#include <list>
#include "Customer.h"
#include "Event.h"

using namespace std;

class MS_Queue_Abd
{
public:
	MS_Queue_Abd(){};

	virtual ~MS_Queue_Abd(){};
	
	MS_Queue_Abd(unsigned int K);
	
	void init();	

	void schNextArr(double a, double s, double r);
	
	Event nextEvent();

	void actOn(Event E);
	
	double getClock()   { return t; }
	
	int getBufferSize() { return buffer.size()-regBufAdj; }
	int getServerSize() { return server.size(); }
	int getSize()       { return buffer.size()+server.size()-regBufAdj; }
	
	double getDepWaitTime()  { return depCust->getWaitTime();    }
	double getDepInSerTime() { return depCust->getInSerTime();   }
	double getDepSerTime()   { return depCust->getSerTime();     }
	double getDepRespTime()  { return depCust->getRespTime();    }
	double getDepPatTime()   { return depCust->getPatTime();     }
	bool   getDepDelayInfo() { return depCust->getDelayInfo();   }
	bool   getDepReneged()   { return depCust->reneged();        }
	
	double getRegWaitTime()  { return regCust->getPatTime();     }
	
/*
 * ======================== for debug use only ============================
 */	
	
	void printStatus();
	void printBuffer();
	void printRegList();
	void printEvent(Event ev);
	void checkStatus();
	void checkDep();
	void checkReg();
	
protected:
	
	unsigned int N;     // limit on number of customer servered at a time
	double t;  // system clock

	priority_queue<Customer*, deque<Customer*>, CustPtCompare>
                    server;    //build a server pool using STL priority_queue
	
	list<Customer*> buffer;    //build a buffer using STL list
	
	priority_queue<Customer*, deque<Customer*>, CustPtCompareR>
                    regList;   //build a reg list using STL priority_queue

	Customer *arrCust, *depCust, *regCust;
	
	int regBufAdj;

};

#endif /*MS_QUEUE_ABD_H_*/
