/******************************************************************\
 * File:           MS_Queue_Abd.cpp for MSQ
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           25 April 2008
\******************************************************************/

#include "MS_Queue_Abd.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

MS_Queue_Abd::MS_Queue_Abd(unsigned int K)
{
	N=K;
}

void MS_Queue_Abd::init()
{
	t=0;

	while(!server.empty())  server.pop();
	while(!buffer.empty())  buffer.pop_front();
	while(!regList.empty()) regList.pop();

	arrCust=new Customer(1,1,1);
	depCust=new Customer(1,1,1);

	regBufAdj=0;
}

void MS_Queue_Abd::schNextArr(double a, double s, double r)
{
	arrCust = new Customer(a+t,s,r);
}

Event MS_Queue_Abd::nextEvent()
{
	Event ev;

	ev.tp=arrival;
	ev.tm=arrCust->getTimeToArr();

	if(!server.empty())
	{
		if(server.top()->getTimeToDep()<=ev.tm)
		{
			ev.tp=departure;
			ev.tm=server.top()->getTimeToDep();
		}
	}

	if(!regList.empty())
	{
		if(regList.top()->getTimeToReg()<=ev.tm)
		{
			ev.tp=reneging;
			ev.tm=regList.top()->getTimeToReg();
		}
	}

	ev.tm-=t;

	return ev;
}


void MS_Queue_Abd::actOn(Event ev){

	Customer* tempCust;

	/*-------step 1: update clock-------------*/
	t+=ev.tm;

	if(ev.tp==arrival)            // case 1: arrival
	{
	   	if(server.size()<N)
	   	{
	   		arrCust->arrToSer(t);
	   		server.push(arrCust);
	   	}
	    else
	    {
	    	arrCust->arrToRegBuf(t);
	    	buffer.push_back(arrCust);
	    	regList.push(arrCust);
	    }
	}
    else if(ev.tp==departure)   // case 2: departure
    {
    	if(!depCust->getDelayInfo() || depCust->reneged() || depCust->Dele()){
    		delete depCust;
    	}
    	depCust=server.top();
    	depCust->depart(t);
    	server.pop();
		if(!buffer.empty())
		{
			if(buffer.front()->reneged()) //new
				regBufAdj--; //new
			tempCust=buffer.front();//new
    		buffer.pop_front();//new
    		tempCust->moveToSer(t);//new
    		server.push(tempCust);//new
		}
    	//while((!buffer.empty()) && buffer.front()->reneged())
    	//{
    	//	tempCust=buffer.front();
    	//	buffer.pop_front();
    	//	delete tempCust;
    	//	regBufAdj--;
    	//}
    	//if(!buffer.empty())
    	//{
    	//	tempCust=buffer.front();
    	//	buffer.pop_front();
    	//	tempCust->moveToSer(t);
    	//	server.push(tempCust);
    	//}
    }
    else if(ev.tp==reneging)    // case 3: reneging
    {
    	regCust=regList.top();
    	//checkReg();
    	regCust->renege(t);
    	regList.pop();
    	regBufAdj++;
    }
    else
    {
    	cerr<<"something wrong: what is the type of next event?"<<endl;
    	exit(-1);
    }

	// do some cleanning

	while(!regList.empty() && regList.top()->getStatus()){
		tempCust=regList.top();
		regList.pop();
		tempCust->toBeDel();
		if(tempCust->getStatus()==2 && tempCust!=depCust)
			delete tempCust;
	}
}

/*
 * ============================ for debug use only =====================
 */

void MS_Queue_Abd::printStatus()
{

	cout<<"_______________\nIt is time "<<t<<": "
		<<arrCust->getTimeToArr()<<" ("<<arrCust->getSerTime()<<") ---> "
		<<buffer.size()-regBufAdj<<" : "<<server.size()<<" --> ";

	if(server.empty())
		cout<<endl;
	else
		cout<<(server.top()->getTimeToDep())<<endl;

	if(!regList.empty())
		cout<<"                    "<<regList.top()->getTimeToReg()<<endl;

}

void MS_Queue_Abd::printBuffer()
{
	list<Customer*>::iterator custIter;

	cout<<"Buffer:    "<<buffer.size()<<" - "<<regBufAdj<<" = ";
	for( custIter = buffer.begin(); custIter != buffer.end(); custIter++ ) {
	    if((*custIter)->reneged()) cout<<"1";
	    else cout<<"0";
	}
	cout<<endl;
}

void MS_Queue_Abd::printRegList()
{

	priority_queue<Customer*, deque<Customer*>, CustPtCompareR>
                    regList2;   //build a reg list using STL priority_queue

	while(!regList2.empty()) regList2.pop();

	cout<<"RegBuffer: "<<regList.size()<<" - "<<" ?"<<" = ";
	while(!regList.empty()){
		cout<<regList.top()->getStatus();
		regList2.push(regList.top());
		regList.pop();
	}

	while(!regList2.empty()){
		regList.push(regList2.top());
		regList2.pop();
	}

	cout<<endl;
}

void MS_Queue_Abd::printEvent(Event event)
{
	cout<<"Next event is: ";
	if(event.tp==arrival) cout<<"Arrival ";
	else if (event.tp==departure) cout<<"Departure ";
	else if (event.tp==reneging) cout<<"Reneging ";

	cout<<" in "<<event.tm<<" time units."<<endl;
}

void MS_Queue_Abd::checkDep()
{
	if (depCust->reneged()){
		cout<<"wrong! dep"<<endl;
		exit(-1);
	}
}

void MS_Queue_Abd::checkReg()
{
	if(regCust->reneged()){
		cout<<"reg customer already in reneged!"<<endl;
		cout<<regCust->getTimeToArr()<<" ("<<regCust->getSerTime()<<") -> "
			<<regCust->getTimeToReg()<<" ("<<regCust->getPatTime()<<") "
			<<regCust->reneged()<<regCust->getStatus()<<endl;
	}
}

void MS_Queue_Abd::checkStatus(){

	list<Customer*>::iterator custIter;

	double realBufSize=0;
	for( custIter = buffer.begin(); custIter != buffer.end(); custIter++ ) {
	    if( ! ((*custIter)->reneged()) )
		realBufSize++;
	}

	if( int(buffer.size())-regBufAdj!= realBufSize){

		printBuffer();

		printRegList();

		exit(-1);
	}
}
