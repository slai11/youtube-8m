/******************************************************************\
 * File:           Event.h for MSQ
 * Language:       C++ (ISO 1998)
 * Copyright:      Jiheng Zhang, Georgia Institute of Technology
 * Date:           12 Feb 2008
\******************************************************************/


#ifndef EVENT_H_
#define EVENT_H_

enum EventType {arrival, departure, reneging};

struct Event
{	

	EventType tp;   
	double tm;      // maturity time

};

#endif /*EVENT_H_*/
