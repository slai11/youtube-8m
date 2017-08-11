/*
 * A multi-server Queue with abandonment
 * Steady-state queue length and virtual waiting time
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <math.h>
#include "Rngs.h"
#include "SampleStat.h"
#include "MS_Queue_Abd.h"
#include "Gaussian.h"

using namespace std;

int main()
{
	double simTime = 2e6;     // let the system run for a max length of time 
	int simNum = 5;          // number of runs

	//unsigned int N = 220;      // numer of servers
	double delay = 60.0;         // prescribed delay

	double lambda = 1.0;      // arrival rate per second
	double mu = 1.0 / 230.0;  // service rate per second
	char outputFileName[] = "Abd_level_css3.txt";
	//double rho = lambda/mu/N;           // traffic intensity
	double rho;

	// double lambda = N*mu*rho; // arrival rate
	// double alpha = 1/gamma;   // abandonment rate
	// double gamma = 5.0;

	int I = 1, J = 1, K = 1;
	RngBase *arr[1], *ser[1], *abd[1];


	arr[0] = new ExpGen(lambda);
	double ca2 = 1;

	double cs2 = 3.0;
	ser[0] = new LogNormalGen(1 / mu, cs2 / mu / mu);
	// ser[0] = new ExpGen(mu);
	// double cs2 = 1.0;
	//ser[0] = new ErlangGen(2*mu, 2); 
	//cs2[0] = 0.5;
	//ser[1] = new DetGen(1.0); 
	//cs2[1] = 0.0;	
	//ser[1] = new ErlangGen(2*mu, 2); 
	//cs2[1] = 0.5;
	//ser[2] = new ExpGen(mu); 
	//cs2[2] = 1;
	//ser[3] = new HyperExp2Gen(1/0.169, 1/2.20327, .5915); 
	//cs2[3] = 3.0;
	//ser[4] = new LogNormalGen(1/mu, 3); 
	//cs2[4] = 3.0;

	// double gamma = 980.12;
	// double alpha = 1.0 / gamma;
	// abd[0] = new ExpGen(alpha);
	// abd[0] = new ErlangGen(3*alpha,3);
	abd[0] = new HyperExp2Gen(0.001, 1.0 / 6.0, 0.98);
	double gamma = 1000.0*0.98 + (1 - 0.98)*6.0;
	//abd[0] = new HyperExp2Gen(0.2, 6.0, 0.9);
	// abd[0] = new HyperExp2Gen(1, 1/3.0, .5);
	// abd[0] = new HyperExp2Gen(alpha*0.3, 79*alpha/30, 0.7);
	// abd[1] = new ExpGen(2.0/3);
	// abd[0] = new UniformGen(0, 10);
	// abd[1] = new ExpGen(.25);

	int sysSize;
	double depWaitTime, depPatTime;

	int startTime = time(NULL);

	srand(time(NULL));

	for (int N = 195; N < 221; N++){
		rho = lambda / N / mu;
		for (int k = 0; k < K; k++){
			for (int j = 0; j < J; j++){
				for (int i = 0; i < I; i++){


					/*   ------ Start Simulation  -----  */

					SampleStat bSize, sSize;
					bSize.reset();
					sSize.reset();

					SampleStat bSizeVar;
					bSizeVar.reset();

					SampleStat bTime;
					bTime.reset();

					SampleStat bTimeVar;
					bTimeVar.reset();

					SampleStat probAbd;
					probAbd.reset();

					SampleStat probAbdGd;
					probAbdGd.reset();


					Counter bSizeOne, sSizeOne;
					Counter bSizeSquareOne;
					Counter bTimeOne;
					Counter bTimeSquareOne;
					Counter probAbdOne;
					Counter probAbdGdOne;
					Counter probWGdOne;

					MS_Queue_Abd q(N);
					q.init();

					Event event;
					for (int run = 0; run < simNum; run++)
					{
						bSizeOne.reset();
						sSizeOne.reset();
						bSizeSquareOne.reset();
						bTimeOne.reset();
						bTimeSquareOne.reset();
						probAbdOne.reset();
						probAbdGdOne.reset();
						probWGdOne.reset();

						while (q.getClock() < simTime*(run + 1))
						{
							event = q.nextEvent();
							sysSize = q.getSize();

							bSizeOne.update(q.getBufferSize(), event.tm);
							bSizeSquareOne.update(q.getBufferSize()*q.getBufferSize(), event.tm);
							sSizeOne.update(q.getServerSize(), event.tm);

							q.actOn(event);

							q.checkStatus();

							if (event.tp == arrival)
							{
								q.schNextArr(arr[i]->genNext(), ser[j]->genNext(), abd[k]->genNext());
							}
							else if (event.tp == departure)
							{
								depWaitTime = q.getDepWaitTime();
								depPatTime = q.getDepPatTime();
								bTimeOne.update(depWaitTime);
								bTimeSquareOne.update(depWaitTime*depWaitTime);

								if ((depWaitTime > delay) && (depPatTime > delay))
									probWGdOne.update(1);
								else probWGdOne.update(0);

								if (!q.getDepReneged())
								{
									probAbdOne.update(0);
									probAbdGdOne.update(0);
								}
								else{
									if (depPatTime > delay) probAbdGdOne.update(1);
									else probAbdGdOne.update(0);
								}
							}
							else
							{
								probAbdOne.update(1);
								// bTimeOne.update(q.getRegWaitTime()); /// Use this line when computing the buffer time
							}
						}
						cout << run << " out of " << simNum << " runs done ..." << endl;

						if (run){
							bSize.update(bSizeOne.average());
							sSize.update(sSizeOne.average());
							bSizeVar.update(bSizeSquareOne.average() - bSizeOne.average()*bSizeOne.average());
							bTime.update(bTimeOne.average());
							bTimeVar.update(bTimeSquareOne.average() - bTimeOne.average()*bTimeOne.average());
							probAbd.update(probAbdOne.average());
							probAbdGd.update(probAbdGdOne.average() / probWGdOne.average());
						}
					}

					ofstream outputFile;
					outputFile.open(outputFileName, ios::out | ios::app);

					//cout.setf(ios::fixed);
					//cout.precision(4);

					cout << "________________________________\nNumber of servers = " << N << ", Arrival rate = " << lambda
						<< ", Service rate = " << mu << ", Mean patience time = " << gamma << ". \n" << endl;
					//outputFile << "\n\n\nNumber of servers = " << N << ", Arrival rate = " << lambda
						//<< ", Service rate = " << mu << ", Mean patience time = " << gamma << ". \n" << endl;

					cout << "Arrival time is " << arr[i]->getName()
						<< "; Service time is " << ser[j]->getName()
						<< "; Patient time is " << abd[k]->getName() << "\n" << endl;
					//outputFile << "Arrival time is " << arr[i]->getName()
						//<< "; Service time is " << ser[j]->getName()
						//<< "; Patient time is " << abd[k]->getName() << "\n" << endl;

					double sigq2 = mu*(rho*ca2 + cs2 + rho - 1);
					double ql = N*mu*(rho - 1)*gamma;
					double qv = N*gamma*sigq2 / 2;
					double vw = gamma*(log(rho));
					double sigw2 = (ca2 / 2 / rho + cs2 / 2 + (rho - 1) / 2 / rho) / mu;
					double vv = sigw2*gamma / N;


					cout << "Abandonment probability = " << probAbd.mean() << "+/-" << probAbd.confidence(0.95) << ", vs " << (rho - 1) / rho << endl;
					cout << endl;
					//outputFile << "Abandonment probability = " << probAbd.mean() << "+/-" << probAbd.confidence(0.95) << ", vs " << (rho - 1) / rho << endl;
					//outputFile << endl;

					cout << "Mean queue length = " << bSize.mean() << "+/-" << bSize.confidence(0.95) << ", vs " << ql << endl;
					cout << "Variance of queue length = " << bSizeVar.mean() << "+/-" << bSizeVar.confidence(0.95) << ", vs " << qv << endl;
					cout << endl;
					//outputFile << "Mean queue length = " << bSize.mean() << "+/-" << bSize.confidence(0.95) << ", vs " << ql << endl;
					//outputFile << "Variance of queue length = " << bSizeVar.mean() << "+/-" << bSizeVar.confidence(0.95) << ", vs " << qv << endl;
					//outputFile << endl;

					cout << "Mean virtual waiting = " << bTime.mean() << "+/-" << bTime.confidence(0.95) << ", vs " << vw << endl;
					cout << "Variance of virtual waiting = " << bTimeVar.mean() << "+/-" << bTimeVar.confidence(0.95) << ", vs " << vv << endl;
					cout << endl;
					//outputFile << "Mean virtual waiting = " << bTime.mean() << "+/-" << bTime.confidence(0.95) << ", vs " << vw << endl;
					//outputFile << "Variance of virtual waiting = " << bTimeVar.mean() << "+/-" << bTimeVar.confidence(0.95) << ", vs " << vv << endl;
					//outputFile << endl;


					cout << "Mean number of busy servers = " << sSize.mean() << "+/-" << sSize.confidence(0.95) << ", vs " << N << endl;
					cout << endl;
					//outputFile << "Mean number of busy servers = " << sSize.mean() << "+/-" << sSize.confidence(0.95) << ", vs " << N << endl;
					//outputFile << endl;

					cout << "P[X<W|min(X,W)>" << delay << "]=" << probAbdGd.mean() << "+/-" << probAbdGd.confidence(0.95) << ", vs " << 1 - cdf(0.5, 0, sqrt(sigw2)) << endl;
					cout << endl;
					//outputFile << "P[X<W|min(X,W)>" << delay << "]=" << probAbdGd.mean() << "+/-" << probAbdGd.confidence(0.95) << ", vs " << 1 - cdf(0.5, 0, sqrt(sigw2)) << endl;
					//outputFile << endl;
					//outputFile << "***********************************************************" << endl;
					outputFile << probAbdGd.mean() << "+/-" << probAbdGd.confidence(0.95) << endl;

					outputFile.close();
				}
			}
		}
	}

	int endTime = time(NULL);
	cout << "\nCPU time: " << (endTime - startTime) / 60 << " min "
		<< (endTime - startTime) % 60 << " seconds" << endl;


	char endchar;
	cin >> endchar;

	return 0;
}
