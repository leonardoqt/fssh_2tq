#ifndef __FSSH__
#define __FSSH__

#include <armadillo>
#include "potential.h"
#include "electronic.h"
#include "ionic.h"

class sh;

class sh
{
public:
	potential *H;
	electronic *ele;
	ionic *ion;
	double thd1, thd2, dtc;
	double stop_x_1, stop_x_2;
	//
	void link_component(potential* HH, electronic* Ele, ionic* Ion);
	void set_param(double Thd1, double Thd2, double Dtc, double stopx1, double stopx2);
	void new_trajectory(arma::cx_vec psi0, double mass, arma::vec x0, arma::vec v0);
	void run_step(int hault_tq1, int hault_tq2);
	//
	// TODO:
	void try_decoherence();
	int stop_traj();
	//
	void print();
};
#endif
