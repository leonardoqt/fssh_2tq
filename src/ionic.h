#ifndef __IONIC__
#define __IONIC__

#include <armadillo>
#include "potential.h"

class ionic;

class ionic
{
public:
	int istate, jstate;
	double mass, del_pot;
	double time_duration;
	potential *H;
	arma::vec x, x_t, v, v_t, a, a_t, dij;
	arma::vec E0; // always associated with x
	arma::mat V0;
	//
	void assign_potential(potential* HH);
	void init(double Mass, arma::vec X, arma::vec V, int Istate);
	// get E,V,T for a proposed move with time dt, store new x in x_t, note that no changes in v evaluated
	// used for generating H1 H2 to evaluate dtq1
	void try_move(double dt, arma::vec& E1, arma::vec& E2, arma::mat& V1, arma::mat& V2, arma::mat& U, arma::mat& T);
	// get new x and v using velocity verlet, update del_pot and dij used for rescale velocity if jstate != istate
	void move_by(int Jstate, double dt); 
	// there are two options in dealing with hop in dtq1
	// 1. when a hop occurs, get the true H at the hopped place; ignore the rest of dtq1
	// 2. supress other possible hops in dtq1, use the final position in tc to update ionic status
	void try_hop();
	//
	void print();
};

#endif
