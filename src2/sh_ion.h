#ifndef __SH_ION__
#define __SH_ION__

#include <armadillo>
#include "sh_ele_engine.h"
#include "sh_eigstate.h"

class sh_ion
{
public:
	int istate;
	double del_pot;
	double time_duration;
	arma::vec mass;
	arma::vec x, x_t, v, v_t, a, a_t, dij;
	//
	// initiate x and v, compute a
	void init(sh_ele_engine& engine, sh_eigstate& eigstate, arma::vec Mass, arma::vec X, arma::vec V, int Istate);
	// store new position in x_t given dt, compute eigenstate of new geometry; used for evaluating dtq1
	void try_move(sh_ele_engine& engine, sh_eigstate& eigstate, double dt);
	void move_and_hop(sh_ele_engine& engine, sh_eigstate& eigstate, int Jstate, double dt);
	// this use input x to re-calculate H, and use input v as v_parallel
	void move_and_hop(arma::vec xx, arma::vec vv_p, sh_ele_engine& engine, sh_eigstate& eigstate, int Jstate, double dt);
};

#endif
