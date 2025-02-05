#ifndef __SH_CONTROL__
#define __SH_CONTROL__

#include <armadillo>
#include "sh_wfc.h"
#include "sh_ion.h"
#include "sh_eigstate.h"
#include "sh_ele_engine.h"

class sh_control
{
private:
	void solve_LD(arma::mat H, arma::mat V1, arma::vec& E2, arma::mat& V2, arma::mat& U, arma::mat& logU);
	void parallel_zy(arma::mat V1, arma::mat& V2, arma::mat& U);
public:
	double thresh1, thresh2, dtc, tmax;
	//
	void init(double Thresh1, double Thresh2, double Dtc, double Tmax);
	void set_new_traj(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, arma::cx_vec psi0, arma::vec mass, arma::vec x0, arma::vec v0);
	void run_step(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, sh_eigstate& eig2, int hault_tq1, int hault_tq2);
	// interpolate H AND force along the guess move direction
	void run_step_interp_H(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, sh_eigstate& eig2, int hault_tq1, int hault_tq2);
	int stop_traj(sh_ion& ion);
	// TODO: add trajectory saving function
};


#endif
