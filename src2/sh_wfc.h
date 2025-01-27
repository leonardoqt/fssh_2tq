#ifndef __SH_WFC__
#define __SH_WFC__

#include <armadillo>

class sh_wfc
{
public:
	int sz;
	// psi is current psi, psi_t is next step, both in adiabatic basis
	arma::cx_vec psi, psi_t;
	// leave space for rho support
	arma::cx_mat rho, rho_t;
	int istate, ntq2;
	double dtq2;
	//
	void init(arma::cx_vec Psi);
	// void init(arma::cx_mat Rho);
	// compute how small dtq1 needs to be, return ntq1
	int query_ntq1(double thresh1, arma::vec E1, arma::mat T, double tc);
	// compute how small dtq2 needs to be, return ntq2, propagate psi using LD method
	int query_ntq2(double thresh2, arma::vec E1, arma::vec E2, arma::mat U, arma::mat T, int Istate, double dtq1);
	// itq2 from 0 to at most ntq2-1
	arma::vec get_hop(arma::vec E1, arma::vec E2, arma::mat T, int itq2);
	void restore_psi();
};

#endif
