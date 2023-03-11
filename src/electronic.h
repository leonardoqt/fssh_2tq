#ifndef __ELECTRONIC_
#define __ELECTRONIC_

#include <armadillo>

class electronic;

class electronic
{
public:
	int sz;
	arma::cx_vec psi, psi_t; // psi is current psi, psi_t is next, both in adiabatic basis
	// no support for rho are implemented for now, but should be trivial
	arma::cx_mat rho;
	arma::mat V;
	//
	arma::vec E1, E2;
	arma::mat U, T;
	int istate, ntq2;
	double dtq2;
	//
	void init(arma::cx_vec Psi, arma::mat VV);
	void init(arma::cx_mat Rho, arma::mat VV);
	int query_ion(double thd1, arma::vec EE1, arma::mat TT, double tc); // check how small dtq1 needs to be, return ntq1
	int query_hop(double thd2, arma::vec EE1, arma::vec EE2, arma::mat UU, arma::mat TT, int Istate, double dtq1); // check how small dtq2 needs to be and propagate psi using LD method
	arma::vec get_hop(int itq2); // itq2 from 0 to ntq2-1 at most
	void restore_psi();
	//
	void print();
};
#endif
