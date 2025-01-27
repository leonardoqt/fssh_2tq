#ifndef __POTENTIAL__
#define __POTENTIAL__

#include <armadillo>

class potential; // time independent part

class potential
{
public:
	int sz;
	double shift, width, coupling, scaling;
	void init(int Sz, double Coupling, double Width, double Shift, double scaling);
	void diab(arma::vec x, arma::mat& H);
	void adiab(arma::vec x, arma::vec& E, arma::mat& V);
	void adiab_parallel(arma::vec x, arma::mat V0, arma::vec& E, arma::mat& V, arma::mat& U, arma::mat& logU);
	void parallel_zy(arma::mat V1, arma::mat& V2, arma::mat& U);
	arma::vec grad(arma::vec x, int istate);
	//
	void diag_parallel(arma::mat H, arma::mat V0, arma::vec& E, arma::mat& V, arma::mat& U, arma::mat& logU);
};

#endif
