#ifndef __SH_EIGSTATE__
#define __SH_EIGSTATE__

#include <armadillo>

// save eigenvalue and eigenstate-related info
// compute logU from current (as base) to another sh_eigstate
class sh_eigstate
{
public:
	arma::vec E;
	// logU from eig1, should also do parallel transport on self 
	virtual void compute_logU(sh_eigstate& eig1, arma::mat& U, arma::mat& logU) = 0;
	virtual ~sh_eigstate() = default;
};

#endif
