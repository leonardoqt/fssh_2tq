#ifndef __SH_ELE_ENGINE__
#define __SH_ELE_ENGINE__

#include "sh_eigstate.h"

class sh_ele_engine
{
public:
	virtual void compute_scf(arma::vec X, sh_eigstate& eig1) = 0;
	virtual void compute_egrad(arma::vec X, int istate, arma::vec& egrad) = 0;
	virtual void compute_drvcp(arma::vec X, int istate, int jstate, arma::vec& drvcp) = 0;
	virtual ~sh_ele_engine() = default;
};

#endif
