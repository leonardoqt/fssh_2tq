#ifndef __SH_TEST_MODEL_
#define __SH_TEST_MODEL_

#include "sh_eigstate.h"
#include "sh_ele_engine.h"
#include "potential.h"

class test_eigstate: public sh_eigstate
{
private:
	void parallel_zy(arma::mat V1, arma::mat& U);
public:
	arma::mat V;
	arma::mat basis;
	//
	void compute_logU(sh_eigstate& eig1, arma::mat& U, arma::mat& logU) override;
};


class test_engine: public sh_ele_engine
{
private:
	potential* scf_engine;
public:
	void init(potential* Scf, int sz);
	void compute_scf(arma::vec X, sh_eigstate& eig1) override;
	void compute_egrad(arma::vec X, int istate, arma::vec& egrad) override;
	void compute_drvcp(arma::vec X, int istate, int jstate, arma::vec& drvcp) override;
};

#endif
