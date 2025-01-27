#include "sh_test_model.h"

using namespace arma;

void test_eigstate::parallel_zy(mat V1, mat& U)
{
	mat V2 = V;
	int sz = V1.n_cols;
	U = V1.t() * V2;
	double ss = sign(det(U));
	U.col(0) *= ss;
	V2.col(0) *= ss;
	int changed = 1;
	while (changed)
	{
		changed = 0;
		for (int t1=0; t1<sz-1; t1++)
		for (int t2=t1+1; t2<sz; t2++)
		{
			double dtr = 3*(U(t1,t1)*U(t1,t1)+U(t2,t2)*U(t2,t2)) + 6*U(t1,t2)*U(t2,t1) + 8*(U(t1,t1)+U(t2,t2)) - 3*(dot(U.row(t1),U.col(t1))+dot(U.row(t2),U.col(t2)));
			if (dtr <-1e-5)
			{
				U.col(t1) *= -1;
				U.col(t2) *= -1;
				V2.col(t1) *= -1;
				V2.col(t2) *= -1;
				changed = 1;
			}
		}
	}
	V = V2;
}

void test_eigstate:: compute_logU(sh_eigstate& eig1, mat& U, mat& logU)
{
	test_eigstate* pp=static_cast<test_eigstate*>(&eig1);
	parallel_zy(pp->V,U);
	logU = real(logmat(U));
	logU = ( logU - logU.t() )/2;
}

void test_engine::init(potential* Scf, int Sz)
{
	scf_engine = Scf;
	scf_engine->init(Sz,0.0,0.0,0.0,0.0);
}

void test_engine::compute_scf(vec X, sh_eigstate& eig1)
{
	test_eigstate* pp=static_cast<test_eigstate*>(&eig1);
	//
	scf_engine->adiab(X,pp->E,pp->V);
}

void test_engine::compute_egrad(vec X, int istate, vec&egrad)
{
	egrad = scf_engine->grad(X,istate);
}

void test_engine::compute_drvcp(vec X, int istate, int jstate, vec& drvcp)
{
	if(istate != jstate)
		drvcp = ones<vec>(X.n_rows);
	else
		drvcp = ones<vec>(X.n_rows);
}
