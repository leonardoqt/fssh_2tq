#include "sh_wfc.h"

using namespace arma;

void sh_wfc::init(cx_vec Psi)
{
	sz = Psi.n_rows;
	psi = psi_t = Psi;
	rho = rho_t = psi * psi.t();
}

int sh_wfc::query_ntq1(double thresh1, vec E1, mat T, double tc)
{
	vec e1 = E1 - mean(E1);
	double g0 = 1e-13;
	for(int t1=0; t1<sz; t1++)
		for(int t2=0; t2<sz; t2++)
			g0 += abs(T(t1,t2)*e1(t1)*e1(t1)*psi(t1)*conj(psi(t1)));
	double dtq1 = sqrt(3*sz*thresh1/tc/g0);
	int ntq1 = ceil(tc/dtq1);
	return ntq1;
}

int sh_wfc::query_ntq2(double thresh2, vec E1, vec E2, mat U, mat T, int Istate, double dtq1)
{
	istate = Istate;
	//
	cx_mat iHt = ( diagmat(E1)+U*diagmat(E2)*U.t() )/2*cx_double(0,-1)*dtq1;
	psi_t = U.t()*expmat(iHt)*psi;
	//
	// calculate hopping probability from istate to other states
	vec hop(sz,fill::zeros);
	for (int ti=istate,tj=0; tj<sz; tj++)
		if (tj != ti)
		{
			double rate = 2*real( T(ti,tj)*psi_t(tj)*conj(psi_t(ti)) / (psi(ti)*conj(psi(ti))+1e-5) )*dtq1;
			if ( rate > 0) hop(tj) = rate;
		}
	//
	// calculate ntq2
	ntq2 = floor(sum(hop)/thresh2)+1;
	if (ntq2 > 1e5)
	{
		std::cout<<"Warning: K2 in SH-tq2 > 10^5, reset it to 10^5!"<<std::endl;
		ntq2 = 100000;
	}
	dtq2 = dtq1 / ntq2;
	return ntq2;
}

vec sh_wfc::get_hop(vec E1, vec E2, mat T, int itq2)
{
	// while psi is changing here, one would restore it to what evolved in LD basis, i.e., psi_t as computed in query_ntq2 when hopping are evaluted
	vec hop_p = vec(sz,fill::zeros);
	cx_mat iHt = ( ( diagmat(E1)+(2*itq2+1.0)/2/ntq2*diagmat(E2-E1) )*cx_double(0,-1) - T )*dtq2;
	cx_vec psi2 = expmat(iHt)*psi;
	//
	for (int ti=istate,tj=0; tj<sz; tj++)
		if (tj != ti)
		{
			double rate = 2*real( T(ti,tj)*psi2(tj)*conj(psi2(ti)) / (psi(ti)*conj(psi(ti))+1e-5) )*dtq2;
			if ( rate > 0) hop_p(tj) = rate;
		}
	psi = psi2;
	return hop_p;
}

void sh_wfc::restore_psi()
{
	psi = psi_t;
}
