#include "electronic.h"

using namespace arma;
using namespace std;

void electronic::init(cx_vec Psi, mat VV)
{
	sz = Psi.n_rows;
	psi = Psi;
	rho = psi * psi.t();
	V = VV;
}

void electronic::init(cx_mat Rho, mat VV)
{
	sz = Rho.n_rows;
	rho = Rho;
	V = VV;
}

int electronic::query_ion(double thd1, vec EE1, mat TT, double tc)
{
	vec e1 = EE1 - mean(EE1);
	double g0 = 0;
	for (int t1=0;t1<sz;t1++)
		for (int t2=0;t2<sz;t2++)
		{
			g0 += abs(TT(t1,t2)*e1(t1)*e1(t1)*psi(t1)*conj(psi(t1)));
		}
	double dtq1 = sqrt(3*sz*thd1/tc/g0);
	int ntq1 = ceil(tc/dtq1);
	return ntq1;
}

int electronic::query_hop(double thd2, vec EE1, vec EE2, mat UU, mat TT, int Istate, double dtq1)
{
	E1 = EE1;
	E2 = EE2;
	U = UU;
	T = TT;
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
			double rate = 2*real( T(ti,tj)*psi_t(tj)*conj(psi_t(ti)) / (psi(ti)*conj(psi(ti))+1e-15) )*dtq1;
			if ( rate > 0) hop(ti) = rate;
		}
	//
	// calculate ntq2
	ntq2 = floor(sum(hop)/thd2)+1;
	dtq2 = dtq1 / ntq2;
	return ntq2;
}

vec electronic::get_hop(int itq2)
{
	vec hop_p = vec(sz,fill::zeros);
	cx_mat iHt = ( ( diagmat(E1)+(2*itq2+1.0)/2/dtq2*diagmat(E2-E1) )*cx_double(0,-1) - T )*dtq2;
	cx_vec psi2 = expmat(iHt)*psi;
	//
	for (int ti=istate,tj=0; tj<sz; tj++)
		if (tj != ti)
		{
			double rate = 2*real( T(ti,tj)*psi2(tj)*conj(psi2(ti)) / (psi(ti)*conj(psi(ti))+1e-15) )*dtq2;
			if ( rate > 0) hop_p(tj) = rate;
		}
	return hop_p;
}

void electronic::restore_psi()
{
	psi = psi_t;
}

void electronic::print()
{
	cout<<"--From electron--"<<endl;
	cout<<"sz = "<<sz<<endl;
	psi.t().print("psi");
	V.print("V");
	cout<<"-----------------"<<endl;
}
