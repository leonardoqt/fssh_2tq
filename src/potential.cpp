#include "potential.h"

using namespace arma;
using namespace std;

void potential::init(int Sz, double Coupling, double Width, double Shift, double Scaling)
{
	sz = Sz;
	coupling = Coupling;
	width = Width;
	shift = Shift;
	scaling = Scaling;
}

// Tully 1
//void potential::diab(vec x, mat& H)
//{
//	H = zeros(sz,sz);
//	double A=0.01, B=1.6, C=0.005, D=1.0;
//	H(0,1) = H(1,0) = C*exp(-D*x(0)*x(0));
//	if (x(0) < 0)
//		H(0,0) = -A*(1 - exp(B*x(0)));
//	else
//		H(0,0) = A*(1 - exp(-B*x(0)));
//	H(1,1) = -H(0,0);
//}

// Tully 2
//void potential::diab(vec x, mat& H)
//{
//	H = zeros(sz,sz);
//	double A=0.1, B=0.28, C=0.015, D=0.06, E = 0.05;
//	H(0,0) = 0;
//	H(1,1) = -A*exp(-B*x(0)*x(0))+E;
//	H(0,1) = H(1,0) = C*exp(-D*x(0)*x(0));
//}

// Z model
void potential::diab(vec x, mat& H)
{
	// off diagonal will be c/sqrt(sz)*exp(-x^2/2w^2)
	H = ones(sz,sz)*coupling/sqrt(sz)*exp(-x(0)*x(0)/2/width/width);
	// diagonal will be +- tanh(2/w*x)+shift*n
	for (int t1=0; t1<sz/2; t1++)
		H(t1,t1) = tanh(2*x(0)/width)+t1*shift;
	for (int t1 = sz/2; t1<sz; t1++)
		H(t1,t1) = -tanh(2*x(0)/width)+(sz-1-t1)*shift;
	H = H * scaling;
}

// W model
//void potential::diab(vec x, mat& H)
//{
//	double pi2 = atan(1)*2;
//	double ang0 = pi2 / sz;
//	double dang = ang0 * 2;
//	// off diagonal will be c/sqrt(sz)*exp(-x^2/2w^2)
//	H = ones(sz,sz)*coupling/sqrt(sz)*exp(-x(0)*x(0)/2/width/width);
//	for (int t1=0; t1<sz; t1++)
//		H(t1,t1) = tanh(tan(pi2-ang0-t1*dang) * x(0))+t1*shift;
//	H = H * scaling;
//}

// Z_ model
//void potential::diab(vec x, mat& H)
//{
//	// off diagonal will be c/sqrt(sz)*exp(-x^2/2w^2)
//	H = zeros(sz,sz);
//	H.col(sz-1) += coupling/sqrt(sz)*exp(-x(0)*x(0)/2/width/width);
//	H.row(sz-1) += coupling/sqrt(sz)*exp(-x(0)*x(0)/2/width/width);
//	// diagonal will be +- tanh(2/w*x)+shift*n
//	for (int t1=0; t1<sz/2; t1++)
//		H(t1,t1) = tanh(2*x(0)/width)+t1*shift;
//	for (int t1 = sz/2; t1<sz-1; t1++)
//		H(t1,t1) = -tanh(2*x(0)/width)+(sz-2-t1)*shift;
//	H(sz-1,sz-1) = int(sz/2)*shift;
//	H = H * scaling;
//}


void potential::adiab(vec x, vec& E, mat& V)
{
	mat H;
	diab(x,H);
	eig_sym(E,V,H);
}

void potential::adiab_parallel(vec x, mat V0, vec &E, mat &V, mat& U, mat &logU)
{
	adiab(x,E,V);
	parallel_zy(V0,V,U);
	logU = real(logmat(U));
	logU = ( logU - logU.t() ) / 2;
}

void potential::parallel_zy(mat V1, mat& V2, mat&U)
{
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
}

vec potential::grad(vec x, int istate)
{
	double dx = 1e-6;
	vec E0, x1;
	mat V, H0, H1;
	//
	diab(x,H0);
	adiab(x,E0,V);
	vec grad_x = x;
	//
	for (size_t t1=0; t1<x.n_cols; t1++)
	{
		x1 = x;
		x1(t1) += dx;
		diab(x1,H1);
		mat tmp = V.col(istate).t()*(H1-H0)*V.col(istate);
		grad_x(t1) = tmp(0,0) / dx;
	}
	return grad_x;
}

void potential::diag_parallel(mat H, mat V0, vec &E, mat &V, mat &U, mat &logU)
{
	eig_sym(E,V,H);
	parallel_zy(V0,V,U);
	logU = real(logmat(U));
	logU = ( logU - logU.t() ) / 2;
}
