#include "ionic.h"

using namespace arma;
using namespace std;

void ionic::assign_potential(potential* HH)
{
	H = HH;
}

void ionic::init(double Mass, vec X, vec V, int Istate)
{
	time_duration = 0;
	istate = Istate;
	mass = Mass;
	x = x_t = X;
	v = v_t = V;
	H->adiab(x,E0,V0);
	a = -H->grad(x,istate)/mass;
}

void ionic::try_move(double dt, vec& E1, vec& E2, mat& V1, mat& V2, mat& U, mat& T)
{
	x_t = x + v * dt + a *dt*dt/2;
	H->adiab_parallel(x_t,V0,E2,V2,U,T);
	T = T / dt;
	E1 = E0;
	V1 = V0;
}

void ionic::move_by(int Jstate, double dt)
{
	// make actual move and rescale velocity if Jstate != istate
	//
	jstate = Jstate;
	x_t = x + v * dt + a *dt*dt/2;
	a_t = -H->grad(x_t,istate)/mass;
	v_t = v + (a + a_t)/2*dt;
	//
	// update dij and del_pot if jstate != istate
	if ( jstate != istate )
	{
		vec e_tmp;
		mat v_tmp;
		H->adiab(x_t,e_tmp,v_tmp);
		del_pot = e_tmp(jstate) - e_tmp(istate);
		//
		// TODO: get dij. not necessary for 1D
	}
	time_duration += dt;
}

void ionic::try_hop()
{
	if ( jstate != istate)
	{
		double ek = 0.5*mass*pow(norm(v_t),2);
		if ( ek >= del_pot )
		{
			// rescale velocity
			// TODO: this expression is not suitable for more than 1d
			v_t = sqrt( (ek-del_pot)/(ek) ) * v_t;
			istate = jstate;
			a_t = -H->grad(x_t,istate)/mass;
		}
		else
		{
			// frustrated hops
			// TODO: this expression is not suitable for more than 1d
			if ( dot(a,v) > 0 && dot(a,H->grad(x_t,istate)) > 0 )
				v_t = -v_t;
		}
	}
	mat V_tmp, U_tmp, T_tmp;
	x = x_t;
	v = v_t;
	a = a_t;
	H->adiab_parallel(x,V0,E0,V_tmp,U_tmp,T_tmp);
	V0 = V_tmp;
}

void ionic::print()
{
	cout<<"--From ionic--"<<endl;
	cout<<"istate = "<<istate<<endl;
	cout<<"mass = "<<mass<<endl;
	x.t().print("x");
	v.t().print("v");
	E0.t().print("E0");
	V0.print("V0");
	cout<<"--------------"<<endl;
}
