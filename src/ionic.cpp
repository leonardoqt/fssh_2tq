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

void ionic::move_by(int Jstate, vec dHdtm, double dt)
{
	// make actual move and rescale velocity if Jstate != istate
	//
	jstate = Jstate;
	x_t = x + v * dt + a *dt*dt/2;
	a_t = -H->grad(x_t,istate)/mass;
	v_t = v + (a + a_t)/2*dt;
	// for K1>1, use interpolated a1
	if (dHdtm.n_elem >1)
	{
		vec v0 = v;
		vec a0 = a, a1;
		double tq = dt / dHdtm.n_elem;
		int bad_termination = 0;
		for(size_t t1=0; t1<dHdtm.n_elem-1; t1++)
		{
			double ff = dHdtm(t1);
			double A = -tq/8*( dot(a0,a0)*tq*tq+4*tq*dot(a0,v0)-8*ff*tq+4*dot(v0,v0) );
			double B = tq*tq/2*dot(a0,a0)+2*tq*dot(a0,v0)-4*ff*tq+2*dot(v0,v0);
			double C = 4*( tq*dot(a0,a0)+dot(a0,v0)+ff );
			if ( B*B-4*A*C < 0)
			{
				bad_termination = 1;
				break;
			}
			double ll1 = (-B+sqrt(B*B-4*A*C)) /2/A;
			double ll2 = (-B-sqrt(B*B-4*A*C)) /2/A;
			vec aa1 = ( 2*a0+ll1*(v0+a0*tq/2) ) / (2-ll1*tq);
			vec aa2 = ( 2*a0+ll2*(v0+a0*tq/2) ) / (2-ll2*tq);
			if ( dot(aa1-a0,aa1-a0) < dot(aa2-a0,aa2-a0) )
			{
				v0 = v0 + (a0+aa1)/2*tq;
				a0 = aa1;
			}
			else
			{
				v0 = v0 + (a0+aa2)/2*tq;
				a0 = aa2;
			}
		}
		if (!bad_termination)
			v_t = v0 + (a0 + a_t)/2*tq;
	}
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
	double E_pot0 = E0(istate), E_k0 = 0.5*mass*dot(v,v);
	mat V_tmp, U_tmp, T_tmp;
	x = x_t;
	v = v_t;
	a = a_t;
	H->adiab_parallel(x,V0,E0,V_tmp,U_tmp,T_tmp);
	V0 = V_tmp;
	double E_pot1 = E0(istate), E_k1 = 0.5*mass*dot(v,v);
	v = sqrt((E_pot0+E_k0-E_pot1)/E_k1)*v;
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
