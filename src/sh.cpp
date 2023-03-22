#include "sh.h"

using namespace arma;
using namespace std;

void sh::link_component(potential* HH, electronic* Ele, ionic* Ion)
{
	H = HH;
	ele = Ele;
	ion = Ion;
}

void sh::set_param(double Thd1, double Thd2, double Dtc, double stopx1, double stopx2)
{
	ion->assign_potential(H);
	thd1 = Thd1;
	thd2 = Thd2;
	dtc = Dtc;
	stop_x_1 = stopx1;
	stop_x_2 = stopx2;
}

void sh::new_trajectory(cx_vec psi0, double mass, vec x0, vec v0)
{
	psi0 = psi0 / norm(psi0);
	vec pp = cumsum( square(abs(psi0)) );
	double rr = randu();
	for (int t1=0; t1<H->sz; t1++)
		if ( rr <= pp(t1) )
		{
			ion->init(mass,x0,v0,t1);
			break;
		}
	ele->init(psi0,ion->V0);
	// some trajectory counting history
}

void sh::run_step(int hault_tq1, int hault_tq2)
{
	vec E1, E2;
	mat V1, V2, U, T;
	// propose a move using dtc
	ion->try_move(dtc,E1,E2,V1,V2,U,T);
	// get required dtq1
	int K1 = ele->query_ion(thd1,E1,T,dtc);
	double dtq1 = dtc / K1;
	//
	//interpolate potential in LD
	vec E_tq0 = E1, E_tq1;
	mat V_tq0 = V1, V_tq1, U_tq, T_tq;
	mat dH = U*diagmat(E2)*U.t()-diagmat(E1);
	int attempt_hop_tq1 = 0, attempt_hop_state = ion->istate;
	for (int itq1=1; itq1<=K1; itq1++)
	{
		mat H_tq1 = V1 * ( diagmat(E1) + itq1*1.0/K1*dH ) * V1.t();
		H->diag_parallel(H_tq1,V_tq0,E_tq1,V_tq1,U_tq,T_tq);
		T_tq /= dtq1;
		//
		if ( !(hault_tq1 && attempt_hop_tq1) )
		{
			int attempt_hop_tq2 = 0;
			// get required dtq2
			int K2 = ele->query_hop(thd2,E_tq0,E_tq1,U_tq,T_tq,attempt_hop_state,dtq1);
			// evaulate hop, ignore if already attempt to hop
			for (int itq2=0; itq2<K2; itq2++)
			{
				if (hault_tq2 && attempt_hop_tq2) break;
				vec hop_p = ele->get_hop(itq2);
				vec accum_p = cumsum(hop_p);
				double rr = randu();
				for (int t3 = 0; t3 < H->sz; t3++)
				{
					if ( rr <= accum_p(t3) )
					{
						attempt_hop_state = t3;
						attempt_hop_tq1 = 1;
						attempt_hop_tq2 = 1;
						break;
					}
				}
			}
		}
		// finish dtq2, restore wfc as calculated in query_hop
		// TODO: in this scheme, we defer actual hop evaluation to the end of tc
		// may also try execute hop right at the correcponding dtq1
		ele->restore_psi();
		E_tq0 = E_tq1;
		V_tq0 = V_tq1;
	}
	// execute hop (or not, just ionic step)
	ion->move_by(attempt_hop_state,dtc);
	//ion->move_by(attemp_hop_state,dHdtm,dtc);
	ion->try_hop();
	ele->V = ion->V0;
}

int sh::stop_traj()
{
	if ( ion->x(0) < stop_x_1 && ion->v(0) < 0 )
		return 1;
	if ( ion->x(0) > stop_x_2 && ion->v(0) > 0 )
		return 1;
	return 0;
}

void sh::print()
{
	cout<<"------From SH------"<<endl;
	cout<<"threadhold1 = "<<thd1<<endl;
	cout<<"threadhold2 = "<<thd2<<endl;
	cout<<"dtc = "<<dtc<<endl;
	ele->print();
	ion->print();
	cout<<"-------------------"<<endl;
}
