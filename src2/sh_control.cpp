#include "sh_control.h"
#include "sh_test_model.h"

using namespace arma;

void sh_control::init(double Thresh1, double Thresh2, double Dtc, double Tmax)
{
	thresh1 = Thresh1;
	thresh2 = Thresh2;
	dtc = Dtc;
	tmax = Tmax;
}

void sh_control::set_new_traj(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, cx_vec psi0, vec mass, vec x0, vec v0)
{
	int sz = psi0.n_rows;
	psi0 = psi0 / norm(psi0);
	vec pp = cumsum( square(abs(psi0)) );
	double rr = randu();
	for(int t1=0; t1<sz; t1++)
		if ( rr <= pp(t1) )
		{
			ion.init(engine,eig1,mass,x0,v0,t1);
			break;
		}
	wfc.init(psi0);
	// TODO: may add some trajectory counting history
}

void sh_control::run_step(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, sh_eigstate& eig2, int hault_tq1, int hault_tq2)
{
	// TODO: check if need to adjust phase of eigs somewhere in the procedure, perhaps inside compute_logU
	// eig1 must be the eigstate used for init the trajectory
	//
	mat U, T;
	// propose a move using dtc
	ion.try_move(engine,eig2,dtc);
	eig2.compute_logU(eig1,U,T);
	//test_eigstate* p1 = static_cast<test_eigstate*> (&eig1);
	//test_eigstate* p2 = static_cast<test_eigstate*> (&eig2);
	//p1->V.print("eig1_V");
	//p2->V.print("eig2_V");
	T = T / dtc;
	int K1 = wfc.query_ntq1(thresh1,eig1.E,T,dtc);
	double dtq1 = dtc / K1;
	std::cout<<"In SH, K1 = "<<K1<<std::endl;
	//
	// interpolate potential in LD
	vec E_tq0 = eig1.E, E_tq1;
	mat V_tq0 = eye(size(T)), V_tq1, U_tq, T_tq;
	mat dH = U*diagmat(eig2.E)*U.t()-diagmat(eig1.E);
	int attempt_hop_tq1 = 0, attempt_hop_state = ion.istate;
	//
	for(int itq1=1; itq1<=K1; itq1++)
	{
		mat H_tq1 = diagmat(eig1.E) + itq1*1.0/K1*dH;
		solve_LD(H_tq1,V_tq0,E_tq1,V_tq1,U_tq,T_tq);
		//std::cout<<"itq1 = "<<itq1<<std::endl;
		//V_tq0.print("V_tq0");
		//U_tq.print("U_tq");
		//U.print("U");
		T_tq /= dtq1;
		//
		// TODO: the expected behavior of hault_tq1 and hault_tq2 is to control whether continue running evolution if a hop is proposed, but the corresponding hopping/dynamics treatment is not implemented, may consider remove them or set both as zero and leave this comment here
		if( !(hault_tq1 && attempt_hop_tq1) )
		{
			int attempt_hop_tq2 = 0;
			// propagate wfc in LD is done here
			int K2 = wfc.query_ntq2(thresh2,E_tq0,E_tq1,U_tq,T_tq,attempt_hop_state,dtq1);
			// evaluate hop
			for(int itq2=0; itq2<K2; itq2++)
			{
				if (hault_tq2 && attempt_hop_tq2) break;
				vec hop_p = wfc.get_hop(E_tq0,E_tq1,T_tq,itq2);
				vec accum_p = cumsum(hop_p);
				double rr = randu();
				for(int t3=0; t3<wfc.sz; t3++)
				{
					if ( rr<=accum_p(t3) )
					{
						attempt_hop_state = t3;
						wfc.istate = attempt_hop_state;
						attempt_hop_tq1 = 1;
						attempt_hop_tq2 = 1;
						break;
					}
				}
			}
		}
		// finish dtq2, restore wfc as calculated in query_ntq2
		wfc.restore_psi();
		E_tq0 = E_tq1;
		V_tq0 = V_tq1;
	}
	ion.move_and_hop(engine,eig2,attempt_hop_state,dtc);
	// TQ: this must be done outside, because here we need to make assign value to "Derived" class
	//eig1 = eig2;
}

void sh_control::run_step_interp_H(sh_wfc& wfc, sh_ion& ion, sh_ele_engine& engine, sh_eigstate& eig1, sh_eigstate& eig2, int hault_tq1, int hault_tq2)
{
	// eig1 must be the eigstate used for init the trajectory
	// must assign eig2 (the derived class) to eig1 after calling this
	//
	mat U, T;
	vec egrad;
	// propose a move using dtc
	ion.try_move(engine,eig2,dtc);
	eig2.compute_logU(eig1,U,T);
	T = T / dtc;
	int K1 = wfc.query_ntq1(thresh1,eig1.E,T,dtc);
	double dtq1 = dtc / K1;
	std::cout<<"In SH, K1 = "<<K1<<std::endl;
	engine.compute_egrad(ion.x_t,ion.istate,egrad);
	vec a_t = -egrad / ion.mass;
	double weight = norm(logmat(U));
	//
	// interpolate potential in LD
	// for each dtq1 step, also run velocity verlet for classical dynamics
	// along the x_t-x direction
	// v will be update based on interpolation between ion.a and a1, depending on whether U_LD of diabats is similar to 1 (ion.a)
	// or U (a_t)
	vec ex = ion.x_t - ion.x;
	double xx = norm(ex);
	ex = ex / xx;
	double x0 = 0;
	vec v0 = ion.v, a0 = ion.a;
	//
	vec E_tq0 = eig1.E, E_tq1;
	mat V_tq0 = eye(size(T)), V_tq1, U_tq, T_tq, U_LD = eye(size(T));
	mat dH = U*diagmat(eig2.E)*U.t()-diagmat(eig1.E);
	int attempt_hop_tq1 = 0, attempt_hop_state = ion.istate;
	//
	for(int itq1=1; itq1<=K1; itq1++)
	{
		// run vv along ex direction
		x0 = x0 + dot(ex,v0)*dtq1 + dot(ex,a0)/2*dtq1*dtq1;
		mat H_tq1 = diagmat(eig1.E) + x0/xx*dH;
		solve_LD(H_tq1,V_tq0,E_tq1,V_tq1,U_tq,T_tq);
		T_tq /= dtq1;
		U_LD = U_LD*U_tq;
		// get interpolate a, using norm(logU)
		double weight_LD = norm(logmat(U_LD));
		vec a1 = ion.a + weight_LD/weight*(a_t-ion.a);
		v0 = v0 + (a0+a1)/2*dtq1;
		a0 = a1;
		std::cout<<"a1 scaling = "<<weight_LD/weight<<std::endl;
		//
		// TODO: the expected behavior of hault_tq1 and hault_tq2 is to control whether continue running evolution if a hop is proposed, but the corresponding hopping/dynamics treatment is not implemented, may consider remove them or set both as zero and leave this comment here
		if( !(hault_tq1 && attempt_hop_tq1) )
		{
			int attempt_hop_tq2 = 0;
			// propagate wfc in LD is done here
			int K2 = wfc.query_ntq2(thresh2,E_tq0,E_tq1,U_tq,T_tq,attempt_hop_state,dtq1);
			// evaluate hop
			for(int itq2=0; itq2<K2; itq2++)
			{
				if (hault_tq2 && attempt_hop_tq2) break;
				vec hop_p = wfc.get_hop(E_tq0,E_tq1,T_tq,itq2);
				vec accum_p = cumsum(hop_p);
				double rr = randu();
				for(int t3=0; t3<wfc.sz; t3++)
				{
					if ( rr<=accum_p(t3) )
					{
						attempt_hop_state = t3;
						wfc.istate = attempt_hop_state;
						attempt_hop_tq1 = 1;
						attempt_hop_tq2 = 1;
						break;
					}
				}
			}
		}
		// finish dtq2, restore wfc as calculated in query_ntq2
		wfc.restore_psi();
		E_tq0 = E_tq1;
		V_tq0 = V_tq1;
	}
	vec x_final = ion.x + x0*ex;
	vec v_final = v0;
	//
	ion.move_and_hop(x_final,v_final,engine,eig2,attempt_hop_state,dtc);
	// TQ: this must be done outside, because here we need to make assign value to "Derived" class
	//eig1 = eig2;
}

int sh_control::stop_traj(sh_ion& ion)
{
	if( ion.time_duration > tmax )
		return 1;
	else
		return 0;
}

void sh_control::solve_LD(mat H, mat V1, vec& E2, mat& V2, mat& U, mat& logU)
{
	eig_sym(E2,V2,H);
	parallel_zy(V1,V2,U);
	logU = real(logmat(U));
	logU = ( logU - logU.t() )/2;
}

void sh_control::parallel_zy(mat V1, mat& V2, mat& U)
{
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
}
