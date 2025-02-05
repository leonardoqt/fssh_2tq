#include "sh_ion.h"

using namespace arma;

void sh_ion::init(sh_ele_engine& engine, sh_eigstate& eigstate, vec Mass, vec X, vec V, int Istate)
{
	time_duration = 0;
	istate = Istate;
	mass = Mass;
	x = x_t = X;
	v = v_t = V;
	//
	vec egrad;
	engine.compute_scf(x,eigstate);
	engine.compute_egrad(x,istate,egrad);
	a = a_t = -egrad / mass;
}

void sh_ion::try_move(sh_ele_engine& engine, sh_eigstate& eigstate, double dt)
{
	x_t = x + v * dt + a*dt*dt/2;
	engine.compute_scf(x_t,eigstate);
}

void sh_ion::move_and_hop(sh_ele_engine& engine, sh_eigstate& eigstate, int jstate, double dt)
{
	// Must be called after try_move and no member(and eigstate) shall be changed since then, otherwise, need to uncomment the following two lines
	// Pure phase change should be fine though
	vec egrad;
	// normal move without hop
	//x_t = x + v * dt + a *dt*dt/2;
	//engine.compute_scf(x_t,eigstate);
	engine.compute_egrad(x_t,istate,egrad);
	a_t = -egrad / mass;
	v_t = v + (a+a_t)/2*dt;
	if(jstate != istate)
	{
		// deal with rescale/frustrated hop
		del_pot = eigstate.E(jstate) - eigstate.E(istate);
		engine.compute_egrad(x_t,jstate,egrad);
		engine.compute_drvcp(x_t,istate,jstate,dij);
		//
		vec dij_unit = dij / norm(dij);
		vec v_para = dij_unit * dot(dij_unit,v_t);
		vec v_perp = v_t - v_para;
		if (norm(v_para)<1e-4) std::cout<<"Warning: norm of velocity in dij direction is less than 1e-4"<<std::endl;
		//
		double Ek = 0.5*dot(mass,v_t%v_t);
		double E0 = 0.5*dot(mass,v_perp%v_perp);
		double Ep = 0.5*dot(mass,v_para%v_para);
		double Ec = 0.5*dot(mass,v_para%v_perp);
		double Er = Ek - E0 - del_pot;
		if ( Ec*Ec+Ep*Er >=0 )
		{
			// successful hop
			// change istate to the hopped one
			istate = jstate;
			// scale v_para > 0 to match energy
			double lambda = ( sqrt(Ec*Ec+Ep*Er)-Ec ) / Ep;
			v_t = v_perp + lambda * v_para;
			a_t = -egrad / mass;
		}
		// TODO: ask vale about this condition
		// currently using eq. 26 of https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00276
		// TODO: may use unit dij as basis to compute lambda
		else if (dot(egrad,dij)*dot(dij,v_t)>0)
		{
			// reverse velocity in dij direction
			double lambdaf = -2*Ec/Ep - 1;
			v_t = v_perp + lambdaf * v_para;
		}
	}
	//
	x = x_t;
	v = v_t;
	a = a_t;
	time_duration += dt;
}

void sh_ion::move_and_hop(vec xx, vec vv, sh_ele_engine& engine, sh_eigstate& eigstate, int jstate, double dt)
{
	x_t = xx;
	v_t = vv;
	vec egrad;
	engine.compute_scf(x_t,eigstate);
	engine.compute_egrad(x_t,istate,egrad);
	a_t = -egrad / mass;
	//
	if(jstate != istate)
	{
		// deal with rescale/frustrated hop
		del_pot = eigstate.E(jstate) - eigstate.E(istate);
		engine.compute_egrad(x_t,jstate,egrad);
		engine.compute_drvcp(x_t,istate,jstate,dij);
		//
		vec dij_unit = dij / norm(dij);
		vec v_para = dij_unit * dot(dij_unit,v_t);
		vec v_perp = v_t - v_para;
		if (norm(v_para)<1e-4) std::cout<<"Warning: norm of velocity in dij direction is less than 1e-4"<<std::endl;
		//
		double Ek = 0.5*dot(mass,v_t%v_t);
		double E0 = 0.5*dot(mass,v_perp%v_perp);
		double Ep = 0.5*dot(mass,v_para%v_para);
		double Ec = 0.5*dot(mass,v_para%v_perp);
		double Er = Ek - E0 - del_pot;
		if ( Ec*Ec+Ep*Er >=0 )
		{
			// successful hop
			// change istate to the hopped one
			istate = jstate;
			// scale v_para > 0 to match energy
			double lambda = ( sqrt(Ec*Ec+Ep*Er)-Ec ) / Ep;
			v_t = v_perp + lambda * v_para;
			a_t = -egrad / mass;
		}
		// TODO: ask vale about this condition
		// currently using eq. 26 of https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00276
		// TODO: may use unit dij as basis to compute lambda
		else if (dot(egrad,dij)*dot(dij,v_t)>0)
		{
			// reverse velocity in dij direction
			double lambdaf = -2*Ec/Ep - 1;
			v_t = v_perp + lambdaf * v_para;
		}
	}
	//
	x = x_t;
	v = v_t;
	a = a_t;
	time_duration += dt;
}
