#include <iostream>
#include <chrono>

#include "sh_control.h"
#include "sh_test_model.h"

using namespace arma;

int main()
{
	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::high_resolution_clock::time_point timepoint;
	timepoint now = clock::now();
	arma_rng::set_seed(now.time_since_epoch().count());
	//
	int el_sz, x_sz = 1;
	double thresh1, thresh2, mass, x0, x_sigma, dt, ek, max_time;
	std::cin>>el_sz>>thresh1>>thresh2>>mass>>x0>>x_sigma>>dt>>ek>>max_time;
	//
	//
	sh_wfc wfc;
	sh_ion ion;
	sh_control control;
	test_engine engine;
	test_eigstate eig1, eig2;
	potential pot;
	//
	engine.init(&pot,el_sz);
	//
	cx_vec psi(el_sz,fill::zeros);
	psi(0) = 1;
	control.init(thresh1,thresh2,dt,max_time);
	//
	{
		vec M = mass*(ones<vec>(x_sz));
		vec x_ini(x_sz,fill::randn);
		vec v_ini(x_sz,fill::randn);
		x_ini = x_ini*x_sigma + x0;
		v_ini = v_ini/2/x_sigma/mass + sqrt(2*ek/mass);
		control.set_new_traj(wfc,ion,engine,eig1,psi,M,x_ini,v_ini);
		while(!control.stop_traj(ion))
		{
			control.run_step(wfc,ion,engine,eig1,eig2,0,0);
			eig1 = eig2;
			// debug
			vec x = ion.x;
			vec v = ion.v;
			vec E = eig2.E;
			double xx = x(0);
			double ek = 0.5*dot(ion.mass,v%v);
			double ep = E(ion.istate);
			std::cout<<ion.time_duration<<'\t'<<xx<<'\t'<<ek<<'\t'<<ep<<'\t'<<ek+ep<<'\t'<<ion.istate<<'\t'<<abs(wfc.psi(0))<<std::endl;
		}
	}
	//
	std::cout<<std::endl;
	return 0;
}
