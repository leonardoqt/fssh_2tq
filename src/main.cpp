#include <iostream>
#include <mpi.h>
#include <chrono>
#include <fstream>
#include "sh.h"

using namespace std;
using namespace arma;

int main()
{
	typedef chrono::high_resolution_clock clock;
	typedef chrono::high_resolution_clock::time_point timepoint;
	timepoint now = clock::now();
	MPI_Init(NULL,NULL);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	arma_rng::set_seed(now.time_since_epoch().count()+rank*10);
	//
	int sz;
	double thd1, thd2;
	double scaling, coupling, width, shift;
	double mass, x0, x_sigma, x_l, x_r, dt;
	double ek0, ek1;
	int nek, state, nsample;
	double max_time_mul;
	//
	if (rank==0)
		cin>>sz>>thd1>>thd2>>scaling>>coupling>>width>>shift>>mass>>x0>>x_sigma>>x_l>>x_r>>dt>>ek0>>ek1>>nek>>state>>nsample>>max_time_mul;
	MPI_Bcast(&sz      , 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&thd1    , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&thd2    , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&scaling , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&coupling, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&width   , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&shift   , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&mass    , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&x0      , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&x_sigma , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&x_l     , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&x_r     , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&dt      , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek0     , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek1     , 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&nek     , 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&state   , 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&nsample , 1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&max_time_mul, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	//
	potential H;
	electronic ele;
	ionic ion;
	sh fssh;
	//
	H.init(sz,coupling,width,shift,scaling);
	fssh.link_component(&H,&ele,&ion);
	fssh.set_param(thd1,thd2,dt,x_l,x_r);
	//
	cx_vec psi(sz,fill::zeros);
	if (state >=sz)
	{
		arma_rng::set_seed(state);
		psi = cx_vec(sz,fill::randn);
	}
	else
		psi(state) = 1;
	psi = psi / norm(psi);
	//
	//-plot potential-
	if (rank==0)
	{
		ofstream diab_surf,adiab_surf;
		diab_surf.open("diabats.dat");
		adiab_surf.open("adiabats.dat");
		vec xxx=linspace(x_l,x_r,1000);
		mat H_diab, V_a;
		vec E_adiab;
		for (auto m1 : xxx)
		{
			vec xxxx = vec(1,fill::ones)*m1;
			H.diab(xxxx,H_diab);
			H.adiab(xxxx,E_adiab,V_a);
			diab_surf<<m1;
			adiab_surf<<m1;
			for (int t2=0; t2<sz; t2++)
			{
				diab_surf<<"\t"<<H_diab(t2,t2);
				adiab_surf<<"\t"<<E_adiab(t2);
			}
			diab_surf<<'\t'<<H_diab(0,1)<<endl;
			adiab_surf<<endl;
		}
		diab_surf.close();
		adiab_surf.close();
	}
	//----------------
	//
	vec num_r, num_t, num_R, num_T;
	double trapped, Trapped;
	// loop over each ek
	vec Ek = linspace(ek0,ek1,nek);
	for(auto iek : Ek)
	{
		num_r = vec(sz,fill::zeros);
		num_t = vec(sz,fill::zeros);
		num_R = vec(sz,fill::zeros);
		num_T = vec(sz,fill::zeros);
		trapped = 0;
		Trapped = 0;
		for(int t1=0; t1<nsample; t1++)
			if ( t1%size == rank)
			{
				vec x_ini(1,fill::randn);
				x_ini(0) = x_ini(0)*x_sigma + x0;
				vec v_ini(1,fill::randn);
				v_ini(0) = v_ini(0)/2/mass/x_sigma + sqrt(2*iek/mass);
				fssh.new_trajectory(psi,mass,x_ini,v_ini);
				double max_time = max_time_mul*(x_r-x_l)/v_ini(0); // max_time_mul times length / initial velocity
				while(!fssh.stop_traj())
				{
					fssh.run_step();
					if (ion.time_duration > max_time)
						break;
				}
				//
				if ( ion.time_duration > max_time )
					trapped += 1;
				else
				{
				if (ion.x(0)<0)
					num_r(ion.istate) += 1;
				else
					num_t(ion.istate) += 1;
				}
			}
		//
		MPI_Allreduce(num_r.memptr(),num_R.memptr(),sz,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(num_t.memptr(),num_T.memptr(),sz,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(&trapped,&Trapped,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		if (rank ==0)
		{
			//cout<<log(iek);
			cout<<iek;
			for (auto m1:num_R)
				cout<<'\t'<<m1/(nsample-Trapped);
			for (auto m1:num_T)
				cout<<'\t'<<m1/(nsample-Trapped);
			cout<<'\t'<<Trapped/nsample<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
