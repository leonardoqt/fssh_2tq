#include <iostream>
#include <mpi.h>
#include <chrono>
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
	//
	if (rank==0)
		cin>>sz>>thd1>>thd2>>scaling>>coupling>>width>>shift>>mass>>x0>>x_sigma>>x_l>>x_r>>dt>>ek0>>ek1>>nek>>state>>nsample;
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
	psi(state) = 1;
	psi = psi / norm(psi);
	//
	vec num_r, num_t, num_R, num_T;
	// loop over each ek
	vec Ek = linspace(ek0,ek1,nek);
	for(auto iek : Ek)
	{
		num_r = vec(sz,fill::zeros);
		num_t = vec(sz,fill::zeros);
		num_R = vec(sz,fill::zeros);
		num_T = vec(sz,fill::zeros);
		for(int t1=0; t1<nsample; t1++)
			if ( t1%size == rank)
			{
				vec x_ini(1,fill::randn);
				x_ini(0) = x_ini(0)*x_sigma + x0;
				vec v_ini(1,fill::randn);
				v_ini(0) = v_ini(0)/2/mass/x_sigma + sqrt(2*iek/mass);
				fssh.new_trajectory(psi,mass,x_ini,v_ini);
				while(!fssh.stop_traj())
					fssh.run_step();
				//
				if (ion.x(0)<0)
					num_r(ion.istate) += 1;
				else
					num_t(ion.istate) += 1;
			}
		//
		MPI_Allreduce(num_r.memptr(),num_R.memptr(),sz,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(num_t.memptr(),num_T.memptr(),sz,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		if (rank ==0)
		{
			//cout<<log(iek);
			cout<<iek;
			for (auto m1:num_R)
				cout<<'\t'<<m1/nsample;
			for (auto m1:num_T)
				cout<<'\t'<<m1/nsample;
			cout<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
