#include <iomanip>
#include <mpi.h>
#include <RandomLib/Random.hpp>
#include "Timer.hpp"
#include "myprogress.hpp"
#include "parallel_efgs.hpp"
#include "ScenarioTree.hpp"

bool print_mw_signals = false;
bool print_mpi_signals = false;
bool print_main_functions = true;
bool print_contributions = false;
bool Queue_diagnostics = false;
std::vector<int> efgs_subproblems; /* holds the block_id's which will be
																			used for upper bounding calculations */

double optimization_time_main = 0;
double optimization_time_main_start = 0;

int main(int argc, char **argv){
double optimization_time_main_start = get_cpu_time();

	/* Initialize MPI */
	int rank, nproc;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);
	MPI_Status status;

	/* Input Reading */	
	std::string 	PROBLEM_CLASS 			= argv[1];
	std::string 	PARAMETER_FILE 			= argv[2];
	int 		N_INSTANCES 			= atoi(argv[3]);
	std::string 	RESULT_FILE 			= argv[4];
	int 		MAX_NUM_NODES_IN_A_PROBLEM 	= atoi(argv[5]);
	int 		LB_STOP_CR 			= atoi(argv[6]);
	double		LB_STOP_TH 			= atof(argv[7]);
	std::string 	PARTITION_FILE 			= argv[8];
	std::string 	EXE_BOUNDING 			= argv[9];
	int 		N_SUBPROBLEMS 			= atoi(argv[10]);
	double 	UB_FRACTION 			= atof(argv[11]);
	int 		UB_SEED 			= atoi(argv[12]);
	int 		UB_STOP_CR 			= atoi(argv[13]);
	double		UB_STOP_TH 			= atof(argv[14]);
	int 		UB_SOL_TYPE 			= atoi(argv[15]);
	int		UB_SOL_PARAM 			= atoi(argv[16]);
	int		SOLVE_FULL_PROB			= atoi(argv[17]);
	int		EQUALLY_LIKELY_SCENARIOS	= atoi(argv[18]);
	int		UNIFORM_MULTIPLICITY		= atoi(argv[19]);
	int		CALCULATE_EEV			= atoi(argv[20]);

	/* EGSO SETUP
	 * --------------------------------------------------------------------------
	 * --------------------------------------------------------------------------
	 */
	/* Param Sturct for Lower Bounding */
	struct{
		unsigned int block_id;
		unsigned int EFGS_type;
		bool EEV_indicator;
		//		double EGSO_stop_threshold;
	} param;

	int blocks[3]={1,1,1};
	MPI_Datatype types[3]={MPI_UNSIGNED, MPI_UNSIGNED, MPI_C_BOOL};
	MPI_Aint displacements[3];

	MPI_Aint unsigned_intex;
	MPI_Type_extent(MPI_UNSIGNED,  &unsigned_intex);
	//	MPI_Type_extent(MPI_C_BOOL,  &boolex);
	//	MPI_Type_extent(MPI_DOUBLE,  &doublex); 

	displacements[0]  =  static_cast<MPI_Aint>(0); 
	displacements[1]  =  unsigned_intex; 
	displacements[2]  =  unsigned_intex+unsigned_intex; 

	MPI_Datatype obj_type;	
	MPI_Type_struct(3,  blocks,  displacements, types, &obj_type);	
	MPI_Type_commit(&obj_type);

	/* MPI struct for passing three variables: int, int double
		 int blocks[3]={1,1,1};
		 MPI_Datatype types[3]={MPI_INT, MPI_INT, MPI_DOUBLE};
		 MPI_Aint displacements[3];

		 MPI_Aint intex, doublex;
		 MPI_Type_extent(MPI_INT,  &intex);
	//	MPI_Type_extent(MPI_DOUBLE,  &doublex); 

	displacements[0]  =  static_cast<MPI_Aint>(0); 
	displacements[1]  =  intex; 
	displacements[2]  =  intex+intex;

	MPI_Datatype obj_type;	
	MPI_Type_struct(3,  blocks,  displacements, types, &obj_type);	
	MPI_Type_commit(&obj_type);
	*/

	int i, msg;
	int next_SP, n_shutdown, num_switch_mode;

	/*[checkup]*/
	int ALLDONE = -7;
	int SWITCH_EFGS_MODE = -8;

	// total number of sub-problems to solve
	int n_completed = 0;
	int n_inprogress = nproc-1;	

	double time_begin = get_cpu_time();

	// an array to hold how many jobs each worker works on; initialized to zero
	int *njobs= new int [nproc]();


	/* EFGS SETUP 
	 * --------------------------------------------------------------------------
	 * --------------------------------------------------------------------------
	 */
	/* Global Variables for Upper Bounding */
	vector<double> efgs_candidates;
	vector<double> optimization_time_candidates;
	int num_sleep;
	int num_workers = nproc-1;
	int num_rows = 2; /* for EFGS */

	/* We have 2 custom structs. One for master_to_worker communication
	 * and the other for worker_to_master communication.  */
#pragma pack(push,1)
	typedef struct
	{
		int node_ID;
		int block_id;
		float rhs_contribution_update;
		int master_to_worker_signal;
	} param_EFGS_master_to_worker;
#pragma pack(pop)

#pragma pack(push,1)
	typedef struct
	{
		int node_ID;
		float rhs_contribution_return;
		double efgs_contribution;
		double optimization_time_contribution;
		int worker_to_master_signal;
	} param_EFGS_worker_to_master;
#pragma pack(pop)

	/* Variables for master_to_worker_signal */
	int SUBMIT = -1;
	int SLEEP = -2;
	int AWAKE = -3;
	int SHUTDOWN = -4;

	/* Variables for worker_to_master_signal */
	int REQUEST = -5;
	int ANSWER = -6;

	int blocks_EFGS_m_to_w[4] = {1,1,1,1};
	MPI_Datatype types_EFGS_m_to_w[4] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_INT};
	MPI_Aint displacements_EFGS_m_to_w[4];

	MPI_Aint int_size;
	MPI_Type_extent (MPI_INT, &int_size);
	MPI_Aint double_size;
	MPI_Type_extent (MPI_DOUBLE, &double_size);
	MPI_Aint float_size;
	MPI_Type_extent (MPI_FLOAT, &float_size);

	displacements_EFGS_m_to_w[0] = static_cast<MPI_Aint>(0);
	displacements_EFGS_m_to_w[1] = int_size;
	displacements_EFGS_m_to_w[2] = int_size + int_size;
	displacements_EFGS_m_to_w[3] = int_size + int_size + float_size;

	MPI_Datatype obj_type_EFGS_master_to_worker;
	MPI_Type_struct (4, blocks_EFGS_m_to_w, displacements_EFGS_m_to_w, 
			types_EFGS_m_to_w, &obj_type_EFGS_master_to_worker);
	MPI_Type_commit (&obj_type_EFGS_master_to_worker);	

	int blocks_EFGS_w_to_m[5] = {1,1,1,1,1};
	MPI_Datatype types_EFGS_w_to_m[5] = {MPI_INT, MPI_FLOAT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
	MPI_Aint displacements_EFGS_w_to_m[5];

	displacements_EFGS_w_to_m[0] = static_cast<MPI_Aint>(0);
	displacements_EFGS_w_to_m[1] = int_size;
	displacements_EFGS_w_to_m[2] = int_size + float_size;
	displacements_EFGS_w_to_m[3] = int_size + float_size + double_size;
	displacements_EFGS_w_to_m[4] = int_size + float_size + double_size + double_size;

	MPI_Datatype obj_type_EFGS_worker_to_master;
	MPI_Type_struct (5, blocks_EFGS_w_to_m, displacements_EFGS_w_to_m, 
			types_EFGS_w_to_m, &obj_type_EFGS_worker_to_master);
	MPI_Type_commit (&obj_type_EFGS_worker_to_master);	

	/* EGSO MASTER
	 * --------------------------------------------------------------------------
	 * --------------------------------------------------------------------------
	 */
	if (rank == 0){	// load balancer

		
		double master_time =get_cpu_time();

		// All subproblems will compute EGSO; but only a pre-defined fraction of the subproblems will also compute EFGS
		// Randomly select those that will also compute EFGS
		std::vector<int> SP_indices;
		std::vector<unsigned short> indicator_efgs(N_SUBPROBLEMS, 0);	// fill a vector of size=N_SUBPROBLEMS with value=0
		RandomLib::Random r;		// create random number object
		r.Reseed(UB_SEED);	// seed with a "unique" seed
		//	std::cout << "Using " << r.Name() << "\n"
		//		  << "with seed " << r.SeedString() << "\n";

		// fill in the vector of scenario indices
		for (int i = 0; i < N_SUBPROBLEMS; i++)
			SP_indices.push_back(i);

		//create a random permutation of the elements of the 'population' vector
		std::random_shuffle(SP_indices.begin(), SP_indices.end(), r);

		int n_efgs;
		if (UB_FRACTION <= 1.0)
		{
			n_efgs = ceil( UB_FRACTION * N_SUBPROBLEMS);
			if (n_efgs > 1000)	// we solve efgs for UB_FRACTION of group subproblems up to a maximum of 10000
				n_efgs = 1000;

			for (int i = 0; i < n_efgs; i++)
			{
				indicator_efgs[SP_indices[i]] = UB_SOL_TYPE; //the first n_efgs entries in the shuffled SP_indices will compute EFGS
				efgs_subproblems.push_back(SP_indices[i]); // add the block_id selected to compute upper bound to vector which
																									 // keeps track of them for later use 
			}
		}
		else
		{
			n_efgs = floor( UB_FRACTION );

			for (int i = 0; i < n_efgs; i++)
			{
				efgs_subproblems.push_back(SP_indices[i]);
			}

			double fraction_efgs_right_away = 0.10;
			for (int i = 0; i < ceil(fraction_efgs_right_away * n_efgs); i++)	// send 10% of efgs problem right away
				indicator_efgs[SP_indices[i]] = UB_SOL_TYPE; //the first n_efgs entries in the shuffled SP_indices will compute EFGS
			for (int i = 0; i < n_efgs - ceil(fraction_efgs_right_away * n_efgs); i++)	// send the remaining 90% of efgs problem at the very end
				indicator_efgs[SP_indices[N_SUBPROBLEMS-i-1]] = UB_SOL_TYPE; //the last n_efgs entries in the shuffled SP_indices will compute EFGS
		}

		/* I comment out the bounding code for EFGS MASTER because I am only testing my parallel upper bounding
		 * code right now */
		/* 

		// send out the next sub-problem when a worker asks for one
		next_SP = 0;
		while (next_SP < N_SUBPROBLEMS){
			// wait until we get a request from a worker
			MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

			// find out who asked for a new sub-problem
			int requestor = status.MPI_SOURCE;

			if (msg != REQUEST)
				n_completed++;

			if (N_SUBPROBLEMS - n_completed < nproc-1)
				n_inprogress = N_SUBPROBLEMS - n_completed;


			// for loadbalancing purposes, we want the longer running problems assigned first
			// any subproblem that also computes efgs necessary has a longer runtime than that only computes egso
			// so we first solve all subproblems that require computing efgs
			param.block_id = SP_indices[next_SP];

			// indicate if this subproblem should also compute efgs
			param.EFGS_type = indicator_efgs[param.block_id];
			//			std::cout << "next_SP:" <<next_SP << "\tindicator_efgs[next_sP]:" << indicator_efgs[next_SP] << std::endl;
			std::cout << "no. to solve= " << N_SUBPROBLEMS - n_completed << "\tno in progress= " << n_inprogress << "\tnext_SP= " << next_SP << std::endl;

			// set up the EEV_indicator only for the first subproblem for which EFGS is not calculated, then turn it off
			if (CALCULATE_EEV == 1 and param.EFGS_type == 0)
			{
				param.EEV_indicator = 1;
				CALCULATE_EEV = 0;
			}
			else
			{
				param.EEV_indicator = 0;
			}

			//			std::cout << "next_SP:" << next_SP << "\tparam.EEV_indicator:" << param.EEV_indicator << std::endl;

			// send the requestor a new sub-problem to solve
			MPI_Send(&param, 1, obj_type, requestor, 0, MPI_COMM_WORLD);

			// we have sent a sub-problem to another worker
			next_SP++;
		}
		*/

		/* Send SWITCH_EFGS_SIGNAL to all worker nodes */
		num_switch_mode = 0;
		while (num_switch_mode < num_workers) 
		{
			/* wait until we get a request from a worker */
			MPI_Recv (&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

			/* find out who asked for a new sub-problem */
			int requestor = status.MPI_SOURCE;

			/* send requestor a sginal to change to EFGS mode */
			param.block_id = SWITCH_EFGS_MODE;
			MPI_Send (&param, 1, obj_type, requestor, 0, MPI_COMM_WORLD);

			/* we have sent a switch mode signal to another worker */
			num_switch_mode++;
			if (print_mw_signals == true)
			{
				cout << "num switch to EFGS mode = " << num_switch_mode 
					<< ", num_workers: " << num_workers << endl;
			}

			/* now send a SLEEP signal, so that worker requests are only those of the EGSO mode */
			/* No longer need to send SLEEP signal, worker goes to sleep automatically
				 param_EFGS param_efgs_send;
				 param_efgs_send.master_to_worker_signal = SLEEP;
				 MPI_Send (&param_efgs_send, 1, obj_type_EFGS, requestor, 0, MPI_COMM_WORLD); 
				 */
			if (print_mw_signals == true)
			{
				cout << "sleep signal sent to worker: " << requestor << endl;
			}
		}

		/* Now wake all the workers back up */
		cout << "now master is waking up all workers, for EFGS mode" << endl;
		for (int k = 1; k <= num_workers; k++)
		{
			if (print_mw_signals == true) 
			{
				cout << "master: sending signal AWAKE to worker, switch EFGS mode: " << k << endl;
				printf("k: %d, num_workers: %d\n", k, num_workers);
			}
			param_EFGS_master_to_worker param_efgs_send;
			param_efgs_send.master_to_worker_signal = AWAKE;
			MPI_Send (&param_efgs_send, 1, obj_type_EFGS_master_to_worker, k, 0, MPI_COMM_WORLD);

			if (print_mw_signals == true) 
			{
				cout << "master: sent signal AWAKE to worker, switch EFGS mode: " << k << endl;
			}
		}

		/* EFGS MASTER
		 * ------------------------------------------------------------------------
		 * ------------------------------------------------------------------------
		 */

		/* NOTE: add efgs_subproblems */
		int master_loop = efgs_subproblems.size(); //perform this many upper bounding calculations
		for (int i = 0; i < master_loop; i++)
		{
			if (print_mpi_signals == true) printf("i:%d, master_loop:%d\n", i, master_loop);

			/* First generate instances of needed variables. TODO */
			int block_id = efgs_subproblems.back();
			efgs_subproblems.pop_back();
			int EFGS_type = 0; /* not important, should be turned off */

			/* Setting up stree */
			std::ostringstream parameter_file, partition_file, result_file;
			parameter_file << PARAMETER_FILE << "." << rank;
			result_file << RESULT_FILE << "." << rank;
			partition_file << PARTITION_FILE << "." << rank;
			/*
			ScenarioTree stree = generate_ScenarioTree(
					PROBLEM_CLASS.c_str(),
					parameter_file.str().c_str(),
					N_INSTANCES,
					result_file.str().c_str(),
					MAX_NUM_NODES_IN_A_PROBLEM,
					LB_STOP_CR,
					LB_STOP_TH,
					partition_file.str().c_str(),
					block_id, 
					EFGS_type, 
					UB_STOP_CR,
					UB_STOP_TH,
					UB_SOL_PARAM,
					rank,
					SOLVE_FULL_PROB,
					EQUALLY_LIKELY_SCENARIOS,
					UNIFORM_MULTIPLICITY,
					CALCULATE_EEV);
		*/

			/* Testing Queue functions */
			if (Queue_diagnostics == true)
			{
				printf("about to call test_Queue()\n");
				test_Queue(
						PROBLEM_CLASS.c_str(),
						parameter_file.str().c_str(),
						N_INSTANCES,
						result_file.str().c_str(),
						MAX_NUM_NODES_IN_A_PROBLEM,
						LB_STOP_CR,
						LB_STOP_TH,
						partition_file.str().c_str(),
						block_id, 
						EFGS_type, 
						UB_STOP_CR,
						UB_STOP_TH,
						UB_SOL_PARAM,
						rank,
						SOLVE_FULL_PROB,
						EQUALLY_LIKELY_SCENARIOS,
						UNIFORM_MULTIPLICITY,
						CALCULATE_EEV);
				printf("done with test_Queue()\n");

				/* Practice Queue */
				vector< vector<int> > Queue;
				vector<int> timestage_1;
				vector<int> timestage_2;
				vector<int> timestage_3;
				timestage_1.push_back(0);
				timestage_2.push_back(1);
				timestage_2.push_back(2);
				timestage_3.push_back(3);
				timestage_3.push_back(4);
				timestage_3.push_back(5);
				timestage_3.push_back(6);
				Queue.push_back(timestage_1);
				Queue.push_back(timestage_2);
				Queue.push_back(timestage_3);
			}

	
		/* some perparation of input parameters for prepare_efgs() */
			int prep_status = -1;
			double efgs = 0;
			map<unsigned int, float> rhs_contribution;
			double optimization_time = 0;
			vector< vector<int> > Queue;

			if (print_main_functions == true) printf("master: prepare_efgs() function calling\n");
			ScenarioTree stree = prepare_efgs(
					PROBLEM_CLASS.c_str(),
					parameter_file.str().c_str(),
					N_INSTANCES,
					result_file.str().c_str(),
					MAX_NUM_NODES_IN_A_PROBLEM,
					LB_STOP_CR,
					LB_STOP_TH,
					partition_file.str().c_str(),
					block_id, 
					EFGS_type, 
					UB_STOP_CR,
					UB_STOP_TH,
					UB_SOL_PARAM,
					rank,
					SOLVE_FULL_PROB,
					EQUALLY_LIKELY_SCENARIOS,
					UNIFORM_MULTIPLICITY,
					CALCULATE_EEV,
					rhs_contribution,
					efgs,
					optimization_time,
					Queue);
			printf("master: prepare_efgs() efgs=%f, optimization_time=%f\n", efgs, optimization_time);	
		
			if (print_main_functions == true)
			{	
				/* printf("master: "); */
				/* print_problem_settings( *(stree.Asettings) ); */
				printf("master: prepare_efgs() function called\n");
				printf("testing out our generated scenario tree: num_stages=%d, num_branches=%d, total_num_cols=%d, total_num_rows=%d\n", 
						stree.num_stages, stree.num_branches, stree.total_num_cols, stree.total_num_rows);
			}

			/* Iterate through number of time stages */
			int num_timestages = Queue.size();

			if (print_mpi_signals == true) cout << "num_timestages: " << num_timestages << endl;

			for (int j = 0; j < num_timestages; j++)
			{
				if (print_mpi_signals == true) cout << "Time stage: " << j+1 << endl;

				/* inner while-loop for each timestage */
				num_sleep = 0;
				while (num_sleep < num_workers)
				{
					/* listen for message from worker */
					param_EFGS_worker_to_master param_efgs_recv;
					MPI_Recv (&param_efgs_recv, 1, obj_type_EFGS_worker_to_master, 
							MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

					/* find out who the message came from */
					int requestor = status.MPI_SOURCE;

					if (print_mw_signals == true)
					{
						cout << "master received message from worker: " << requestor 
							<< endl << "        worker_to_master_signal: " 
							<< param_efgs_recv.worker_to_master_signal << endl; 
					}

					/* multiple conditions to determine how to respond */
					if (param_efgs_recv.worker_to_master_signal == REQUEST)
					{
						if (print_mw_signals == true)
						{
							cout << "master received signal REQUEST from worker: " << requestor << endl;
						}
						/* if no more jobs */
						if (Queue[j].empty() == true)
						{
							/* set the sleep signal */
							param_EFGS_master_to_worker param_efgs_send;
							param_efgs_send.master_to_worker_signal = SLEEP;

							/* send the sleep signal */
							MPI_Send (&param_efgs_send, 1, obj_type_EFGS_master_to_worker, requestor, 0, MPI_COMM_WORLD);
							if (print_mw_signals == true)
							{
								cout << "master: sending signal SLEEP to worker: " 
									<< requestor << endl;
							}
							/* iterate sleep variable */
							num_sleep++;
						}
						else /* there are remaining jobs */
						{
							/* set SUBMIT signal and problem data */
							param_EFGS_master_to_worker param_efgs_send;
							param_efgs_send.master_to_worker_signal = SUBMIT;
							int node_ID = Queue[j].back();
							param_efgs_send.node_ID = node_ID;
							Queue[j].pop_back();
							param_efgs_send.block_id = block_id;
							param_efgs_send.rhs_contribution_update = rhs_contribution[stree.node_row_start(stree.ancestor(node_ID))];

							MPI_Send (&param_efgs_send, 1, obj_type_EFGS_master_to_worker, requestor, 0, MPI_COMM_WORLD);
							if (print_mw_signals == true)
							{
								cout << "master; sending signal SUBMIT to worker: " 
									<< requestor << endl;
							}
						}
					}
					/* Worker sends back an answer, master receives */
					else if (param_efgs_recv.worker_to_master_signal == ANSWER)
					{
						if (print_mw_signals == true)
						{
							cout << "master received signal ANSWER from worker: " << requestor << endl;
						}
						efgs += param_efgs_recv.efgs_contribution;
						optimization_time += param_efgs_recv.optimization_time_contribution;
						int node_ID = param_efgs_recv.node_ID;
						rhs_contribution[stree.node_row_start(node_ID)] = param_efgs_recv.rhs_contribution_return;
						if (print_contributions == true)
						{
							printf("master: efgs=%f, efgs_contribution=%f, optimization_time_contribution=%f\n"
									, efgs, param_efgs_recv.efgs_contribution, param_efgs_recv.optimization_time_contribution);
						}
						/* note: EFGS_MPI_gap I don't use */
					}
					else
					{
						if (print_mw_signals == true)
						{
							cout << "param_efgs_recv.worker_to_master_signal: " 
								<< param_efgs_recv.worker_to_master_signal << endl;
						}
						printf("ERROR: worker_to_master_signal not set properly, signal from worker %d\n", requestor);
					}
				}

				/* Now we AWAKE all workers in this timestage */
				for (int k = 1; k <= num_workers; k++)
				{
					if (print_mw_signals == true)
					{
						cout << "master: sending signal AWAKE to worker: " << k << endl;
					}
					param_EFGS_master_to_worker param_efgs_send;
					param_efgs_send.master_to_worker_signal = AWAKE;
					MPI_Send (&param_efgs_send, 1, obj_type_EFGS_master_to_worker, k, 0, MPI_COMM_WORLD);
					if (print_mw_signals == true)
					{
						cout << "master: sent signal AWAKE to worker: " << k << endl;
					}
				}
			}

			/*add these to master list */
			efgs_candidates.push_back(efgs);
			optimization_time_candidates.push_back(optimization_time);
			printf("master: master_loop: %d, efgs: %f, optimization_time: %f\n", 
					master_loop, efgs, optimization_time);
		}

		/* We find the minimum efgs_candidate, the lowest upper bound */
		double efgs_min = efgs_candidates[0];
		double optimization_time_efgs_min = 0;
		int efgs_min_index = 0;
		for (int i = 0; i < efgs_candidates.size(); i++)
		{
			if (efgs_candidates[i] < efgs_min)
			{
				efgs_min = efgs_candidates[i];
				efgs_min_index = i;
			}
		}
		optimization_time_efgs_min = optimization_time_candidates[efgs_min_index];
		cout << "efgs_min: " << efgs_min << " optimization_time: " << optimization_time_efgs_min << endl;


		// all sub-problems have been sent out, start sending shutdown signals
		n_shutdown = 0;
		while (n_shutdown < nproc-1){
			// wait until we get a request from a worker
			param_EFGS_worker_to_master param_efgs_recv;
			MPI_Recv (&param_efgs_recv, 1, obj_type_EFGS_worker_to_master, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if (print_mw_signals == true)
			{
				cout << "master awaiting contact from worker, to send SHUTDOWN signal" 
					<< endl;
			}

			// find out who asked for a new sub-problem
			int requestor = status.MPI_SOURCE;

			// send the requestor a shutdown signal
			param_EFGS_master_to_worker param_efgs_send;
			param_efgs_send.master_to_worker_signal = SHUTDOWN;
			MPI_Send (&param_efgs_send, 1, obj_type_EFGS_master_to_worker, requestor, 0, MPI_COMM_WORLD);
			if (print_mw_signals == true)
			{
				cout << "master: sending signal SHUTDOWN to worker: " << requestor << endl;
			}

			// we have sent a shutdown signal to another worker
			n_shutdown++;
			printf("num_shutdown=%d\n", n_shutdown);
		}


		/* EGSO MASTER, finishing up
		 * ------------------------------------------------------------------------
		 * ------------------------------------------------------------------------
		 */
		/*
			 std::cout<< "Master time:" << get_cpu_time()- master_time<<std::endl;

			 std::ofstream final_result_file( RESULT_FILE.c_str() ) ;
		// concatenate the result files from each worker
		for (int worker_id = 1; worker_id < nproc; worker_id++)
		{
		std::ostringstream result_filename;
		result_filename << RESULT_FILE << "." << worker_id;
		std::ifstream result_file( result_filename.str().c_str() ) ;

		final_result_file << result_file.rdbuf();
		}
		final_result_file.close();

		cout << "Summarizing result file for progress plots..." << flush;
		double total_time = progressSummary(RESULT_FILE.c_str(), nproc);
		cout << "DONE" << endl;
		*/
	}
	/* EGSO WORKER
	 * --------------------------------------------------------------------------
	 * --------------------------------------------------------------------------
	 */
			else {	// worker
				/* variable for worker mode */
				bool worker_mode_EGSO = true;

				param.block_id = REQUEST;
				while (worker_mode_EGSO == true)
				{
					// ask rank0 for a sub-problem to solve
					MPI_Send(&param.block_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					cout << "worker: " << rank << 
						" asking master for a subproblem to solve" << endl;

					// receive from rank0 the sub-problem to solve
					MPI_Recv(&param, 1, obj_type, 0, 0, MPI_COMM_WORLD, &status);

					if (param.block_id == ALLDONE){
						// no more sub-problems to solve
						printf("Worker %d worked on %d tasks and got shutdown signal\n", rank, njobs[rank]);
						break;
					}
					else if (param.block_id == SWITCH_EFGS_MODE) /* switching signal */
					{
						cout << "SWITCH_EFGS_MODE received, rank: " << rank << endl;
						worker_mode_EGSO = false;
					}
					else {
						std::ostringstream parameter_file, partition_file, result_file;
						parameter_file << PARAMETER_FILE << "." << rank;
						result_file << RESULT_FILE << "." << rank;
						partition_file << PARTITION_FILE << "." << rank;
						// solve my_SP sub-problem
						njobs[rank]++;
						std::ostringstream runCMD;
						runCMD	<< EXE_BOUNDING.c_str() << " " 
							<< PROBLEM_CLASS.c_str() << " "
							<< parameter_file.str().c_str() << " "
							<< N_INSTANCES << " "
							<< result_file.str().c_str() << " "
							<< MAX_NUM_NODES_IN_A_PROBLEM << " "
							<< LB_STOP_CR << " " 
							<< LB_STOP_TH << " "
							<< partition_file.str().c_str() << " "
							<< param.block_id << " "
							<< param.EFGS_type << " "
							<< UB_STOP_CR << " "
							<< UB_STOP_TH << " "
							<< UB_SOL_PARAM << " "
							<< rank << " "
							<< SOLVE_FULL_PROB << " "
							<< EQUALLY_LIKELY_SCENARIOS << " "
							<< UNIFORM_MULTIPLICITY << " "
							<< param.EEV_indicator;

						if (param.EFGS_type != 0)
							std::cout << "Worker " << rank << " (job " << njobs[rank] << ") contains an EFGS calculation." << std::endl;
						if (param.EEV_indicator == 1)
							std::cout << "Worker " << rank << " (job " << njobs[rank] << ") contains VSS calculation." << std::endl;
						//printf("Worker %d (job %d) will run: %s\n", rank, njobs[rank], runCMD.str().c_str());
						system(runCMD.str().c_str());
						// sleep(1);
					}
				}		
				//		std::cout << "cumulative worker " << rank<< " time:" << get_cpu_time() - time_begin << std::endl;

				/* EFGS WORKER
				 * ------------------------------------------------------------------------
				 * ------------------------------------------------------------------------
				 */
				bool sleep;

				/* Workers sleep while SWITCH_EFGS_MODE messages are sent out to rest */
				sleep = true;
				while (sleep == true)
				{
					// wait for awake signal 
					param_EFGS_master_to_worker param_efgs_recv;
					MPI_Recv (&param_efgs_recv, 1, obj_type_EFGS_master_to_worker, 0, 0, MPI_COMM_WORLD, &status);
					if (print_mw_signals == true)
					{
						cout << "worker: " << rank 
							<< " alseep after SWITCH_EFGS_MODE, awaiting signal AWAKE" << endl;
					}

					if (param_efgs_recv.master_to_worker_signal == AWAKE)
					{
						sleep = false;
					}
				}
				if (print_mw_signals == true)
				{
					cout << "worker: " << rank 
						<< " has received AWAKE signal, ready for EFGS mode" << endl;
				}

				while (1)
				{
					/* ask master for suproblem to solve */
					param_EFGS_worker_to_master param_efgs_send;
					param_efgs_send.worker_to_master_signal = REQUEST;

					MPI_Send (&param_efgs_send, 1, obj_type_EFGS_worker_to_master, 0, 0, MPI_COMM_WORLD);
					if (print_mw_signals == true)
					{
						cout << "worker: " << rank << " sent signal REQUEST: " 
							<< param_efgs_send.worker_to_master_signal << endl;
					}

					/* receive response from master */
					param_EFGS_master_to_worker param_efgs_recv;
					MPI_Recv (&param_efgs_recv, 1, obj_type_EFGS_master_to_worker, 0, 0, MPI_COMM_WORLD, &status);
					if (print_mw_signals == true)
					{
						cout << "worker: " << rank << " received response from master" << endl
							<< "        master_to_worker_signal: " << 
							param_efgs_recv.master_to_worker_signal << endl;;
					}

					/* now check conditions to determine the message */
					if (param_efgs_recv.master_to_worker_signal == SUBMIT)
					{
						/* Have a node to solve */
						/* load parameters */
						int node_ID = param_efgs_recv.node_ID;
						float rhs_contribution_update = param_efgs_recv.rhs_contribution_update;
						int block_id = param_efgs_recv.block_id;
						int EFGS_type = 0;				
						double EFGS_MIP_gap = 0;
						double efgs_contribution = 0;
						float rhs_contribution_return = 0;
						double optimization_time_contribution = 0;

						/* Setting up stree */
						std::ostringstream parameter_file, partition_file, result_file;
						parameter_file << PARAMETER_FILE << "." << rank;
						result_file << RESULT_FILE << "." << rank;
						partition_file << PARTITION_FILE << "." << rank;
						/*
							 ScenarioTree stree = generate_ScenarioTree(
							 PROBLEM_CLASS.c_str(),
							 parameter_file.str().c_str(),
							 N_INSTANCES,
							 result_file.str().c_str(),
							 MAX_NUM_NODES_IN_A_PROBLEM,
							 LB_STOP_CR,
							 LB_STOP_TH,
							 partition_file.str().c_str(),
							 block_id, 
							 EFGS_type, 
							 UB_STOP_CR,
							 UB_STOP_TH,
							 UB_SOL_PARAM,
							 rank,
							 SOLVE_FULL_PROB,
							 EQUALLY_LIKELY_SCENARIOS,
							 UNIFORM_MULTIPLICITY,
							 CALCULATE_EEV);
							 printf("worker: ");
							 print_problem_settings( *(stree.Asettings) );				
							 */
						/* Call the worker function */
						int worker_status = -1;
						worker_status = node_solve_timestage_method(
								PROBLEM_CLASS.c_str(),
								parameter_file.str().c_str(),
								N_INSTANCES,
								result_file.str().c_str(),
								MAX_NUM_NODES_IN_A_PROBLEM,
								LB_STOP_CR,
								LB_STOP_TH,
								partition_file.str().c_str(),
								block_id, 
								EFGS_type, 
								UB_STOP_CR,
								UB_STOP_TH,
								UB_SOL_PARAM,
								rank,
								SOLVE_FULL_PROB,
								EQUALLY_LIKELY_SCENARIOS,
								UNIFORM_MULTIPLICITY,
								CALCULATE_EEV,

								rank,
								node_ID,
								rhs_contribution_update,
								EFGS_MIP_gap,
								efgs_contribution,
								rhs_contribution_return,
								optimization_time_contribution);

						if (print_mw_signals == true)
						{
							cout << "worker: " << rank << "done the computation" << endl;
						}

						/* set parameters for return */
						param_EFGS_worker_to_master param_efgs_send;
						param_efgs_send.worker_to_master_signal = ANSWER;
						param_efgs_send.node_ID = node_ID;
						param_efgs_send.efgs_contribution = efgs_contribution;
						param_efgs_send.rhs_contribution_return = rhs_contribution_return;
						param_efgs_send.optimization_time_contribution = optimization_time_contribution;

						MPI_Send (&param_efgs_send, 1, obj_type_EFGS_worker_to_master, 0, 0, MPI_COMM_WORLD);
						if (print_mw_signals == true)
						{
							cout << "worker: " << rank << " sent signal ANSWER: " 
								<< param_efgs_send.worker_to_master_signal << endl;
						}
					}
					else if (param_efgs_recv.master_to_worker_signal == SLEEP)
					{ /* SLEEP signal, wait until next time stage */
						sleep = true;
						while (sleep == true)
						{
							/* wait for AWAKE signal */
							param_EFGS_master_to_worker param_efgs_recv;
							MPI_Recv (&param_efgs_recv, 1, obj_type_EFGS_master_to_worker, 0, 0, MPI_COMM_WORLD, &status);
							if (print_mw_signals == true)
							{
								cout << "worker: " << rank 
									<< " currently asleep. waiting for AWAKE signal" << endl;
							}
							if (param_efgs_recv.master_to_worker_signal == AWAKE)
							{
								sleep = false;
								if (print_mw_signals == true)
								{
									cout << "worker: " << rank << " received response from master"
										<< endl << "master_to_worker_signal: " <<
										param_efgs_recv.master_to_worker_signal << endl;
								}
							}
						}
					}
					else if (param_efgs_recv.master_to_worker_signal == SHUTDOWN)
					{ /* SHUTDOWN signal */
						printf("worker %d got shutdown signal\n", rank);
						break;
					}
					else /* ERROR */
					{
						printf("ERROR: param_efgs_recv.master_to_worker_signal not set properly in worker %d\n", rank);
					}
				}
			}

			// Terminate

			MPI_Type_free(&obj_type);
			MPI_Type_free(&obj_type_EFGS_master_to_worker);
			MPI_Type_free(&obj_type_EFGS_worker_to_master);
			MPI_Finalize();
			
			// print out optimization time for entire main function
			optimization_time_main = get_cpu_time() - optimization_time_start;
			printf("optimization_time_main: %d\n", optimization_time_main);
			return(0);

}
