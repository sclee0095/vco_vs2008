#ifndef _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_
#define _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_

#include <math.h>
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>

//// some global variables
//extern int n;			// width
//extern int p;			// height
//extern double* D;
//extern double* S;
//extern double* W;
//extern double* Q;
//extern double* L;
//extern double* PD; // SCLEE: distance in pixels
//extern double* start_points;
//extern double* end_points;
//extern double* H;
//extern double* values;
//extern int nb_iter_max;
//extern int nb_start_points;
//extern int nb_end_points;

typedef bool (*T_callback_intert_node)(int i, int j, int ii, int jj);

// main function
void perform_front_propagation_2d_addpd(T_callback_intert_node callback_insert_node = NULL);
void perform_front_propagation_2d_addpd(
	int i_n,
	int i_p,
	double* i_D,
	double* i_S,
	double* i_W,
	double* i_Q,
	double* i_L,
	double* i_PD,
	double* i_start_points,
	double* i_end_points,
	double* i_H,
	double* i_values,
	int i_nb_iter_max,
	int i_nb_start_points,
	int i_nb_end_points,
	double **o_D,
	T_callback_intert_node callback_insert_node = NULL);

#endif // _PERFORM_FRONT_PROPAGATION_2D_ADDPD_H_