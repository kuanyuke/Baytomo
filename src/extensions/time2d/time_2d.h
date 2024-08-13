#ifndef TIME_2D_H
#define TIME_2D_H

#define NO_ERROR 0
#define ERR_INFBUG (-1)
#define ERR_MULTUNI (-2)
#define ERR_MALLOC (-3)
#define ERR_RECURS (-4)
#define ERR_EPS (-5)
#define ERR_RANGE (-6)
#define ERR_PHYS (-7)
#define ERR_DIM (-8)

int time_2d(float *HS, float *T, int NX, int NY, float XS, float YS, float EPS_INIT, int MESSAGES);

#endif // TIME_2D_H

