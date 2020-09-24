#include <stdio.h>  
#include <math.h>
#include <float.h>
#define GSL_MAX(a,b) ((a) > (b) ? (a) : (b))
#define GSL_SUCCESS 1
struct gsl_function_struct
{
	double(*function) (double x, void * params);
	void * params;
};

typedef struct gsl_function_struct gsl_function;


#define GSL_DBL_EPSILON        2.2204460492503131e-16
#define GSL_FN_EVAL(F,x) (*((F)->function))(x,(F)->params)



static void
central_deriv(const gsl_function * f, double x, double h,
	double *result, double *abserr_round, double *abserr_trunc)
{
	double fm1 = GSL_FN_EVAL(f, x - h);
	double fp1 = GSL_FN_EVAL(f, x + h);

	double fmh = GSL_FN_EVAL(f, x - h / 2);
	double fph = GSL_FN_EVAL(f, x + h / 2);

	double r3 = 0.5 * (fp1 - fm1);
	double r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;

	double e3 = (fabs(fp1) + fabs(fm1)) * GSL_DBL_EPSILON;
	double e5 = 2.0 * (fabs(fph) + fabs(fmh)) * GSL_DBL_EPSILON + e3;

	double dy = GSL_MAX(fabs(r3 / h), fabs(r5 / h)) *(fabs(x) / h) * GSL_DBL_EPSILON;
	*result = r5 / h;
	*abserr_trunc = fabs((r5 - r3) / h); 
	*abserr_round = fabs(e5 / h) + dy;   
}

int
gsl_deriv_central (const gsl_function * f, double x, double h,
                   double *result, double *abserr)
{
  double r_0, round, trunc, error;
  central_deriv (f, x, h, &r_0, &round, &trunc);
  error = round + trunc;

  if (round < trunc && (round > 0 && trunc > 0))
    {
      double r_opt, round_opt, trunc_opt, error_opt;
      double h_opt = h * pow (round / (2.0 * trunc), 1.0 / 3.0);
      central_deriv (f, x, h_opt, &r_opt, &round_opt, &trunc_opt);
      error_opt = round_opt + trunc_opt;
      if (error_opt < error && fabs (r_opt - r_0) < 4.0 * error)
        {
          r_0 = r_opt;
          error = error_opt;
        }
    }
  *result = r_0;
  *abserr = error;

  return GSL_SUCCESS;
}

double f(double x, void * params)
{
	(void)(params); /* avoid unused parameter warning */
	return pow(x, 1.5);
}


int main()
{
	gsl_function F;
	double result, abserr;

	F.function = &f;
	F.params = 0;

	printf("f(x) = x^(3/2)\n");

	gsl_deriv_central(&F, 3.0, 1e-8, &result, &abserr);
	printf("x = 2.0\n");
	printf("f'(x) = %.10f +/- %.10f\n", result, abserr);
	printf("exact = %.10f\n\n", 1.5 * sqrt(2.0));
	return 0;
}