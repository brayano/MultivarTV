#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP _MultivarTV_gen_mesh(SEXP, SEXP, SEXP);
extern SEXP _MultivarTV_mbs(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _MultivarTV_mbspredict(SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_MultivarTV_gen_mesh",   (DL_FUNC) &_MultivarTV_gen_mesh,   3},
    {"_MultivarTV_mbs",        (DL_FUNC) &_MultivarTV_mbs,        9},
    {"_MultivarTV_mbspredict", (DL_FUNC) &_MultivarTV_mbspredict, 3},
    {NULL, NULL, 0}
};

void R_init_MultivarTV(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
