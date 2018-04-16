// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// tell armadillo to use superlu for spsolve()
#define ARMA_USE_SUPERLU
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
#include "utils.hpp"

#define EPS 0.0001
using namespace arma; 
typedef std::vector<int> VEC;
typedef arma::mat MAT;

// FUNCTION DEFINITIONS

int prod(int p, VEC vec){
	int i;
	int dims_prod = 1;
	for ( i=0; i< p; i++)
		dims_prod *= vec[i];
	return dims_prod;
}

double prodd(vec avec){
	int i;
	double dims_prod = 1.0;
	for ( i=0; i< avec.size(); i++)
		dims_prod *= avec(i);
	return dims_prod;
}

VEC range(int min, int max){
	VEC vec(max-min+1); // If you do not know size, then do not declare and use vec.push_back(min+i); 
	for (int i=0; i<max-min+1; i++){
		vec[i] = min + i;
	}
	return vec;
}

int tensor2vector(int p, VEC multi_ind, VEC dims){
	int vec_ind;
	int i,j, dims_prod;
	vec_ind = multi_ind[0];
	for (i=1; i<p; i++){
		dims_prod=1;
		for (j=0; j< i; j++){
			dims_prod *= dims[j];
		}
		vec_ind += multi_ind[i]*dims_prod;
	}
	return vec_ind;
}

VEC vector2tensor(int p, int vec_ind, VEC dims){
	VEC multi_ind(p);
	int i, j, dims_prod;
	int ind2 = vec_ind + 1;

	for (i=p; i>0; i--){
		dims_prod=1;
		for (j=0; j< i-1; j++){
			dims_prod *= dims[j];
		}
		multi_ind[i-1] = std::max(1,(int)ceil((float)ind2/dims_prod)) - 1;
		ind2 -= multi_ind[i-1]*dims_prod;
	}
	return multi_ind;
}

VEC dec2binary(int n, int p){
	VEC binaryNum_long(1000);	// array to store binary number, allow big enough for large n
		// counter for binary array
	int i = 0;
	while (n > 0) { 
		binaryNum_long[i] = n % 2;  // storing remainder in binary array
		n = n / 2;
		i++;
	}
	VEC binaryNum(p);
  // printing binary array in reverse order
	for (int j = p-1; j >= p - i - 1; j--)
		binaryNum[j] = binaryNum_long[p-1-j];
	for (int j = 0; j < p- i - 2; j++)
		binaryNum[j] = 0;
	return binaryNum;
}

umat fd_binaries(int p){
	int i;
	int alpha = 1<<p;
	umat fd_bins(alpha-1, p);
	for (i=0; i < alpha - 1; i++){
		VEC binnum = dec2binary(i+1,p);
		uvec abinnum = conv_to<uvec>::from(binnum);
		fd_bins.row(i) = trans(abinnum);
	}
	return fd_bins;
}

uvec get_col_inds( int p, VEC dims, int direction){
	VEC ind(p,0);
	int i;
	int dims_prod = prod(p,dims);
	VEC col_inds;
	for ( i=0; i < dims_prod; i++){
		uvec arma_ind = conv_to<uvec>::from(ind);
		uvec arma_dims = conv_to<uvec>::from(dims);
		bool status = all(arma_dims - arma_ind > 0);
		if (status && ind[direction] + 1 < dims[direction]){
			col_inds.push_back( tensor2vector(p, ind, dims) );
			ind[direction] += 1;
			col_inds.push_back( tensor2vector(p, ind, dims));
		}
		ind = vector2tensor(p, i+1, dims);
	}
	uvec col_inds_uvec = conv_to<uvec>::from(col_inds);
	return col_inds_uvec; 
}
 // MUST DEFINE COL_INDS BEFORE ROW_INDS!!

uvec get_row_inds(uvec col_inds){
	VEC row_inds;
	int i;
	for (i=0; i<col_inds.n_rows/2; i++){
		row_inds.push_back(i);
		row_inds.push_back(i);
	}
	uvec row_inds_uvec = conv_to<uvec>::from(row_inds);
	return row_inds_uvec; 
}

vec get_vals(uvec col_inds){
	VEC val;
	int i;
	for (i=0; i<col_inds.n_rows/2; i++){
		val.push_back(1.0);
		val.push_back(-1.0);
	}
	vec vals = conv_to<vec>::from(val);
	return vals; 
}

sp_mat build_diffmat(int p, VEC dims, int direction){
	uvec col_ind = get_col_inds(p,dims,direction);
	uvec row_ind = get_row_inds(col_ind);
	umat locations(2,col_ind.n_rows);
	locations.row(0) = trans(row_ind);
	locations.row(1) = trans(col_ind);
	vec vals = get_vals(col_ind);
	sp_mat D(locations,vals);
	return D;
}

sp_mat mixedpartial(int p, VEC dims, uvec binary){
	VEC indices, newdims;
	int i,j;
	for (i=0; i<p; i++){
		if (binary[i]==1){
			indices.push_back(i);
		}
	}
	int n_mats = indices.size();
	std::vector<sp_mat> smats;
	mat I; I.eye(p,p);
	vec armadims = conv_to<vec>::from(dims);
	for (j=0; j<n_mats; j++){
		if (j==0){
			smats.push_back( build_diffmat(p,dims,0) );
		}
		else{
			armadims -= trans(I.row(indices[j-1]));
			newdims = conv_to<VEC>::from(armadims);
			smats.push_back( build_diffmat(p,newdims,indices[j]) );
		}
	}
	sp_mat D, Dnext;
	if (n_mats==1){
		return smats[0];
	}	
	else{
		D = smats[n_mats-1];
		int k;
		for (k = n_mats-2; k>= 0; k--){
			Dnext = smats[k];
			D = D*Dnext;
		}
		return D;
	}
}

sp_mat binary2diffmat(int p, VEC dims, uvec binary){
	sp_mat D;
	if (sum(binary)==1){
		VEC rng = range(0,p-1);
		uvec urng = conv_to<uvec>::from(rng);
		D = build_diffmat(p, dims, sum(urng % binary) );
	}
	else{
		D = mixedpartial(p,dims,binary);
	}
	return D;
}

vec ew_power(vec x, vec pows){
	int i;
	vec xpows(x.size());
	for (i=0; i< x.size(); i++){
		xpows[i] = pow(x[i],pows[i]);
	}
	return xpows;
}

sp_mat create_D(int p, vec dims, vec deltas){
	VEC meshdims = conv_to<VEC>::from(dims);
	umat binaries = fd_binaries(p);
	vec pones; pones.ones(p);
	int n_mats = binaries.n_rows;
	int i;
	sp_mat Di;
	sp_mat Dstack = binary2diffmat(p,meshdims,trans(binaries.row(n_mats-1)));
	for (i=0; i<n_mats-1; i++){
		double delts = prodd(ew_power(deltas, pones-binaries.row(i).t()) );
		Di = binary2diffmat(p,meshdims,binaries.row(i).t())*delts;
		Dstack = join_vert(Dstack, Di);
	}
	return Dstack; 
}

MAT create_mesh(mat data, vec dims){
	VEC meshdims = conv_to<VEC>::from(dims);
	int p = data.n_cols;
	int max_m = *max_element(meshdims.begin(),meshdims.end());
	int ntheta = prod(p,meshdims);
	mat unilat_mesh(p,max_m);
	int i,j;
	for (i=0; i<p; i++){
		unilat_mesh.row(i) = linspace<rowvec>( min(data.col(i))-EPS, max(data.col(i))+EPS,meshdims[i] );
	}
	MAT mesh(ntheta,p);
	for (i=0; i<ntheta; i++){
		VEC m_index = vector2tensor(p,i,meshdims);
		rowvec vals_m_index(p);
		for (j=0; j<p; j++){
			vals_m_index[j] = unilat_mesh(j,m_index[j]);
		}
		mesh.row( i ) = vals_m_index;
	}
	return mesh;
}

vec create_deltas(mat data, vec dims){
	vec deltas(data.n_cols);
	int i;
	for (i=0; i<data.n_cols; i++){
		deltas[i] = (max(data.col(i))-min(data.col(i)) + 2*EPS )/dims[i];
	}
	return deltas;
}

// Nearest point in n-dimensions

uword nearest1_unit(rowvec target, MAT choices){
	int ntheta = choices.n_rows;
	int i;
	vec dists(ntheta);
	for (i=0; i<ntheta; i++){
		//dists[i] = sum( pow(target-choices[i],2) );
		dists[i] = sum( pow(target-choices.row(i),2) );
	}
	uword mininds = dists.index_min();
	//uvec mininds = find( (dists - min(dists) )== 0);
	return mininds;
}

uvec nearest1( mat data, MAT mesh){
	uvec fit_inds(data.n_rows);
	int i;
	for (i=0; i< data.n_rows; i++){
		fit_inds[i] = nearest1_unit(data.row(i), mesh);
	}
	return fit_inds;
}

sp_mat nearest_interp_matrix(mat data, MAT mesh){
	// Get column index for nearest mesh value
	uvec col_ind = nearest1(data,mesh);
	// Generate numbers 0-- n-1 for each observation
	VEC row_ind_VEC = range(0,data.n_rows - 1);
	uvec row_ind = conv_to<uvec>::from(row_ind_VEC); // convert to arma
	// Generate coefficient, i.e. 1.0
	vec vals(data.n_rows); vals.fill(1);
	// Prepare for sp_mat
	umat locations(2,data.n_rows);
	locations.row(0) = row_ind.t();
	locations.row(1) = col_ind.t();
	// Define sp_mat
	sp_mat O(locations, vals, data.n_rows, mesh.n_rows);
	return O;
}

vec cg(sp_mat A, vec b){
	// Initialize x internally
	vec x(b.n_rows); x.fill(0.0); 
	vec d = b - A * x;
	vec r = A.t()*d;
	vec p = r;
	double rsold0 = norm(r);
	//Rcpp::Rcout << "rsold0 = " << rsold0 << std::endl;
	double rsold = pow(rsold0,2);
	double rsnew = rsold + 1.0;
	mat t = A * p;
	double alpha;
	int iter = 0;
	int MAXIT = b.n_rows < 2000 ? b.n_rows : 2000; // assign minimum(b.n_rows,1000)
	//Rcpp::Rcout << "MAXIT = " << MAXIT << std::endl;
	while (sqrt(rsnew) >= 0.0001*rsold0) {
		alpha = rsold / pow(norm(t),2);
		//Rcpp::Rcout << "rsold = " << rsold << std::endl;
		//Rcpp::Rcout << "rsnew = " << rsnew << std::endl;
		x += alpha * p;
		d -= alpha * t;
		r = A.t()*d;
		rsnew = pow(norm(r),2);
		iter += 1;
		if (iter == MAXIT){
		  //Rcpp::Rcout << "Convergence of Conjugate Gradient reached! " << std::endl;
		  break;
		}
		p = r + (rsnew / rsold) * p; // conjugate directions
		t = A * p;
		rsold = rsnew;
		//Rcpp::Rcout << "residual = " << rsnew << std::endl;
	}
	return x;
}


vec mypinv(sp_mat a, vec Oty){
	sp_mat ata = a.t()*a;
	// CONJUGATE GRADIENT to approximate lambda_max
	vec b = cg(ata,Oty);
	vec out = a*b;
	return out;
}

double lam_max_pinv(arma::sp_mat a, arma::vec Oty){
	vec A = mypinv(a,Oty);
  double tune = norm(A, 1); // else "inf"
  return tune; 
	//vec Apos = abs(A);
	//double tune = max(Apos);
	//return pow(tune,2); 
}

vec rowmean(mat A){
	vec rmeans(A.n_rows);
	int i;
	for (i=0; i<A.n_rows; i++){
		rmeans[i] = mean(A.row(i));
	}
	return rmeans;
}

// CROSS-VALIDATION FUNCTIONS/STRUCTS
// Modified to return list in case we want to expose to R.
Rcpp::List kfold(int k, arma::mat data, arma::vec y){
	int i;
	int ntest = data.n_rows/k;
	arma::field<arma::mat> Xtrain(k);
	arma::field<arma::mat> Xtest(k);
	arma::field<arma::vec> Ytrain(k);
	arma::field<arma::vec> Ytest(k);
	arma::mat datamat = arma::join_horiz(data,y);
	arma::mat datamat_shuffled = arma::shuffle(datamat, 1); // randomly shuffle rows
	for (i=0; i<k; i++){
		arma::mat Xshuffled = datamat_shuffled.cols(0,data.n_cols-1);
		arma::vec Yshuffled = datamat_shuffled.col(data.n_cols);
		int first = i*ntest; int last = (i+1)*ntest-1;
		Xtest[i] = Xshuffled.rows(first, last ) ;
		Ytest[i] = Yshuffled.rows(first, last) ;

		Xshuffled.shed_rows(first, last);
		Yshuffled.shed_rows(first, last);
		Xtrain[i] = Xshuffled ;
		Ytrain[i] = Yshuffled ;
	}
	return Rcpp::List::create(Rcpp::Named("Xtest")=Xtest,Rcpp::Named("Xtrain")=Xtrain,
                           Rcpp::Named("Ytest")=Ytest,Rcpp::Named("Ytrain")=Ytrain);
}

arma::vec kfoldinds( int n, int k ) {
  arma::vec indices(n);
  
  for (int i = 0; i < n; i++ )
    indices[ i ] = i%k;
  
  arma::vec inds_shuffled = arma::shuffle( indices,1 );
  
  return inds_shuffled;
}
