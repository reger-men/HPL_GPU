#include "backend/hpl_backendCommon.h"
#include "hpl.h"

void HPL_piplen(HPL_T_panel* PANEL,
                const int    K,
                const int*   IPID,
                int*         IPLEN,
                int*         IWORK) {

  const int nprow   = PANEL->grid->nprow;
  const int jb      = PANEL->jb;
  const int nb      = PANEL->nb;
  const int ia      = PANEL->ia;
  const int icurrow = PANEL->prow;

  int* iwork = IWORK + jb;

  /*
   * Compute IPLEN
   */
  for(int i = 0; i <= nprow; i++) IPLEN[i] = 0;

  for(int i = 0; i < K; i += 2) {
    const int src = IPID[i];
    int       srcrow;
    Mindxg2p(src, nb, nb, srcrow, 0, nprow);
    if(srcrow == icurrow) {
      const int dst = IPID[i + 1];
      int       dstrow;
      Mindxg2p(dst, nb, nb, dstrow, 0, nprow);
      if((dstrow != srcrow) || (dst - ia < jb)) IPLEN[dstrow + 1]++;
    }
  }

  for(int i = 1; i <= nprow; i++) { IPLEN[i] += IPLEN[i - 1]; }
}

void HPL_plindx(HPL_T_panel* PANEL, const int K, const int* IPID, int* IPA, int* LINDXU, int* LINDXAU,
int* LINDXA, int* IPLEN, int* PERMU, int* IWORK) {

  const int myrow   = PANEL->grid->myrow;
  const int nprow   = PANEL->grid->nprow;
  const int jb      = PANEL->jb;
  const int nb      = PANEL->nb;
  const int ia      = PANEL->ia;
  const int iroff   = PANEL->ii;
  const int icurrow = PANEL->prow;

  int* iwork = IWORK + jb;

  HPL_piplen(PANEL, K, IPID, IPLEN, IWORK);


  if(myrow == icurrow) {
    // for all rows to be swapped
    int ip = 0, ipU = 0;
    for(int i = 0; i < K; i += 2) {
      const int src = IPID[i];
      int       srcrow;
      Mindxg2p(src, nb, nb, srcrow, 0, nprow);

      if(srcrow == icurrow) { // if I own the src row
        const int dst = IPID[i + 1];
        int       dstrow;
        Mindxg2p(dst, nb, nb, dstrow, 0, nprow);

        int il;
        Mindxg2l(il, src, nb, nb, myrow, 0, nprow);

        if((dstrow == icurrow) && (dst - ia < jb)) {
          // if I own the dst and it's in U

          PERMU[ipU]  = dst - ia;      // row index in U
          iwork[ipU]  = IPLEN[dstrow]; // Index in AllGathered U
          ipU++;

          LINDXU[IPLEN[dstrow]] = il - iroff; // Index in AllGathered U
          IPLEN[dstrow]++;
        } else if(dstrow != icurrow) {
          // else if I don't own the dst

          // Find the IPID pair with dst as the source
          int j = 0;
          int fndd;
          do {
            fndd = (dst == IPID[j]);
            j += 2;
          } while(!fndd && (j < K));
          // This pair must have dst being sent to a position in U

          PERMU[ipU]  = IPID[j - 1] - ia; // row index in U
          iwork[ipU]  = IPLEN[dstrow];    // Index in AllGathered U
          ipU++;

          LINDXU[IPLEN[dstrow]] = il - iroff;    // Index in AllGathered U
          IPLEN[dstrow]++;
        } else if((dstrow == icurrow) && (dst - ia >= jb)) {
          //else I own the dst, but it's not in U

          LINDXAU[ip] = il - iroff; //the src row must be in the first jb rows

          int il;
          Mindxg2l(il, dst, nb, nb, myrow, 0, nprow);
          LINDXA[ip] = il - iroff; //the dst is somewhere below
          ip++;
        }
      }
    }
    *IPA = ip;
  } else {
    // for all rows to be swapped
    int ip = 0, ipU = 0;
    for(int i = 0; i < K; i += 2) {
      const int src = IPID[i];
      int       srcrow;
      Mindxg2p(src, nb, nb, srcrow, 0, nprow);
      const int dst = IPID[i + 1];
      int       dstrow;
      Mindxg2p(dst, nb, nb, dstrow, 0, nprow);

      if(myrow == dstrow) { // if I own the dst row
        int il;
        Mindxg2l(il, dst, nb, nb, myrow, 0, nprow);
        LINDXU[ip] = il - iroff; // Local A index of incoming row
        ip++;
      }

      if(srcrow == icurrow) {

        if((dstrow == icurrow) && (dst - ia < jb)) {
          // If the row is going into U
          PERMU[ipU] = dst - ia;      // row index in U
          iwork[ipU] = IPLEN[dstrow]; // Index in AllGathered U
          IPLEN[dstrow]++;
          ipU++;
        } else if(dstrow != icurrow) {
          // If the row is going to another rank
          // (So src must be in U)

          // Find the IPID pair with dst as the source
          int j = 0;
          int fndd;
          do {
            fndd = (dst == IPID[j]);
            j += 2;
          } while(!fndd && (j < K));
          // This pair must have dst being sent to a position in U

          PERMU[ipU] = IPID[j - 1] - ia; // row index in U
          iwork[ipU] = IPLEN[dstrow];    // Index in AllGathered U
          IPLEN[dstrow]++;
          ipU++;
        }
      }
    }
    *IPA = 0;
  }

  HPL_perm(jb, iwork, PERMU, IWORK);
  /*
   * Reset IPLEN to its correct value
   */
  for(int i = nprow; i > 0; i--) IPLEN[i] = IPLEN[i - 1];
  IPLEN[0] = 0;
}

int HPL_scatterv(double* BUF, const int* SCOUNT, const int* DISPL,
                 const int RCOUNT, int ROOT, MPI_Comm COMM) {
  int rank, ierr;
  MPI_Comm_rank(COMM, &rank);

  if (rank==ROOT) {
    ierr = MPI_Scatterv(BUF, SCOUNT, DISPL, MPI_DOUBLE, MPI_IN_PLACE,  RCOUNT, MPI_DOUBLE, ROOT, COMM);
  } else {
    ierr = MPI_Scatterv(NULL, SCOUNT, DISPL, MPI_DOUBLE, BUF, RCOUNT, MPI_DOUBLE, ROOT, COMM);
  }

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}

int HPL_allgatherv(double* BUF, const int SCOUNT, const int* RCOUNT,
                   const int* DISPL, MPI_Comm COMM) {

  int ierr = MPI_Allgatherv(MPI_IN_PLACE, SCOUNT, MPI_DOUBLE, BUF, RCOUNT, DISPL, MPI_DOUBLE, COMM);

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}

int HPL_pdpanel_bcast(HPL_T_panel* PANEL) {

  if(PANEL == NULL) {
    return HPL_SUCCESS;
  }
  if(PANEL->grid->npcol <= 1) {
    return HPL_SUCCESS;
  }
  MPI_Comm comm  = PANEL->grid->row_comm;
  int root  = PANEL->pcol;
  if(PANEL->len <= 0) return (HPL_SUCCESS);
  MPI_Request req;
  int ierr = MPI_Ibcast(PANEL->dL2, PANEL->len, MPI_DOUBLE, root, comm, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
