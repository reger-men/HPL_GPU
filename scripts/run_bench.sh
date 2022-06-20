#1GPU
echo "1GPU"
cp ./config/HPL_1GPU.dat ./HPL.dat
sh ./mpirun_hpl.sh


#2GPU
echo "2GPU 1xMI250"
cp ./config/HPL_2GPU.dat ./HPL.dat
sh ./mpirun_hpl.sh

#4GPU
echo "4GPU 2xMI250"
cp ./config/HPL_4GPU.dat ./HPL.dat
sh ./mpirun_hpl.sh

#8GPU
echo "8GPU 4xMI250"
cp ./config/HPL_8GPU.dat ./HPL.dat
sh ./mpirun_hpl.sh
