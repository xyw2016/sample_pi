
INC_DIRS=. $(NVSDKCOMPUTE_ROOT)/CL
LIBS=-lOpenCL

cl_spec.o:cl_spec.h cl_spec.cpp
	g++  -O3 -static -c -std=c++11 -Wall -DUNIX -g -DDEBUG cl_spec.cpp $(INC_DIRS:%=-I%)  $(LIBS)
main.o:main.cpp
	g++ -O3 -static -c --std=c++11 -Wall -DUNIX -g -DDEBUG main.cpp $(INC_DIRS:%=-I%)  $(LIBS)
main:main.o cl_spec.o 
	g++  main.o  cl_spec.o -o main $(INC_DIRS:%=-I%)  $(LIBS)

clean:
	rm -f *.o main 
