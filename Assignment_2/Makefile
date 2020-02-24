CXX = icc
CXXFLAGS = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dot_product

all : $(TARGET1)
    
$(TARGET1) : dp_openmp.c kernel_openmp.c dp_openmp.h
	$(CXX) $(CXXFLAGS) -o $(TARGET1) dp_openmp.c kernel_openmp.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
