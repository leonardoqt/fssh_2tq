ROOT_DIR=$(shell pwd)
ODIR  = $(ROOT_DIR)/obj
SDIR  = $(ROOT_DIR)/src
ODIR2 = $(ROOT_DIR)/obj2
SDIR2 = $(ROOT_DIR)/src2

CXX   = mpicxx
CFLAG = -std=c++11 -larmadillo -lopenblas -lmpi
CXX2  = g++
CFLAG2= -std=c++11 -larmadillo -lopenblas
 
DEPS  = $(shell ls $(SDIR)/*.h)
SRC   = $(shell ls $(SDIR)/*.cpp)
OBJ   = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(SRC))
DEPS2 = $(shell ls $(SDIR2)/*.h)
SRC2  = $(shell ls $(SDIR2)/*.cpp)
OBJ2  = $(patsubst $(SDIR2)/%.cpp,$(ODIR2)/%.o,$(SRC2))

fssh.x : $(OBJ)
	$(CXX) -o $@ $^ $(CFLAG)

$(ODIR)/%.o : $(SDIR)/%.cpp $(DEPS) | $(ODIR)/.
	$(CXX) -c -o $@ $< $(CFLAG)

fssh2.x : $(OBJ2)
	$(CXX2) -o $@ $^ $(CFLAG2)

$(ODIR2)/%.o : $(SDIR2)/%.cpp $(DEPS2) | $(ODIR2)/.
	$(CXX2) -c -o $@ $< $(CFLAG2)

%/. : 
	mkdir -p $(patsubst %/.,%,$@)
	
.PRECIOUS: %/.
.PHONY: clean clean_dat clean_all
clean:
	rm -rf *.x $(ODIR) $(ODIR2)
clean_dat:
	rm -rf *.dat *.out
clean_all:
	rm -rf *.x *.dat *.out $(ODIR) $(ODIR2)
