418RAD: 418rad.o bsp.o cudabsp.o cudarad.o raytracer.o
	nvcc -m64 418rad.o bsp.o cudabsp.o cudarad.o raytracer.o -std=c++11 -o rad -g -lcudart -lcudadevrt

bsp.o: bsp.cpp bsp.h
	nvcc -m64 bsp.cpp -O3 -I ./include -std=c++11 -c -g -arch=compute_35 -rdc=true

418rad.o: 418rad.cpp bsp.h
	nvcc -m64 418rad.cpp -O3 -I ./include -std=c++11 -c -g -arch=compute_35 -rdc=true

cudabsp.o: cudabsp.cu cudabsp.h
	nvcc -m64 cudabsp.cu -O3 -I ./include -std=c++11 -c -g -arch=compute_35 -rdc=true

cudarad.o: cudarad.cu cudarad.h
	nvcc -m64 cudarad.cu -O3 -I ./include -std=c++11 -c -g -arch=compute_35 -rdc=true

raytracer.o: raytracer.cu raytracer.h
	nvcc -m64 raytracer.cu -O3 -I ./include -std=c++11 -c -g -arch=compute_35 -rdc=true

comparelumps.o: comparelumps.cpp bsp.h
	nvcc comparelumps.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

comparelumps: comparelumps.o bsp.o
	nvcc comparelumps.o bsp.o -Wall -Wextra -Wfatal-errors -std=c++11 -o comparelumps -g

examinebsp.o: examinebsp.cpp bsp.h
	nvcc examinebsp.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

examinebsp: examinebsp.o bsp.o
	nvcc examinebsp.o bsp.o -Wall -Wextra -Wfatal-errors -std=c++11 -o examinebsp -g

clean:
	-rm -f *.o
	-rm -f *.exe
	-rm -f 418RAD
	-rm -f rad
	-rm -f *.out
	-rm -f comparelumps
	-rm -f examinebsp
