418RAD: 418rad.o bsp.o
	g++ 418rad.o bsp.o -Wfatal-errors -std=c++11 -o 418RAD -g

bsp.o: bsp.cpp bsp.h
	g++ bsp.cpp -Wfatal-errors -std=c++11 -c -g

418rad.o: 418rad.cpp bsp.h
	g++ 418rad.cpp -Wfatal-errors -std=c++11 -c -g

comparelumps.o: comparelumps.cpp bsp.h
	g++ comparelumps.cpp -Wfatal-errors -std=c++11 -c -g

comparelumps: comparelumps.o bsp.o
	g++ comparelumps.o bsp.o -Wfatal-errors -std=c++11 -o comparelumps -g

examinebsp.o: examinebsp.cpp bsp.h
	g++ examinebsp.cpp -Wfatal-errors -std=c++11 -c -g

examinebsp: examinebsp.o bsp.o
	g++ examinebsp.o bsp.o -Wfatal-errors -std=c++11 -o examinebsp -g

clean:
	rm *.o
	rm *.exe
