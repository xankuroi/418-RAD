418RAD: 418rad.o bsp.o
	g++ 418rad.o bsp.o -Wall -Wextra -Wfatal-errors -std=c++11 -o 418RAD -g

bsp.o: bsp.cpp bsp.h
	g++ bsp.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

418rad.o: 418rad.cpp bsp.h
	g++ 418rad.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

comparelumps.o: comparelumps.cpp bsp.h
	g++ comparelumps.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

comparelumps: comparelumps.o bsp.o
	g++ comparelumps.o bsp.o -Wall -Wextra -Wfatal-errors -std=c++11 -o comparelumps -g

examinebsp.o: examinebsp.cpp bsp.h
	g++ examinebsp.cpp -I ./include -Wall -Wextra -Wfatal-errors -std=c++11 -c -g

examinebsp: examinebsp.o bsp.o
	g++ examinebsp.o bsp.o -Wall -Wextra -Wfatal-errors -std=c++11 -o examinebsp -g

clean:
	-rm -f *.o
	-rm -f *.exe
	-rm -f 418RAD
	-rm -f comparelumps
	-rm -f examinebsp
