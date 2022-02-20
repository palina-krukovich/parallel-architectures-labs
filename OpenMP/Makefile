compile_seq:
	@g++ -O2 a2-sequential.cpp

compile_par_taskloop:
	@g++ -fopenmp -O2 taskloop.cpp
	
compile_par_task:
	@g++ -fopenmp -O2 task.cpp

run:
	@rm -f out_m.txt
	@rm -f out_c.txt
	@touch out_m.txt
	@touch out_c.txt
	@./a.out
	@rm out_m.txt
	@rm out_c.txt

test_run:
	@rm -f out_m.txt
	@rm -f out_c.txt
	@touch out_m.txt
	@touch out_c.txt
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@./a.out
	@tr \  \\n<out_m.txt|jq -s add/length|sed s/^/Mandelbrot:\ /
	@tr \  \\n<out_c.txt|jq -s add/length|sed s/^/Convolution:\ /
	@rm out_m.txt
	@rm out_c.txt

cmp_taskloop:
	@cmp mandelbrot.ppm mandelbrot-taskloop.ppm

cmp_task:
	@cmp mandelbrot.ppm mandelbrot-task.ppm
