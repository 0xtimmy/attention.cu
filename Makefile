

testall:
	nvcc -o testall.exe ./testall.cu
	./testall.exe

clean:
	rm ./testall.exe