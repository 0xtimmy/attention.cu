
FUNC=

testall:
	nvcc -o testall.exe ./testall.cu
	./testall.exe

clean:
	rm ./testall.exe

test:
	nvcc -o ./tests/$(FUNC)/run.exe ./tests/$(FUNC)/run.cu
	python ./tests/$(FUNC)/test.py
	./tests/$(FUNC)/run.exe

testv:
	nvcc -o ./tests/$(FUNC)/run.exe ./tests/$(FUNC)/run.cu
	python ./tests/$(FUNC)/test.py
	./tests/$(FUNC)/run.exe -v

