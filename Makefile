sequential:
	gcc cosine.c -lm -O3 -o cosine-seq.exe
cuda:
	nvcc -arch=sm_13 cosine.cu -O3 -o cosine-gpu.exe
clean:
	rm cosine-seq.exe cosine-gpu.exe
