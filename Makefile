sequential:
	gcc cosine.c -lm -o cosine-seq.exe
cuda:
	nvcc -arch=sm_13 cosine.cu -o cosine-gpu.exe
clean:
	rm cosine-seq.exe cosine-gpu.exe
