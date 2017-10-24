# local-embedding
1. (Done)Adaptive MeanShift Code: Generate some synthetic data and see what the code outputs. Don’t have to read the code line by line. 

2. (Done)Save the data file and use Adaptive Meanshift Code to run on the written file. 

1. (Done)实现spherical meanshift

2. (Done. Looks good)看一下spherical meanshift 在synthetic data上的效果。试着调一下bandwidth 参数，看看在synthetic data 上参数对结果的变化。

3. 如果在high dimension效果不好的话，可以试着用PCA 等降维algorithm 在去做下meanshift。 如果能降到2-3维，也可以在paper上体现效果图。It seems that Mean shift might not work well in higher dimensions. In higher dimensions , the number of local maxima is pretty high and it might converge to a local optima soon. 

4. Parameter Tuning:
	1). Bandwidth v.s. Number of cluster (dblp with 13855 points): It is very sensitive to the bandwidth. It seems that with lower dimensions, number of clusters generated are less sensitive to bandwidth. 
	dimension=100: 0.7—>9780, 0.8—>5736, 0.9—>2218, 1.0—>286, 1.03–>19, 1.05–>2, 1.1—>1. 1.04 seems to be a good choice for dblp level 0. bandwidth = estimate_bandwidth(data, quantile=0.004)
	dimension=20: 0.8—>23, 0.83—>8, 0.84—>7, 0.85—>4, 0.9—>2. 

	2) Choice of bandwidth parameter h is critical. A large h might result in incorrect clustering and might merge distinct clusters. A very small h might result in too many clusters. When using kNN to determining h, the choice of k influences the value of h. For good results, k has to increase when the dimension of the data increases.

5. Dimensionality Reduction:
	1) (Done) Lower the dimension of word embeddings. Do check if the word embeddings represent the text well. The results is bad when dimension is 5. The results is kind of acceptable when dimension is 10. It's better when dimension is 20. The semantic meaning is well captured when dimension is 30. Considering both semantic meaning and dimensionality issue, we should use dimension = 20-30. 
	2) Try some adaptive MeanShift method suitable for high-dimensional data (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.9132&rep=rep1&type=pdf) (https://ac.els-cdn.com/S0030402612002902/1-s2.0-S0030402612002902-main.pdf?_tid=2b4ed052-a93a-11e7-8289-00000aacb360&acdnat=1507145476_bc874212de9dec7188ececb022872ac8)
	3) Try PCA or other algorithms to reduce dimension (https://link.springer.com/content/pdf/10.1007%2F978-3-642-42054-2_77.pdf)

6. Keyword Output: 
	1) We may consider the frequency of the keyword occurrence and output only the frequent words. We can make a function = frequency * keyword_representativeness. 
