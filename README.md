# local-embedding
1. (Done)实现spherical meanshift

2. (Done. Looks good)看一下spherical meanshift 在synthetic data上的效果。试着调一下bandwidth 参数，看看在synthetic data 上参数对结果的变化。

3. 如果在high dimension效果不好的话，可以试着用PCA 等降维algorithm 在去做下meanshift。 如果能降到2-3维，也可以在paper上体现效果图。It seems that Mean shift might not work well in higher dimensions. In higher dimensions , the number of local maxima is pretty high and it might converge to a local optima soon. 

4. Bandwidth v.s. Number of cluster (dblp with 13855 points, dimension:100): 0.7—>9780, 0.8—>5736, 0.9—>2218, 1.0—>286, 1.03–>19, 1.05–>2, 1.1—>1. 1.04 seems to be a good choice for dblp level 0. 

