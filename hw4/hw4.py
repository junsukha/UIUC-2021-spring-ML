import torch
import hw4_utils
import matplotlib.pyplot as plt

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid. #two centroids
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    
    # r_matrix = torch.argmin()
    if X is None:
        X, init_c = hw4_utils.load_data()
    print(init_c)
    d1, N = X.shape
    d2, K = init_c.shape
 

    
    
    
#     new_c = torch.empty(2,2)
#     old_c = init_c   
    new_c = init_c
    print(new_c)
#     while True:
    for num in range(n_iters):

        r_matrix = torch.zeros(N,K)
#         print(r_matrix)
        for i in range (N):#data
            distance = 999999
            

            #assign r matrix
            
            for k in range(K):#centroid
                new_distance = torch.dist(X[:,i], new_c[:,k])
#                 print(new_distance)
                if distance > new_distance:
                    distance = new_distance
                    index = k
            r_matrix[i,index] = 1
            
#         print(r_matrix)

        #assgin new centroids
        new_c =   X @ r_matrix
        for k in range (K):
            print(sum(r_matrix[:,k]))
            new_c[:,k] /= sum(r_matrix[:,k])

#         old_c = new_c.clone()

    init_c[:,0] = new_c[:,1]
    init_c[:,1] = new_c[:,0]
#     print(new_c.shape)
#     hw4_utils.vis_cluster(new_c[:,0], X[0,:], new_c[:,1], X[1,:])
    
    plt.plot(X[0,:], X[1,:], 'ro')
    plt.plot(init_c[0,:], init_c[1,:] ,'bo')
    print(init_c)
    return init_c
        

   
