import numpy as np
from random import shuffle

def compute_line(points):
    # [[x1 y1]  [a  = [1
    #  [x2 y2]]  b] =  1]
    # x = np.array(points)[:,0:-1]
    # y = np.array(points)[:,-1:]
    # ones = np.ones((x.shape[0],1))
    # a = np.concatenate((x,ones),axis=1)
    # b = y
    a = np.array(points)
    b = np.ones((a.shape[0],1))
    return np.linalg.lstsq(a, b)[0].reshape((a.shape[1]))


def get_inliers(params, points, cuttoff):
    final_matches = []
    for match in matches:
        A = get_A([match], features1, features2)
        cost = np.linalg.norm(np.dot(A,model))
        # print("cost: "+str(cost))
        if cost < cuttoff:
            final_matches.append(match)

    # return final_matches
    return final_matches

def ransac(points, sample_num, num_iter, cuttoff):
    best_inliers = []
    for i in range(num_iter):
        shuffle(points)
        sample = points[:sample_num]
        inliers = get_inliers(sample, points, cuttoff)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
        # print("iteration "+str(i)+" / "+str(num_iter))

    return params


if __name__ == '__main__':
    points = [(0,1)]
    params = compute_line(points)
    print(params)
    print("\n")
    points = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)]
    params = compute_line(points)
    print(params)
    print("\n")
    points = [(0,1,3),(1,2,3),(2,3,3),(3,4,3)]
    params = compute_line(points)
    print(params)
    print("\n")
    points = [(2,0),(2,1),(2,2)]
    params = compute_line(points)
    print(params)