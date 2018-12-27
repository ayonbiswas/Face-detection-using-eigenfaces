import os
from PIL import Image
import numpy as np      #version 1.12.1
import cv2              #openCV version 3.3.0
import sys
"""
read_img() constructs the training image list and test image list by taking 70% for training and 30% for testing
it also assigns labels to the images for testing

output: training and test image list , labels
"""
def read_img(path):
    test_id = [ ]
    train_id = []
    test = []
    subject_count = 1
    train = [ ]
    ct = 1
    for dire , sub_dir , files in os.walk(path):
        for sbd in sub_dir:
            current_folder = os.path.join(dire , sbd)

            ct = 1
            for img_files in os.listdir(current_folder):

                if(ct<8):
                    img = Image.open(os.path.join(current_folder , img_files))

                    train.append(np.asarray(img , dtype=np.uint8))
                    train_id.append(subject_count)
                else:
                    img = Image.open(os.path.join(current_folder , img_files))

                    test.append(np.asarray(img , dtype=np.uint8))
                    test_id.append(subject_count)
                ct = ct+1


            subject_count = subject_count + 1

    return train ,test, train_id, test_id
"""
create_img_vector() generates image matrix given a list of images

output: vector of images
"""

def create_img_vector(x):
    if (len(x) == 0):
        return np.asarray([ ])
    else:
        temp = np.empty((x[ 0 ].size , 0) , dtype=x[ 0 ].dtype)
        for col in x:
            temp = np.hstack((temp , np.asarray(col).reshape(-1 , 1)))

    return temp

"""
principal_components() takes input as image matrix , the required number of principal components

outputs: eigenvalues, eigenvectors, mean of images
"""

def principal_components(x , num_comp):

    img_dim , n  = x.shape
    mu = x.mean(axis=1)
    temp = np.empty((img_dim , 0) , dtype=x[ 0 ].dtype)
    for k in xrange(n):
        temp = np.hstack((temp , np.asarray(mu).reshape(-1 , 1)))

    mu = temp
    if (img_dim < n):
        cov = np.dot(x , x.T)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(cov)

    else:
        cov = np.dot(x.T , x)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(cov)
        eigenvectors = np.dot(x , eigenvectors)
        for i in range(n):
            eigenvectors[ : , i ] = eigenvectors[ : , i ] / np.linalg.norm(eigenvectors[ : , i ])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[ idx ]

    eigenvectors = eigenvectors[ : , idx ]
    if ((num_comp > n) or (num_comp < 0)):
        num_comp = n
    e_val = eigenvalues[ 0:num_comp ]

    e_vector = eigenvectors[ : , 0:num_comp ]

    return [ e_val , e_vector , mu ]
"""
projection() takes the projection of mean shifted data matrix x-mu along w
"""

def projection(w , x , mu):
    return np.dot(w.T , x - mu)
"""
distance() computes the euclidean norm between two set of points
"""

def distance(x , y):

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    return np.sqrt(np.sum(np.power((x - y) , 2)))
"""
predict() takes in the test image vector, principal component matrix, mean, projected training images and the image labels
assigns label to the test image corresponding to the face with least distance

output: label and minimum distance
"""

def predict(test , w , mu , p_x , sub_id):
    min_id = 1


    p_test = projection(w , test , mu)

    min_dist = distance(p_x[ : , 0 ] , p_test)

    for i in xrange(1 , p_x.shape[ 1 ]):

        dist = distance(p_x[ : , i ] , p_test)
        if (dist < min_dist):

            min_dist = dist
            min_id = sub_id[ i ]

    return min_id , min_dist

"""
main() implements all the subroutines to carry out facial recognition and compute the accuracy
"""
def main():
    train ,test, train_id, test_id = read_img("D:\\att")
    s = create_img_vector(train)
    id_list = []


    [ p_val , p_vec , mu ] = principal_components(s , 100)
    r = create_img_vector(test)


    ct =0

    train_projected = projection(p_vec,s,mu)

    for i in xrange(r.shape[ 1 ]):

        test_vec = r[:,i].reshape(-1,1)

        [ id , dist ] = predict(test_vec , p_vec , np.asarray(mu[ : , 0 ]).reshape(-1 , 1) , train_projected, train_id)

        id_list.append(id)
    for i in xrange(len(id_list)):
        if (id_list[i]==test_id[i]):

            ct =ct +1


    acc = float(ct)/len(id_list)*100.0
    print acc



main()






