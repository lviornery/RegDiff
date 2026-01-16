from regdiff import regdiff
import numpy as np
import numpy.linalg as npl
import scipy as sp

def test_matrix_diff():
    diff,diff2,d = regdiff.matrix_diff(5)
    assert np.allclose(diff,np.asarray([
        [-1.5,2,-0.5,0,0],
        [-0.5,0,0.5,0,0],
        [0,-0.5,0,0.5,0],
        [0,0,-0.5,0,0.5],
        [0,0,0.5,-2,1.5]]))
    assert np.allclose(diff2,np.asarray([
        [2, -5, 4, -1,0],
        [1,-2,1,0,0],
        [0,1,-2,1,0],
        [0,0,1,-2,1],
        [0,-1,4,-5,2]]))
    assert np.allclose(d,np.asarray([
        [-1,1,0,0,0],
        [0,-1,1,0,0],
        [0,0,-1,1,0],
        [0,0,0,-1,1]]))
    diff,diff2,d = regdiff.matrix_diff(5,0.1)
    assert np.allclose(diff,np.asarray([
        [-15,20,-5,0,0],
        [-5,0,5,0,0],
        [0,-5,0,5,0],
        [0,0,-5,0,5],
        [0,0,5,-20,15]]))
    assert np.allclose(diff2,np.asarray([
        [200, -500, 400, -100,0],
        [100,-200,100,0,0],
        [0,100,-200,100,0],
        [0,0,100,-200,100],
        [0,-100,400,-500,200]]))
    assert np.allclose(d,np.asarray([
        [-10,10,0,0,0],
        [0,-10,10,0,0],
        [0,0,-10,10,0],
        [0,0,0,-10,10]]))

def test_matrix_anti_diff():
    a,a2 = regdiff.matrix_anti_diff(5)
    assert np.allclose(a,np.asarray([
        [0.5,0.5,0,0,0],
        [0.5,1,0.5,0,0],
        [0.5,1,1,0.5,0],
        [0.5,1,1,1,0.5]]))
    assert np.allclose(a2,np.asarray([
        [0.25,0.25,0,0,0],
        [0.75,1,0.25,0,0],
        [1.25,2,1,0.25,0],
        [1.75,3,2,1,0.25]]))
    a,a2 = regdiff.matrix_anti_diff(5,0.1)
    assert np.allclose(a,np.asarray([
        [0.05,0.05,0,0,0],
        [0.05,0.1,0.05,0,0],
        [0.05,0.1,0.1,0.05,0],
        [0.05,0.1,0.1,0.1,0.05]]))
    assert np.allclose(a2,np.asarray([
        [0.0025,0.0025,0,0,0],
        [0.0075,0.01,0.0025,0,0],
        [0.0125,0.02,0.01,0.0025,0],
        [0.0175,0.03,0.02,0.01,0.0025]]))

def test_reg_diff():
    test_x = np.asarray([0,1,4,9,16,25,36,49,64,81,100])
    test_dx = np.asarray([0.9455,1.2192,4.5008,5.7021,8.1431,9.9856,11.8851,14.2714,15.5243,18.7559,18.9798])
    test_ddx = np.asarray([1.0259,1.0281,2.2012,2.2025,2.2006,2.0091,1.9684,1.9652,1.9640,1.9636,1.9635])
    u0 = np.zeros_like(test_x)
    u0 = np.stack((u0,u0))

    dx,ddx = regdiff.reg_diff(test_x,0.001,0.01,u0=u0)
    assert np.allclose(dx,test_dx,atol=1e-4)
    assert np.allclose(ddx,test_ddx,atol=1e-4)

    dx,ddx = regdiff.reg_diff(test_x,0.001,0.01,dx=0.1,u0=u0)
    assert np.allclose(dx,10*test_dx,atol=1e-3)
    assert np.allclose(ddx,100*test_ddx,atol=1e-2)