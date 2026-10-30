import torch

import torchregdiff

torch.set_default_dtype(torch.float64)

def test_matrix_diff():
    diff,diff2,d = torchregdiff.matrix_diff(5)
    assert torch.allclose(diff,torch.tensor([
        [-1.5,2,-0.5,0,0],
        [-0.5,0,0.5,0,0],
        [0,-0.5,0,0.5,0],
        [0,0,-0.5,0,0.5],
        [0,0,0.5,-2,1.5]]).to(torch.get_default_dtype()))
    assert torch.allclose(diff2,torch.tensor([
        [2,-5,4,-1,0],
        [1,-2,1,0,0],
        [0,1,-2,1,0],
        [0,0,1,-2,1],
        [0,-1,4,-5,2]]).to(torch.get_default_dtype()))
    assert torch.allclose(d,torch.tensor([
        [-1,1,0,0,0],
        [0,-1,1,0,0],
        [0,0,-1,1,0],
        [0,0,0,-1,1]]).to(torch.get_default_dtype()))
    diff,diff2,d = torchregdiff.matrix_diff(5,0.1)
    assert torch.allclose(diff,torch.tensor([
        [-15,20,-5,0,0],
        [-5,0,5,0,0],
        [0,-5,0,5,0],
        [0,0,-5,0,5],
        [0,0,5,-20,15]]).to(torch.get_default_dtype()))
    assert torch.allclose(diff2,torch.tensor([
        [200, -500, 400, -100,0],
        [100,-200,100,0,0],
        [0,100,-200,100,0],
        [0,0,100,-200,100],
        [0,-100,400,-500,200]]).to(torch.get_default_dtype()))
    assert torch.allclose(d,torch.tensor([
        [-10,10,0,0,0],
        [0,-10,10,0,0],
        [0,0,-10,10,0],
        [0,0,0,-10,10]]).to(torch.get_default_dtype()))

def test_matrix_anti_diff():
    a,a2 = torchregdiff.matrix_anti_diff(5)
    assert torch.allclose(a,torch.tensor([
        [0.5,0.5,0,0,0],
        [0.5,1,0.5,0,0],
        [0.5,1,1,0.5,0],
        [0.5,1,1,1,0.5]]).to(torch.get_default_dtype()))
    assert torch.allclose(a2,torch.tensor([
        [0.25,0.25,0,0,0],
        [0.75,1,0.25,0,0],
        [1.25,2,1,0.25,0],
        [1.75,3,2,1,0.25]]).to(torch.get_default_dtype()))
    a,a2 = torchregdiff.matrix_anti_diff(5,0.1)
    assert torch.allclose(a,torch.tensor([
        [0.05,0.05,0,0,0],
        [0.05,0.1,0.05,0,0],
        [0.05,0.1,0.1,0.05,0],
        [0.05,0.1,0.1,0.1,0.05]]).to(torch.get_default_dtype()))
    assert torch.allclose(a2,torch.tensor([
        [0.0025,0.0025,0,0,0],
        [0.0075,0.01,0.0025,0,0],
        [0.0125,0.02,0.01,0.0025,0],
        [0.0175,0.03,0.02,0.01,0.0025]]).to(torch.get_default_dtype()))

def test_reg_diff():
    test_x = torch.tensor([0,1,4,9,16,25,36,49,64,81,100]).to(torch.get_default_dtype())
    test_dx = torch.tensor([0.9455,1.2192,4.5008,5.7021,8.1431,9.9856,11.8851,14.2714,15.5243,18.7559,18.9798]).to(torch.get_default_dtype())
    test_ddx = torch.tensor([1.0259,1.0281,2.2012,2.2025,2.2006,2.0091,1.9684,1.9652,1.9640,1.9636,1.9635]).to(torch.get_default_dtype())
    u0 = torch.zeros_like(test_x).to(torch.get_default_dtype())
    u0 = torch.stack((u0,u0))

    dx,ddx = torchregdiff.reg_diff(test_x,0.001,0.01,u0=u0,diag=True)
    assert torch.allclose(dx,test_dx,atol=1e-4)
    assert torch.allclose(ddx,test_ddx,atol=1e-4)

    dx,ddx = torchregdiff.reg_diff(test_x,0.001,0.01,dx=0.1,u0=u0)
    assert torch.allclose(dx,10*test_dx,atol=1e-3)
    assert torch.allclose(ddx,100*test_ddx,atol=1e-2)