import torch.nn as nn

class SE3Comp(nn.Module):
    def __init__(self):
        super(SE3Comp, self).__init__()
        pass
    
    def forward(self, Tg, xi):
        """
        Tg: <Torch.tensor> SE(3) R^7 (x, y, z, ww, wx, wy, wz)
        xi: <Torch.tensor> se(3) R^6 (rho1, rho2, rho3, omega_x, omega_y, omega_z)
        
        return Composed SE(3) in R^7 format
        """
        rho   = xi[:, 0:3]
        omega = xi[:, 3:6] #torch.Size([batchSize, 3, 1])
        
        R, V = so3_RV(torch.squeeze(omega))
        Txi = torch.zeros(batchSize,4,4)
        Txi[:, 0:3, 0:3] = R
        Txi[:, 3,3] = 1.0
        Txi[:, 0:3, 3] = torch.squeeze(torch.bmm(V, rho))
        
        Tg_matrix = torch.zeros(batchSize,4,4)
        q = Tg[:, 3:7]
        Tg_matrix[:, 0:3, 0:3] = q_to_Matrix(q)
        Tg_matrix[:, 0, 3] = Tg[:, 0]
        Tg_matrix[:, 1, 3] = Tg[:, 1]
        Tg_matrix[:, 2, 3] = Tg[:, 2]
        
        T_combine = torch.bmm(Txi, Tg_matrix)
        return 
    
    def q_to_Matrix(q):
        qw = q[:, 0]
        qx = q[:, 1]
        qy = q[:, 2]
        qz = q[:, 3]
        M = torch.zeros(q.size()[0], 3, 3)

        M[:, 0, 0] = torch.squeeze( 1 - 2*torch.mul(qy,qy) - 2*torch.mul(qz,qz) )
        M[:, 1, 0] = torch.squeeze( 2*torch.mul(qx,qy) + 2*torch.mul(qz,qw) )
        M[:, 2, 0] = torch.squeeze( 2*torch.mul(qx,qz) - 2*torch.mul(qy,qw) )

        M[:, 0, 1] = torch.squeeze( 2*torch.mul(qx,qy) - 2*torch.mul(qz,qw) )
        M[:, 1, 1] = torch.squeeze( 1 - 2*torch.mul(qx,qx) - 2*torch.mul(qz,qz) )
        M[:, 2, 1] = torch.squeeze( 2*torch.mul(qy,qz) + 2*torch.mul(qx,qw) )

        M[:, 0, 2] = torch.squeeze( 2*torch.mul(qx,qz) + 2*torch.mul(qy,qw) )
        M[:, 1, 2] = torch.squeeze( 2*torch.mul(qy,qz) - 2*torch.mul(qx,qw) )
        M[:, 2, 2] = torch.squeeze( 1 - 2*torch.mul(qx,qx) - 2*torch.mul(qy,qy) )
    
        return M
    
    def so3_RV(omega):
        """
        (3-tuple)
        omega = torch.zeros(batchSize, 3)

        return batchx3x3 matrix R after exponential mapping, V
        """
        batchSize = omega.size()[0]
        omega_x = omega[:, 0]
        omega_y = omega[:, 1]
        omega_z = omega[:, 2]

        #paramIndex = paramIndex + 3
        omega_skew = torch.zeros(batchSize,3,3)
        """
        0    -oz  oy  0
        oz   0   -ox  0
        -oy  ox   0   0
        0    0    0   0
        """
        omega_skew[:, 1, 0] = omega_z.clone()
        omega_skew[:, 2, 0] = -1 * omega_y

        omega_skew[:, 0, 1] = -1 * omega_z
        omega_skew[:, 2, 1] = omega_x.clone()

        omega_skew[:, 0, 2] = omega_y.clone()
        omega_skew[:, 1, 2] = -1 * omega_x

        omega_skew_sqr = torch.bmm(omega_skew,omega_skew)
        theta_sqr = torch.pow(omega_x,2) +\
                    torch.pow(omega_y,2) +\
                    torch.pow(omega_z,2)
        theta = torch.pow(theta_sqr,0.5)
        theta_cube = torch.mul(theta_sqr, theta)#
        sin_theta = torch.sin(theta)
        sin_theta_div_theta = torch.div(sin_theta,theta)

        one_minus_cos_theta = torch.ones(theta.size()) - torch.cos(theta)
        one_minus_cos_div_theta_sqr = torch.div(one_minus_cos_theta,theta_sqr)

        theta_minus_sin_theta = theta - torch.sin(theta)
        theta_minus_sin_div_theta_cube = torch.div(theta_minus_sin_theta, theta_cube)

        sin_theta_div_theta_tensor            = torch.ones(omega_skew.size())
        one_minus_cos_div_theta_sqr_tensor    = torch.ones(omega_skew.size())
        theta_minus_sin_div_theta_cube_tensor = torch.ones(omega_skew.size())
        for b in range(batchSize):
            if theta_sqr[b] > 1e-12:#self.threshold:
                sin_theta_div_theta_tensor[b] = sin_theta_div_theta[b]
                one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr[b]
                theta_minus_sin_div_theta_cube_tensor[b] = theta_minus_sin_div_theta_cube[b]
            else:
                sin_theta_div_theta_tensor[b]  = 1
                one_minus_cos_div_theta_sqr_tensor[b] = 0
                theta_minus_sin_div_theta_cube_tensor[b] = 1.0 / 6.0

        completeTransformation = torch.zeros(batchSize,3,3)

        completeTransformation[:, 0, 0] += 1
        completeTransformation[:, 1, 1] += 1
        completeTransformation[:, 2, 2] += 1   
        completeTransformation = completeTransformation +\
            torch.mul(sin_theta_div_theta_tensor,omega_skew) +\
            torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)


        V = torch.zeros(batchSize,3,3)    
        V[:, 0, 0] += 1
        V[:, 1, 1] += 1
        V[:, 2, 2] += 1 
        V = V + torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew) +\
            torch.mul(theta_minus_sin_div_theta_cube_tensor, omega_skew_sqr)
        return completeTransformation, V
    
    def expo(xi):
        """ exponential map """
        upsilon = v[0:3, :]
        omega = sophus.Vector3(v[3], v[4], v[5])
        so3 = sophus.So3.exp(omega)
        Omega = sophus.So3.hat(omega)
        Omega_sq = Omega * Omega
        theta = sympy.sqrt(sophus.squared_norm(omega))
        V = (sympy.Matrix.eye(3) +
             (1 - sympy.cos(theta)) / (theta**2) * Omega +
             (theta - sympy.sin(theta)) / (theta**3) * Omega_sq)
        return Se3(so3, V * upsilon)
    
    def hat(v):
        upsilon = sophus.Vector3(v[0], v[1], v[2])
        omega = sophus.Vector3(v[3], v[4], v[5])
        return so3hat(omega).\
            row_join(upsilon).\
            col_join(sympy.Matrix.zeros(1, 4))
            
    def so3hat(o):
        return sympy.Matrix([[0, -o[2], o[1]],
                             [o[2], 0, -o[0]],
                             [-o[1], o[0], 0]])            
            
            
            
            