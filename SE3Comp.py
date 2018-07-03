import torch 
import torch.nn as nn

class SE3Comp(nn.Module):
    def __init__(self):
        super(SE3Comp, self).__init__()
        self.threshold_square = 1e-1
        self.threshold_cube = 1e-1
    
    def forward(self, Tg, xi):
        """
        Tg: <Torch.tensor> SE(3) R^7 (x, y, z, ww, wx, wy, wz)
            Tg = torch.zeros(batchSize, 7, 1)
        xi: <Torch.tensor> se(3) R^6 (rho1, rho2, rho3, omega_x, omega_y, omega_z)
            xi_vec = torch.zeros(batchSize, 6, 1)
        return Composed SE(3) in R^7 format
        """
        assert isinstance(Tg, type(torch.zeros(1))),'Tg with wrong datatype, should be torch.Tensor'
        assert isinstance(xi, type(torch.zeros(1))),'Tg with wrong datatype, should be torch.Tensor'

        
        rho   = xi[:, 0:3]
        omega = xi[:, 3:6] #torch.Size([batchSize, 3, 1])
        batchSize = xi.size()[0]
        R, V = self.so3_RV(torch.squeeze(omega, dim=2))
        Txi = torch.zeros(batchSize,4,4)
        Txi[:, 0:3, 0:3] = R
        Txi[:, 3,3] = 1.0
        Txi[:, 0:3, 3] = torch.squeeze(torch.bmm(V, rho))
        
        Tg_matrix = torch.zeros(batchSize,4,4)
        Tg_matrix[:, 3, 3] = 1.0
        q = Tg[:, 3:7]
        Tg_matrix[:, 0:3, 0:3] = self.q_to_Matrix(q)
        Tg_matrix[:, 0, 3] = torch.squeeze(Tg[:, 0])
        Tg_matrix[:, 1, 3] = torch.squeeze(Tg[:, 1])
        Tg_matrix[:, 2, 3] = torch.squeeze(Tg[:, 2])
        T_combine_M = torch.bmm(Txi, Tg_matrix)
        
        return self.batchMtoR7(T_combine_M)
    
    def batchMtoR7(self,M):
        batchSize = M.size()[0]
        cat = None
        for i in range(batchSize):
            a = self.MtoR7(M[i])
            if i == 0:
                cat = torch.unsqueeze(a, dim=0)
                continue
            cat = torch.cat([cat,torch.unsqueeze(a, dim=0)])
            
        return cat
    
    def MtoR7(self,M):#no batch
        R7 = torch.zeros(7,1)

        R7[0] = M[ 0, 3] # [2] to [2, 1]
        R7[1] = M[ 1, 3] # [2] to [2, 1]
        R7[2] = M[ 2, 3] # [2] to [2, 1]
        #https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        t = 0
        if M[2, 2] < 0:
            if M[0, 0] > M[1, 1]:#
                t = 1 + M[0, 0] - M[1, 1] - M[2, 2]
                q = [M[2, 1]-M[1, 2],  t,  M[0, 1]+M[1, 0],  M[2, 0]+M[0, 2]]
            else:#
                t = 1 - M[0, 0] + M[1, 1] - M[2, 2]
                q = [M[0, 2]-M[2, 0],  M[0, 1]+M[1, 0],  t,  M[1, 2]+M[2, 1]]
        else:
            if M[0, 0] < -M[1, 1]:#
                t = 1 - M[0, 0] - M[1, 1] + M[2, 2]
                q = [M[1, 0]-M[0, 1],  M[2, 0]+M[0, 2],  M[1, 2]+M[2, 1],  t]
            else:#
                t = 1 + M[0, 0] + M[1, 1] + M[2, 2]
                q = [t,  M[2, 1]-M[1, 2],  M[0, 2]-M[2, 0],  M[1, 0]-M[0, 1]]
        R7[3], R7[4], R7[5], R7[6] = q
        R7[3] *= 0.5 / torch.sqrt(t)
        R7[4] *= 0.5 / torch.sqrt(t)
        R7[5] *= 0.5 / torch.sqrt(t)
        R7[6] *= 0.5 / torch.sqrt(t)
        if R7[3] < 0:
            R7[3] *= -1
            R7[4] *= -1
            R7[5] *= -1
            R7[6] *= -1
        return R7
        
    def q_to_Matrix(self, q):
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
    
    def so3_RV(self, omega):
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
        sin_theta_div_theta[sin_theta_div_theta != sin_theta_div_theta] = 0 # set nan to zero

        one_minus_cos_theta = torch.ones(theta.size()) - torch.cos(theta)
        one_minus_cos_div_theta_sqr = torch.div(one_minus_cos_theta,theta_sqr)

        theta_minus_sin_theta = theta - torch.sin(theta)
        theta_minus_sin_div_theta_cube = torch.div(theta_minus_sin_theta, theta_cube)

        sin_theta_div_theta_tensor            = torch.ones(omega_skew.size())
        one_minus_cos_div_theta_sqr_tensor    = torch.ones(omega_skew.size())
        theta_minus_sin_div_theta_cube_tensor = torch.ones(omega_skew.size())
        
        # sin_theta_div_theta do not need linear approximation
        sin_theta_div_theta_tensor = sin_theta_div_theta
        
        for b in range(batchSize):
            if theta_sqr[b] > self.threshold_square:
                one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr[b]
            elif theta_sqr[b] < 1e-6:
                one_minus_cos_div_theta_sqr_tensor[b] = 0#0.5
            else:#Taylor expansion
                c = 1.0 / 2.0
                c += theta[b]**(4*1) / 720.0#np.math.factorial(6) 
                c += theta[b]**(4*2) / 3628800.0#np.math.factorial(6+4) 
                c -= theta[b]**(2) / 24.0#np.math.factorial(4) 
                c -= theta[b]**(2 + 4) / 40320.0#np.math.factorial(4+4) 
                one_minus_cos_div_theta_sqr_tensor[b] = c
                
            if theta_cube[b] > self.threshold_cube:
                theta_minus_sin_div_theta_cube_tensor[b] = theta_minus_sin_div_theta_cube[b]
            elif theta_sqr[b] < 1e-6:
                theta_minus_sin_div_theta_cube_tensor[b] = 0#1.0 / 6.0
            else:#Taylor expansion
                s = 1.0 / 6.0
                s += theta[b]**(4*1) / 5040.0
                s += theta[b]**(4*2) / 39916800.0
                s -= theta[b]**(2) / 120.0
                s -= theta[b]**(2 + 4) / 362880.0
                theta_minus_sin_div_theta_cube_tensor[b] = s

        completeTransformation = torch.zeros(batchSize,3,3)

        completeTransformation[:, 0, 0] += 1
        completeTransformation[:, 1, 1] += 1
        completeTransformation[:, 2, 2] += 1   

        sin_theta_div_theta_tensor = torch.unsqueeze(sin_theta_div_theta_tensor, dim=1)
        completeTransformation = completeTransformation +\
            self.vecMulMat(sin_theta_div_theta_tensor,omega_skew) +\
            torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)


        V = torch.zeros(batchSize,3,3)    
        V[:, 0, 0] += 1
        V[:, 1, 1] += 1
        V[:, 2, 2] += 1 
        V = V + torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew) +\
            torch.mul(theta_minus_sin_div_theta_cube_tensor, omega_skew_sqr)
        return completeTransformation, V
    
    def vecMulMat(self, vec, mat):
        mat_view = mat.view(vec.size()[0], -1)
        out = mat_view * vec
        return out.view(mat_view.size()[0], mat.size()[1], -1)
    

            
            
            