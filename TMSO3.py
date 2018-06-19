
import torch.nn as nn

class TransformationMatrix3x4SO3(nn.Module):
    def __init__(self, useRotation=False, useScale=False, useTranslation=False):
        #super(TransformationMatrix3x4SO3, self).__init__()
        
        self.fullMode = not(useRotation or useScale or useTranslation)

        if not self.fullMode:
            self.useRotation = useRotation
            self.useScale = useScale
            self.useTranslation = useTranslation
            
        self.threshold = 1e-12
        
#     def forward(self, x):   
#         return 0

    def check(self, x):
        """
        x is a numpy array with shape (7,)
        """
        if self.fullMode:
            assert x.shape[0]==7, 'Expected 7 parameters, got ' + str(x.shape[0]) 
        else:
            numberParameters = 0
            if self.useRotation:
                numberParameters = numberParameters + 3            
            if self.useScale:
                numberParameters = numberParameters + 1
            if self.useTranslation:
                numberParameters = numberParameters + 3
            
            assert x.shape[0]==numberParameters, 'Expected ' + str(numberParameters) + ' parameters, got ' + str(x.shape[0]) 
    
    def forward(self, _tranformParams):
        """
        _tranformParams: <class 'torch.Tensor'>
            fullMode: (batchSize, 3, 4) tensor 
            else      (batchSize, 4, 4) tensor 
        """
        transformParams = _tranformParams
        batchSize = transformParams.dim()
        if self.fullMode:
            self.output = transformParams.view(batchSize, 3, 4)
        else:
            completeTransformation = torch.zeros(batchSize,4,4)
            
            completeTransformation[:, 0, 0] += 1
            completeTransformation[:, 1, 1] += 1
            completeTransformation[:, 2, 2] += 1
            completeTransformation[:, 3, 3] += 1   
            
            transformationBuffer = torch.Tensor(batchSize,4,4)
            
            paramIndex = 0
            if self.useRotation:
                omega_x = transformParams[:,paramIndex,:]
                omega_y = transformParams[:,paramIndex+1,:]
                omega_z = transformParams[:,paramIndex+2,:]
                
                paramIndex = paramIndex + 3
                omega_skew = torch.zeros(batchSize,4,4)
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
                sin_theta = torch.sin(theta)
                sin_theta_div_theta = torch.cdiv(sin_theta,theta)
                
                one_minus_cos_theta = torch.ones(theta.size()) - torch.cos(theta)
                one_minus_cos_div_theta_sqr = torch.cdiv(one_minus_cos_theta,theta_sqr)
                sin_theta_div_theta_tensor  = torch.ones(omega_skew.size())
                one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew.size())
                for b in range(batchSize):
                    if theta_sqr[b] > self.threshold:
                        sin_theta_div_theta_tensor[b] = sin_theta_div_theta[b]
                        one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr[b]
                    else:
                        sin_theta_div_theta_tensor[b]  = 1
                        one_minus_cos_div_theta_sqr_tensor[b] = 0
                
                completeTransformation = completeTransformation +\
                    torch.mul(sin_theta_div_theta_tensor,omega_skew) +\
                    torch.mul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)
                    
            self.rotationOutput = completeTransformation[:, 0:3, 0:3].clone()
            if self.useScale:
                paramIndex = paramIndex + 1
                transformationBuffer = torch.zeros(batchSize,4,4)
                transformationBuffer[:, 0, 0] = scaleFactors
                transformationBuffer[:, 1, 1] = scaleFactors
                transformationBuffer[:, 2, 2] = 1
                completeTransformation = \
                    torch.bmm(completeTransformation, transformationBuffer)
            
            self.scaleOutput = completeTransformation[:, 0:3, 0:3].clone()
            
            if self.useTranslation:
                txs = transformParams[:,paramIndex,:]
                tys = transformParams[:,paramIndex+1,:]
                tzs = transformParams[:,paramIndex+2,:]
                
                transformationBuffer = torch.zeros(batchSize,4,4)
                transformationBuffer[:, 0, 0] = 1
                transformationBuffer[:, 1, 1] = 1
                transformationBuffer[:, 2, 2] = 1
                transformationBuffer[:, 3, 3] = 1
                
                transformationBuffer[:, 0, 3] = txs
                transformationBuffer[:, 1, 3] = tys
                transformationBuffer[:, 2, 3] = tzs
                
                completeTransformation = \
                    torch.bmm(completeTransformation, transformationBuffer)
            self.output = completeTransformation[:, 0:3, :]
        if _transformParams.dim()==1: # total dimension
            self.output = self.output[0, :, :]
        return self.output
                
                
                
    def updateGradInput(self, _tranformParams, _gradParams):
        
        if _tranformParams.dim() == 1:
            transformParams = addOuterDim(_tranformParams)
            gradParams = addOuterDim(_gradParams).clone()
        else:
            transformParams = _tranformParams
            gradParams = _gradParams.clone()
            
        batchSize = transformParams.size()[0]#size(1)
        
        if self.fullMode:
            self.gradInput = gradParams.view(batchSize, 6)
        else:
            paramIndex = transformParams.size()[1] - 1#:size(2)
            self.gradInput = self.gradInput.view(transformParams.size())
            
            if self.useTranslation:
                gradInputTranslationParams = self.gradInput[:, paramIndex-3, paramIndex-1]#????
                tParams = torch.Tensor(batchSize, 1, 3)
                """
                The dimension of transformParams here should be batchx3x4
                """
                tParams[:, 0, :] = transformParams[:, paramIndex-2, :]
                tParams[:, 1, :] = transformParams[:, paramIndex-1, :]
                tParams[:, 2, :] = transformParams[:, paramIndex, :]
                paramIndex = paramIndex - 3#????
                
                selectedOutput = self.scaleOutput
                selectedGradParams = torch.t(gradParams[:,:,3])
                
                gradInputTranslationParams = (torch.bmm(selectedGradParams, selectedOutput)).clone()
                #gradientCorrection = torch.bmm(torch.transpose(selectedGradParams,2,1), tParams) #????
            if self.useScale:
                gradInputScaleparams = self.gradInput[:, paramIndex, :]
                sParams = transformParams[:,:,paramIndex] #????
                paramIndex = paramIndex-1
                
                selectedOutput = self.rotationOutput
                selectedGradParams = gradParams[:, 0:2, 0:2]
                x = torch.mul(selectedOutput, selectedGradParams)
                gradInputScaleparams = torch.sum(torch.sum(x, dim=1), dim=1)
                
                gradParams[:, 0, 0] = torch.mul(gradParams[:, 0, 0], sParams)
                gradParams[:, 0, 1] = torch.mul(gradParams[:, 0, 1], sParams)
                gradParams[:, 1, 0] = torch.mul(gradParams[:, 1, 0], sParams)
                gradParams[:, 1, 1] = torch.mul(gradParams[:, 1, 1], sParams)
                
            if self.useRotation:
                rotationDerivative = torch.zeros(batchSize, 3, 3)
                gradInputRotationParams = self.gradInput:narrow[:,0,:]
                    
                rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,0, self.threshold)
                selectedGradParams = gradParams[:, 0:3, 0:3]
                x = torch.mul(rotationDerivative, selectedGradParams)
                gradInputRotationParams = torch.sum(torch.sum(x, dim=1), dim=1)
                
                rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,1, self.threshold)
                gradInputRotationParams = self.gradInput[:, 1, :]
                
                x = torch.mul(rotationDerivative, selectedGradParams)
                gradInputRotationParams = torch.sum(torch.sum(x, dim=1), dim=1)
                
                rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,2, self.threshold)
                gradInputRotationParams = self.gradInput[:, 2, :]
                x = torch.mul(rotationDerivative, selectedGradParams)
                gradInputRotationParams = torch.sum(torch.sum(x, dim=1), dim=1)
        if _tranformParams.dim()==1:
            self.gradInput = self.gradInput[0, :, :]
        return self.gradInput
                
                        
    def __addOuterDim(t):
        sizes = t.shape
        newsizes = torch.LongStorage( len(sizes)+1 )
        newsizes[0] = 1
        for i in range(len(sizes)):
            newsizes[i+1] = sizes[i]

        return t.view(newsizes)
        
    def __dR_by_dvi(transparams, RotMats, which_vi, threshold):
        omega_x = transparams[:, 0, :]
        omega_y = transparams[:, 1, :]
        omega_z = transparams[:, 2, :]
        
        omega_skew = torch.zeros(RotMats.size())
        omega_skew[:, 1, 0] = omega_z.clone()
        omega_skew[:, 2, 0] = -1 * omega_y
                  
        omega_skew[:, 0, 1] = -1 * omega_z
        omega_skew[:, 2, 1] = omega_x.clone()
                  
        omega_skew[:, 0, 2] = omega_y.clone()
        omega_skew[:, 1, 2] = -1 * omega_x

        Id_minus_R_ei = torch.zeros(RotMats.size()[0],RotMats.size()[1],1)
        Id_minus_R_ei[:, which_vi, :] += 1
        
        I = torch.zeros(RotMats.size()[0], RotMats.size()[1], RotMats.size()[2])
        assert RotMats.size()[1] == 3
        assert RotMats.size()[2] == 3
        
        I[:, 0, 0] += 1
        I[:, 1, 1] += 1
        I[:, 2, 2] += 1
        
        Id_minus_R_ei = torch.bmm(torch.add(I,-RotMats), Id_minus_R_ei)
        
        #--- cross product 
        v_cross_Id_minus_R_ei = torch.bmm(omega_skew,Id_minus_R_ei)
        cross_x = v_cross_Id_minus_R_ei[:, 0, :]
        cross_y = v_cross_Id_minus_R_ei[:, 1, :]
        cross_z = v_cross_Id_minus_R_ei[:, 2, :]
        
        vcross = torch.zeros(RotMats.size())
        vcross[:, 1, 0] = cross_z.clone()
        vcross[:, 2, 0] = -1 * cross_y
                  
        vcross[:, 0, 1] = -1 * cross_z
        vcross[:, 2, 1] = cross_x.clone()
                  
        vcross[:, 0, 2] = cross_y.clone()
        vcross[:, 1, 2] = -1 * cross_x
        
        omega_mag = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)
        omega_selected = transparams[:, which_vi, :]
        
        for b in range(omega_skew.size()[0]):
            if  omega_mag[b] > threshold:
                v_i = omega_selected[b]
                omega_skew[b] = omega_skew[b].mul(v_i) + vcross[b]
                omega_skew[b].div(omega_mag[b])
            else:
                e_i = torch.zeros(3,1)
                e_i[which_vi, :] = 1
                
                eMat = torch.zeros(3,3)
                """
                [a]x = ( 0  -a3  a2
                        a3   0 -a1
                        -a2  a1  0 )
                """
                eMat[0][1] = -e_i[2]
                eMat[0][2] =  e_i[1]	
                
                eMat[1][0] =  e_i[2]
                eMat[1][2] = -e_i[0]
                
                eMat[2][0] = -e_i[1]
                eMat[2][1] =  e_i[0]
                
                omega_skew[b] = eMat 
                
        return torch.bmm(omega_skew, RotMats)










